"""Extract dense document vectors with a Hugging Face transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import iter_records


def masked_mean(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    total = (last_hidden_state * mask).sum(dim=1)
    return total / mask.sum(dim=1).clamp(min=1e-9)


class TransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "mean",
        max_length: int | None = None,
        normalize: bool = True,
        trust_remote_code: bool = False,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        model_limit = int(getattr(self.tokenizer, "model_max_length", 512))
        self.max_length = max_length or min(model_limit, 8192)
        self.pooling = pooling
        self.normalize = normalize

    def _pool(self, outputs: Any, attention_mask: Any) -> Any:
        if self.pooling == "cls":
            vectors = outputs.last_hidden_state[:, 0]
        else:
            vectors = masked_mean(outputs.last_hidden_state, attention_mask)
        if self.normalize:
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with self.torch.inference_mode():
                outputs = self.model(**tokens)
                pooled = self._pool(outputs, tokens["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())
        if not vectors:
            hidden = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden), dtype=np.float32)
        return np.concatenate(vectors).astype(np.float32, copy=False)

    def encode_long(
        self,
        texts: Iterable[str],
        *,
        batch_size: int,
        chunk_tokens: int,
        chunk_overlap: int,
    ) -> np.ndarray:
        if chunk_overlap >= chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        max_content_tokens = self.max_length - self.tokenizer.num_special_tokens_to_add(
            pair=False
        )
        if chunk_tokens > max_content_tokens:
            raise ValueError(
                f"chunk_tokens ({chunk_tokens}) exceeds the model's usable window "
                f"({max_content_tokens}); lower --chunk-tokens or raise --max-length"
            )
        document_vectors: list[np.ndarray] = []
        stride = chunk_tokens - chunk_overlap
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = [
                self.tokenizer.decode(token_ids[i : i + chunk_tokens])
                for i in range(0, max(len(token_ids), 1), stride)
                if token_ids[i : i + chunk_tokens]
            ] or [""]
            chunk_vectors = self.encode(chunks, batch_size=batch_size)
            vector = chunk_vectors.mean(axis=0)
            if self.normalize:
                norm = np.linalg.norm(vector)
                vector = vector / max(norm, 1e-12)
            document_vectors.append(vector)
        return np.asarray(document_vectors, dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    embedder = TransformerEmbedder(
        args.model,
        pooling=args.pooling,
        max_length=args.max_length,
        normalize=args.normalize,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )
    ids_list: list[str] = []
    vector_batches: list[np.ndarray] = []
    record_batch: list[tuple[str, str]] = []

    def encode_batch(batch: list[tuple[str, str]]) -> None:
        if not batch:
            return
        ids_list.extend(item[0] for item in batch)
        texts = [item[1] for item in batch]
        if args.chunk_tokens:
            vectors = embedder.encode_long(
                texts,
                batch_size=args.batch_size,
                chunk_tokens=args.chunk_tokens,
                chunk_overlap=args.chunk_overlap,
            )
        else:
            vectors = embedder.encode(texts, batch_size=args.batch_size)
        vector_batches.append(vectors)

    records = iter_records(
        args.input,
        input_format=args.input_format,
        limit=args.limit,
    )
    for index, row in enumerate(records):
        record_batch.append(
            (str(row.get(args.id_field, index)), str(row.get(args.text_field, "")))
        )
        if len(record_batch) >= args.batch_size:
            encode_batch(record_batch)
            record_batch = []
    encode_batch(record_batch)

    ids = np.asarray(ids_list)
    if vector_batches:
        embeddings = np.concatenate(vector_batches).astype(np.float32, copy=False)
    else:
        hidden_size = int(getattr(embedder.model.config, "hidden_size", 0))
        embeddings = np.empty((0, hidden_size), dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "model": args.model,
            "pooling": args.pooling,
            "normalized": args.normalize,
            "max_length": embedder.max_length,
            "chunk_tokens": args.chunk_tokens,
            "chunk_overlap": args.chunk_overlap if args.chunk_tokens else None,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        output,
        embeddings=embeddings,
        ids=ids,
        metadata=np.asarray(metadata),
    )
    print(f"Saved {embeddings.shape} embeddings to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL or CSV")
    parser.add_argument("--output", required=True, help="Output .npz")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="note_id")
    parser.add_argument(
        "--input-format", choices=["auto", "csv", "jsonl"], default="auto"
    )
    parser.add_argument("--limit", type=int, help="Process only the first N input rows")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", help="For example: cuda, cuda:1, or cpu")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
