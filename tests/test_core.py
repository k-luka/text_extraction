import json

import numpy as np
import pytest

from text_extraction.io import append_jsonl, iter_records, read_jsonl, write_jsonl
from text_extraction.structured import build_user_prompt, load_schema, parse_json_object


def test_json_parser_accepts_plain_and_fenced_objects():
    expected = {"value": 1}
    assert parse_json_object(json.dumps(expected)) == expected
    assert parse_json_object('```json\n{"value": 1}\n```') == expected
    assert parse_json_object('answer: {"value": 1}') == expected


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    append_jsonl(path, [{"note_id": "1", "text": "hello"}])
    assert read_jsonl(path) == [{"note_id": "1", "text": "hello"}]
    write_jsonl(path, [{"note_id": "2", "text": "replacement"}])
    assert read_jsonl(path) == [{"note_id": "2", "text": "replacement"}]


def test_csv_streaming_and_limit(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("note_id,NOTE_TEXT\n1,first\n2,second\n", encoding="utf-8")
    rows = list(iter_records(path, limit=1))
    assert rows == [{"note_id": "1", "NOTE_TEXT": "first"}]


def test_example_schema_is_valid():
    schema = load_schema("schemas/discharge_renal.json")
    assert schema["additionalProperties"] is False


def test_user_prompt_includes_task_rules_and_delimits_note():
    schema = {
        "title": "Test extraction",
        "description": "Do not infer missing values.",
    }
    prompt = build_user_prompt("Patient text", schema)
    assert "Do not infer missing values." in prompt
    assert "<clinical_note>\nPatient text\n</clinical_note>" in prompt
