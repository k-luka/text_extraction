import json

import numpy as np
import pytest

from text_extraction.io import append_jsonl, read_jsonl
from text_extraction.structured import load_schema, parse_json_object


def test_json_parser_accepts_plain_and_fenced_objects():
    expected = {"value": 1}
    assert parse_json_object(json.dumps(expected)) == expected
    assert parse_json_object('```json\n{"value": 1}\n```') == expected
    assert parse_json_object('answer: {"value": 1}') == expected


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    append_jsonl(path, [{"note_id": "1", "text": "hello"}])
    assert read_jsonl(path) == [{"note_id": "1", "text": "hello"}]


def test_example_schema_is_valid():
    schema = load_schema("schemas/discharge_renal.json")
    assert schema["additionalProperties"] is False

