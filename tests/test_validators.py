"""Tests for agentops_fw.validators module."""
import pytest

from agentops_fw.validators import validate_json


class TestValidateJson:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        ok, err = validate_json({"name": "Alice"}, schema)
        assert ok is True
        assert err == ""

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        ok, err = validate_json({}, schema)
        assert ok is False
        assert "name" in err

    def test_wrong_type(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        ok, err = validate_json({"age": "not_int"}, schema)
        assert ok is False
        assert len(err) > 0

    def test_empty_schema_accepts_anything(self):
        ok, err = validate_json({"anything": True}, {})
        assert ok is True

    def test_array_schema(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        ok, _ = validate_json([1, 2, 3], schema)
        assert ok is True

        ok, err = validate_json([1, "two", 3], schema)
        assert ok is False

    def test_nested_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                }
            },
            "required": ["user"],
        }
        ok, _ = validate_json({"user": {"id": 42}}, schema)
        assert ok is True

        ok, _ = validate_json({"user": {"id": "not_int"}}, schema)
        assert ok is False
