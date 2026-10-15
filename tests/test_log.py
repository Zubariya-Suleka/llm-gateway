import json
import logging

from llm_gateway.log import JsonFormatter


def _format(**extra) -> dict:
    record = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "hello", **extra})
    return json.loads(JsonFormatter().format(record))


def test_basic_fields():
    line = _format()

    assert line["msg"] == "hello"
    assert line["level"] == "info"
    assert line["logger"] == "t"
    assert "ts" in line


def test_extra_fields_become_top_level_keys():
    line = _format(model="groq/llama-3.1-8b-instant", latency_ms=812)

    assert line["model"] == "groq/llama-3.1-8b-instant"
    assert line["latency_ms"] == 812
