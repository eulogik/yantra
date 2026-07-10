#!/usr/bin/env python3
"""Tool Execution Verifier — schema validation + (optional) mock execution.

This is the verifier that powers EG-OPD: given a predicted call and the tool's
JSON schema, it returns a reward r in {0, 0.5, 1}:
  0   = schema-invalid (missing required / wrong type / unknown enum)
  0.5 = schema-valid but value looks wrong (heuristic)
  1   = schema-valid AND (executed in mock registry with success)
"""
from __future__ import annotations

import json
from typing import Any


class ToolVerifier:
    def __init__(self, mock_registry: dict | None = None):
        # mock_registry: {tool_name: callable(**args) -> bool/value}
        self.registry = mock_registry or {}

    def validate_schema(self, args: dict, schema: dict) -> tuple[bool, str]:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for r in required:
            if r not in args or args[r] in (None, ""):
                return False, f"MissingRequiredArgument: {r}"
        for k, v in args.items():
            spec = props.get(k)
            if spec is None:
                continue
            t = spec.get("type")
            if t == "integer" and not isinstance(v, int):
                return False, f"TypeError: {k} expected int"
            if t == "number" and not isinstance(v, (int, float)):
                return False, f"TypeError: {k} expected number"
            if t == "string" and not isinstance(v, str):
                return False, f"TypeError: {k} expected string"
            if "enum" in spec and v not in spec["enum"]:
                return False, f"EnumError: {k} not in {spec['enum']}"
        return True, ""

    def reward(self, tool_name: str, args: dict, schema: dict) -> tuple[float, str]:
        ok, err = self.validate_schema(args, schema)
        if not ok:
            return 0.0, err
        fn = self.registry.get(tool_name)
        if fn is None:
            return 0.5, "schema-valid (no mock executor)"
        try:
            res = fn(**args)
            return 1.0, f"executed: {res}"
        except Exception as e:  # noqa: BLE001
            return 0.5, f"exec-error: {e}"


if __name__ == "__main__":
    schema = {"properties": {"city": {"type": "string"}}, "required": ["city"]}
    v = ToolVerifier()
    print(v.reward("weather.get_forecast", {"city": "Paris"}, schema))
    print(v.reward("weather.get_forecast", {}, schema))
