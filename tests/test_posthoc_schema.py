"""Schema compatibility for append-only posthoc redo metadata."""

import pytest

from rsibench.records import validate


def _row():
    return {"task_id": "t", "round": 1, "commit": "abc", "status": "ok", "score": 0.5}


def test_posthoc_schema_keeps_existing_rows_valid():
    assert validate("posthoc", _row()) == []


@pytest.mark.parametrize("supersedes", [None, "2026-10-11T12:00:00+00:00"])
def test_posthoc_schema_accepts_redo_metadata(supersedes):
    row = {**_row(), "redo": True, "supersedes": supersedes,
           "redo_reason": "Earlier model calls failed"}
    assert validate("posthoc", row) == []


@pytest.mark.parametrize("field,value", [("redo", "true"), ("supersedes", 1),
                                         ("redo_reason", None)])
def test_posthoc_schema_checks_redo_metadata_types(field, value):
    errors = validate("posthoc", {**_row(), field: value})
    assert len(errors) == 1
    assert errors[0].startswith(field + ":")
