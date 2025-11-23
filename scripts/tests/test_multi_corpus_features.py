from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.core.core_evaluate import compute_basic_metrics, group_indices_by_field
from scripts.core.core_prepare import write_tsv
from scripts.core.core_utils import resolve_profile_base


def test_resolve_single_profile_defaults():
    params = resolve_profile_base("ideo_quick")
    assert params["merge_mode"] == "single"
    assert params["dataset_id"] == params["corpus_id"]
    assert params["corpora"] and len(params["corpora"]) == 1


def test_resolve_multi_profile():
    params = resolve_profile_base("multi_demo")
    assert len(params["corpora"]) == 2
    assert params["dataset_id"] == "web1_web2"
    assert params["source_field"] == "corpus_id"
    assert params["corpus_id"] == "web1_web2"


def test_write_tsv_includes_source_column(tmp_path: Path):
    docs = [
        {
            "id": "doc1",
            "label": "a",
            "label_raw": "a",
            "text": "hello",
            "modality": "web",
            "meta": {"corpus_id": "web1"},
        },
        {
            "id": "doc2",
            "label": "b",
            "label_raw": "b",
            "text": "world",
            "modality": "web",
            "meta": {"corpus_id": "web2"},
        },
    ]
    tsv_path = tmp_path / "sample.tsv"
    write_tsv(str(tsv_path), docs, source_field="corpus_id")

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert reader.fieldnames and "corpus_id" in reader.fieldnames
    assert rows[0]["corpus_id"] == "web1"
    assert rows[1]["corpus_id"] == "web2"


def test_group_metrics_by_field():
    rows = [
        {"corpus_id": "web1"},
        {"corpus_id": "web1"},
        {"corpus_id": "web2"},
    ]
    y_true = ["a", "b", "a"]
    y_pred = ["a", "b", "b"]

    groups = group_indices_by_field(rows, "corpus_id")
    assert set(groups.keys()) == {"web1", "web2"}

    metrics_by = {
        val: compute_basic_metrics([y_true[i] for i in idxs], [y_pred[i] for i in idxs])
        for val, idxs in groups.items()
    }

    assert set(metrics_by.keys()) == {"web1", "web2"}
    assert metrics_by["web1"]["accuracy"] == 1.0
    assert metrics_by["web2"]["accuracy"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
