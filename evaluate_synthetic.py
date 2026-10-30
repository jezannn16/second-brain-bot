"""Reproducible, offline retrieval and fake-provider contract evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag import ProviderAnswer
from second_brain import add_document, answer, connect, evaluate_retrieval


class _FakeProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def generate(self, question: str, evidence: list[dict[str, object]]) -> ProviderAnswer:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]


def evaluate_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("documents"), list) or not isinstance(dataset.get("cases"), list):
        raise TypeError("Dataset must contain document and case lists")
    if not dataset.get("synthetic"):
        raise ValueError("This local evaluator accepts synthetic datasets only")

    db = connect(":memory:")
    try:
        source_ids: dict[str, int] = {}
        for document in dataset["documents"]:
            if not isinstance(document, dict) or not all(isinstance(document.get(k), str) for k in ("source", "title", "content")):
                raise ValueError("Every dataset document requires source, title, and content strings")
            source = document["source"]
            if source in source_ids:
                raise ValueError(f"Duplicate dataset source: {source}")
            source_ids[source], _ = add_document(db, source, document["title"], document["content"])

        cases = dataset["cases"]
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                raise TypeError("Every evaluation case requires a string id")
            expected = case.get("expected_sources")
            citations = case.get("expected_citations")
            if (
                not isinstance(expected, list)
                or any(source not in source_ids for source in expected)
                or citations != expected
            ):
                raise ValueError(f"Case {case['id']} has invalid expected_sources/expected_citations")
        retrieval = evaluate_retrieval(db, cases)

        backup_id = source_ids["backup-policy.md"]
        valid = _FakeProvider(ProviderAnswer("Synthetic answer from the backup policy.", [backup_id]))
        valid_result = answer(db, "Where should encrypted backups be stored offline?", provider=valid)

        invalid_citation = _FakeProvider(ProviderAnswer("Synthetic answer.", [999_999_999]))
        invalid_result = answer(db, "Where should encrypted backups be stored offline?", provider=invalid_citation)

        no_evidence = _FakeProvider(ProviderAnswer("This must not be used.", []))
        abstention_result = answer(db, "Quasars xylophones teapots", provider=no_evidence)

        # An in-schema response with a retrieved source citation can still make an unsupported claim.
        # This probe documents the limit of validation; it is not counted as a model-quality score.
        unsupported = _FakeProvider(ProviderAnswer("Backups are required to be stored online.", [backup_id]))
        unsupported_result = answer(db, "Where should encrypted backups be stored offline?", provider=unsupported)

        contract = {
            "fake_provider_scenarios": 4,
            "valid_structured_answer_and_retrieved_citation_accepted": (
                valid.calls == 1 and valid_result["mode"] == "llm"
                and [item["document_id"] for item in valid_result["citations"]] == [backup_id]
            ),
            "citation_outside_retrieval_rejected": (
                invalid_citation.calls == 1 and invalid_result["mode"] == "retrieval-fallback"
            ),
            "no_evidence_abstains_without_provider_call": (
                no_evidence.calls == 0 and abstention_result["mode"] == "abstained"
                and not abstention_result["citations"]
            ),
            "unsupported_claim_with_valid_citation_is_not_detected": (
                unsupported.calls == 1 and unsupported_result["mode"] == "llm"
            ),
            "real_model_accuracy": None,
            "accuracy_note": (
                "No real model was called. Fake-provider probes test schema/citation boundaries only; "
                "valid citations do not establish that claims are entailed by cited text."
            ),
        }
        contract["all_contract_checks_passed"] = all(
            value for key, value in contract.items()
            if key not in {"fake_provider_scenarios", "real_model_accuracy", "accuracy_note"}
        )
        return {
            "dataset": {"name": dataset.get("name"), "synthetic": True, "documents": len(source_ids), "cases": len(cases)},
            "retrieval": retrieval,
            "provider_contract_probes": contract,
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline synthetic Second Brain RAG evaluation")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "eval" / "synthetic_rag.json")
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(args.dataset), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
