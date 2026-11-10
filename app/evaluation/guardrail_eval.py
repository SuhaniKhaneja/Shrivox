"""
Offline evaluation of the deterministic guardrails: no API key, no network, no
database -- so it runs in CI on every change and in seconds.

    python -m app.evaluation.guardrail_eval

Reads app/evaluation/guardrail_cases.json. READ ITS "_about" FIRST: the cases are
self-written, so the numbers here are a regression signal and a way to compare
guardrail versions, not an estimate of production accuracy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.guardrails import check_input, check_output

CASES_PATH = Path(__file__).with_name("guardrail_cases.json")


@dataclass
class InputReport:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0


@dataclass
class OutputReport:
    correct: int = 0
    total: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0


def load_cases(path: Path = CASES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_input(cases: list[dict]) -> InputReport:
    report = InputReport()
    for case in cases:
        got, want = not check_input(case["text"]).safe, case["expect"] == "escalate"
        if got and want:
            report.tp += 1
        elif got:
            report.fp += 1
        elif want:
            report.fn += 1
        else:
            report.tn += 1
        if got != want:
            report.errors.append({"kind": "false_positive" if got else "false_negative", **case})
    return report


def evaluate_output(cases: list[dict]) -> OutputReport:
    report = OutputReport(total=len(cases))
    for case in cases:
        history = [m["content"] for m in case.get("history", [])]
        result = check_output(case["response"], grounded_in_kb=bool(case["context"]),
                              context=case["context"], conversation=[*history, case["user"]])
        got = result.flag or "safe"
        if got == case["expect"]:
            report.correct += 1
        else:
            report.errors.append({"got": got, **case})
    return report


def main() -> None:
    cases = load_cases()
    inp, out = evaluate_input(cases["input"]), evaluate_output(cases["output"])
    print(f"input guard : n={len(cases['input'])}  precision={inp.precision:.2f}  recall={inp.recall:.2f}  "
          f"(TP={inp.tp} FP={inp.fp} FN={inp.fn} TN={inp.tn})")
    print(f"output guard: n={out.total}  accuracy={out.accuracy:.2f}")
    for error in [*inp.errors, *out.errors]:
        print("  MISS", error)
    gaps = evaluate_input(cases.get("known_gaps", []))
    print(f"known gaps  : {len(cases.get('known_gaps', []))} cases the rules are KNOWN to get wrong "
          f"({gaps.fp + gaps.fn} still wrong) -- listed so they are not forgotten:")
    for gap in cases.get("known_gaps", []):
        print("  GAP", gap["text"], "--", gap["why"])


if __name__ == "__main__":
    main()
