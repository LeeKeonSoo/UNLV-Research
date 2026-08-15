from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.cruxeval_windows_runner import (
    install_cruxeval_windows_compatibility,
)


def test_windows_runner_preserves_canonical_pass_and_wrong_answer_fail() -> None:
    install_cruxeval_windows_compatibility()
    from utils_general import evaluate_score

    code = """def f(a, b, c):
    result = {}
    for item in a, b, c:
        result.update(dict.fromkeys(item))
    return result
"""
    reference = (code, "(1, ), (1, ), (1, 2)", "{1: None, 2: None}")
    passing = evaluate_score((["{1: None, 2: None}"], reference, "output"))
    failing = evaluate_score((["{1: None}"], reference, "output"))
    assert passing == [True]
    assert failing == [False]


if __name__ == "__main__":
    test_windows_runner_preserves_canonical_pass_and_wrong_answer_fail()
    print("CRUXEval Windows runner pass/fail contract passed")
