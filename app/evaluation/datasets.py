from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from app.core.exceptions import EvaluationError
from app.schemas.evaluation import EvaluationCase

_CASES = TypeAdapter(list[EvaluationCase])


def load_cases(path: Path) -> list[EvaluationCase]:
    try:
        cases = _CASES.validate_json(path.read_bytes())
    except (OSError, ValidationError):
        raise EvaluationError(f"Could not load evaluation cases from {path}.") from None
    if not cases:
        raise EvaluationError("The evaluation dataset must contain at least one case.")
    if len({case.id for case in cases}) != len(cases):
        raise EvaluationError("Evaluation case IDs must be unique.")
    return cases
