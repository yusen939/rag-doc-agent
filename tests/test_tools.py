import pytest

from app.services.tools import ToolExecutionError, safe_calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+2*3", 7.0),
        ("(120+30)*0.8", 120.0),
        ("10/4", 2.5),
        ("2**3", 8.0),
    ],
)
def test_safe_calculate(expression, expected):
    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "open('/etc/passwd')",
        "1/0",
        "2**100",
        "",
    ],
)
def test_safe_calculate_rejects_unsafe_or_invalid_expressions(expression):
    with pytest.raises(ToolExecutionError):
        safe_calculate(expression)
