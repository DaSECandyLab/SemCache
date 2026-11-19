"""Relative deviation of scalar aggregate outputs, independent of row F1."""

from decimal import Decimal, InvalidOperation


def aggregate_deviation(actual, reference):
    for result in (actual, reference):
        if (
            len(result["columns"]) != 1
            or len(result["rows"]) != 1
            or len(result["rows"][0]) != 1
        ):
            raise ValueError(
                "Relative deviation requires a one-row, one-column aggregate"
            )
    value, expected = actual["rows"][0][0], reference["rows"][0][0]
    output = {
        "aggregate_value": value,
        "reference_value": expected,
        "absolute_deviation": None,
        "relative_deviation": None,
        "deviation_infinite": False,
        "aggregate_equal": value == expected,
        "scoring_status": "aggregate_scored",
        "deviation_status": "numeric",
    }
    if value is None or expected is None:
        output["deviation_status"] = (
            "both_null" if value is expected else "null_mismatch"
        )
        return output
    if isinstance(value, bool) or isinstance(expected, bool):
        raise TypeError("Boolean values cannot be scored as numeric aggregates")
    try:
        value, expected = Decimal(str(value)), Decimal(str(expected))
    except InvalidOperation as error:
        raise ValueError("Aggregate output must be numeric or SQL NULL") from error
    if not value.is_finite() or not expected.is_finite():
        raise ValueError("Aggregate output must be finite")
    difference = abs(value - expected)
    output.update(
        absolute_deviation=float(difference), aggregate_equal=value == expected
    )
    if expected == 0:
        output["deviation_status"] = "zero_reference"
        output["deviation_infinite"] = difference != 0
        output["relative_deviation"] = None if difference else 0.0
    else:
        output["relative_deviation"] = float(difference / abs(expected))
    return output
