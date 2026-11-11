"""Current query-design acceptance policy; observed row counts are never clipped."""

TARGET_MINIMUM_ROWS = 50
MINIMUM_ROWS = 30
MAXIMUM_ROWS = 300


def row_verdict(rows):
    if rows < MINIMUM_ROWS:
        label = "结果明显少于50行"
    elif rows > MAXIMUM_ROWS:
        label = f"结果超过{MAXIMUM_ROWS}行"
    else:
        label = "接近50行，符合目标" if rows < TARGET_MINIMUM_ROWS else "符合目标"
    return {
        "target_minimum_rows": TARGET_MINIMUM_ROWS,
        "acceptable_minimum_rows": MINIMUM_ROWS,
        "maximum_rows": MAXIMUM_ROWS,
        "rows": rows,
        "classification": label,
        "status": "passed"
        if MINIMUM_ROWS <= rows <= MAXIMUM_ROWS
        else "needs_adjustment",
    }
