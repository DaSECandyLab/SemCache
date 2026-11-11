"""Prompt block orders shared by candidate search and model calibration."""

OPERATOR = "OPERATOR_SPECIFICATION"
TASK = "TASK_DESCRIPTION"
DATA = "DATA_BLOCK"

BLOCK_ORDERS = (
    (OPERATOR, TASK, DATA),
    (OPERATOR, DATA, TASK),
    (DATA, OPERATOR, TASK),
)

# Fixed one-hot positions preserve checkpoint coordinates; unused orders are excluded from candidate enumeration.
ENCODED_BLOCK_ORDERS = (
    (OPERATOR, TASK, DATA),
    (OPERATOR, DATA, TASK),
    (TASK, OPERATOR, DATA),
    (TASK, DATA, OPERATOR),
    (DATA, OPERATOR, TASK),
    (DATA, TASK, OPERATOR),
)
