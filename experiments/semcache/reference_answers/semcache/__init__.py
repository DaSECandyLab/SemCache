"""Collect training references using SemCache, a teacher and fixed GOTDF."""

from .cli import main as main
from .collector import (
    PROTOCOL as PROTOCOL,
)
from .collector import (
    collect as collect,
)
from .collector import (
    configured_teacher as configured_teacher,
)
from .collector import (
    timestamp as timestamp,
)
from .collector import (
    validate_saved_answer as validate_saved_answer,
)
