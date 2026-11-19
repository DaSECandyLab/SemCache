"""Collect the selected five-system query outputs with one external teacher."""

from .runner import execute_system as execute_system
from .runner import main as main
from .setup import (
    DATASETS as DATASETS,
)
from .setup import (
    SYSTEMS as SYSTEMS,
)
from .setup import (
    bind_plop_runtime as bind_plop_runtime,
)
from .setup import (
    initialize as initialize,
)
from .setup import (
    plop_runtime as plop_runtime,
)
from .setup import (
    read as read,
)
from .setup import (
    render as render,
)
from .setup import (
    selected_queries as selected_queries,
)
from .setup import (
    skipped as skipped,
)
from .setup import (
    teacher_model as teacher_model,
)
