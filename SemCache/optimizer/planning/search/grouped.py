"""Reference-controller adapter for the shared native group pruning policy."""


def grouped_order(problem, states, prediction, config):
    from planning.kernels import _search_core

    return _search_core.prune_structure_groups(
        problem.native_problem,
        [state.native_handle for state in states],
        prediction,
        maximum_groups=config.max_structure_groups,
        per_group=config.max_implementations_per_group,
    )["order"]
