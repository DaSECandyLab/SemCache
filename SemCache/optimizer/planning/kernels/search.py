"""Run Beam on frozen operator statistics, prepared once before expansion."""


def run_native(search, problem):
    """Search shared operator statistics; online sampling is prepared once in C++."""
    from .resident import run_resident
    from .sampling import prepare_profile

    if problem.sampler is None:
        problem.sampler = prepare_profile(problem)
        try:
            return run_resident(search, problem)
        finally:
            problem.sampler = None
    if not hasattr(problem.sampler, "native_sampler"):
        raise ValueError(
            "Beam search requires frozen operator statistics or the independent "
            "statistics preparation backend; candidate row samplers are unsupported"
        )
    return run_resident(search, problem)
