"""Reference admission rules; native rank_layer implements the same contract."""


def rank_layer(prediction, preference, minimum=None, *, diagnostics=None):
    """Preserve both objectives; apply the accuracy target only at completion."""
    if preference not in ("fastest", "accurate"):
        raise ValueError("preference must be fastest or accurate")
    if diagnostics is not None:
        diagnostics.update(accuracy_threshold=None, accuracy_relaxations=0)
    return list(prediction["scores"]), []


def accuracy_time_order(prediction):
    """Order partial states by quality first, then predicted time.

    This is an intermediate Beam cutoff order only.  Accuracy constraints are
    applied to complete plans by the final selection step, never here.
    """
    times = prediction.get("times", [])
    accuracies = prediction.get("accuracies", [])

    def key(index):
        accuracy = accuracies[index]
        seconds = times[index]
        return (
            -(accuracy if accuracy is not None else -1.0),
            seconds if seconds is not None else float("inf"),
            index,
        )

    return sorted(range(len(prediction["scores"])), key=key)
