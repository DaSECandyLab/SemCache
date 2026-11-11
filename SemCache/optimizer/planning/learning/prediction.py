"""One unconditioned time/quality pair per state; constraints stay in search."""


def paired_prediction(values):
    return {
        "scores": [(-t, a) for t, a in values],
        "times": [t for t, a in values],
        "accuracies": [a for t, a in values],
        "time_reasons": [None] * len(values),
    }


def completion_prediction(model, features):
    return paired_prediction(model.values_features(features))
