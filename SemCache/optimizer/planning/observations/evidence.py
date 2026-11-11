"""Read-only JSON evidence: safe sharing without copying entire depth tables."""


def _readonly(*args, **kwargs):
    raise TypeError("Published sampling evidence is read-only")


class _EvidenceDict(dict):
    __slots__ = ("_initialized",)

    def __init__(self, *args, **kwargs):
        if hasattr(self, "_initialized"):
            _readonly()
        super().__init__(*args, **kwargs)
        self._initialized = True

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = _readonly

    def __deepcopy__(self, memo):
        return self


class _EvidenceList(list):
    __slots__ = ("_initialized",)

    def __init__(self, *args, **kwargs):
        if hasattr(self, "_initialized"):
            _readonly()
        super().__init__(*args, **kwargs)
        self._initialized = True

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = (
        reverse
    ) = sort = __iadd__ = __imul__ = _readonly

    def __deepcopy__(self, memo):
        return self


def freeze_evidence(value, memo=None):
    """Freeze an acyclic JSON bank once, preserving existing container aliases."""
    if type(value) not in (dict, list, tuple):
        return value
    if memo is None:
        memo = {}
    identity = id(value)
    if identity not in memo:
        memo[identity] = (
            _EvidenceDict(
                {key: freeze_evidence(item, memo) for key, item in value.items()}
            )
            if type(value) is dict
            else (
                tuple(freeze_evidence(item, memo) for item in value)
                if type(value) is tuple
                else _EvidenceList(freeze_evidence(item, memo) for item in value)
            )
        )
    return memo[identity]


def is_frozen_evidence(value):
    """Prove deep immutability before retaining a sample by identity."""
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return True
    if kind is _EvidenceDict:
        return all(
            is_frozen_evidence(key) and is_frozen_evidence(item)
            for key, item in value.items()
        )
    if kind in (_EvidenceList, tuple):
        return all(is_frozen_evidence(item) for item in value)
    return False
