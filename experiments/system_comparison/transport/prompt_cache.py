"""Exact completed-response reuse, scoped to one comparison configuration."""

import copy

from experiments.system_comparison.references.cache import TeacherCache, canonical_response


class PromptCache(TeacherCache):
    """Reuse storage/LRU while disabling the teacher verification policy."""

    def resolve(self, key, infer, normalize=lambda response: response):
        with self.key_lock(key):
            entry = self._read(key)
            info = {"key": key, "system": self.system, "hit": False,
                    "verification_calls": 0, "vote_calls": 0}
            if entry["confirmed"]:
                return copy.deepcopy(entry["responses"][0]), {**info, "hit": True}
            response = infer()
            # Truncated, empty and malformed provider responses are never reused.
            try:
                canonical = canonical_response(normalize(response))
            except (ValueError, KeyError, IndexError, TypeError):
                return response, {**info, "stored": False}
            self._write(key, {"responses": [copy.deepcopy(response)], "values": [canonical],
                              "confirmed": True, "winner": 0})
            return response, {**info, "stored": True}
