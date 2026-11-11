"""Expose physical answers and cached answers for logical row accounting only.

Physical-request consumers keep using the original semantic_row events; these
expanded views must never be re-ingested as KV history or model latency samples.
"""


def logical_rows(events):
    sources = {}
    for event in events:
        kind = event.get("event")
        if kind == "semantic_row":
            if event.get("parse_status") == "validated":
                if event.get("request_seq") is not None:
                    sources[event["request_seq"]] = event
                yield event
        elif kind == "semantic_result_cache_hit":
            if event.get("parse_status") != "validated":
                raise ValueError("Result cache hit must contain a validated result")
            physical_fields = {
                "request_seq",
                "submit_ns",
                "response_ns",
                "server_start_ns",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
            }
            if physical_fields & event.keys():
                raise ValueError(
                    "Result cache hit cannot contain physical request telemetry"
                )
            boundary = event.get("after_request_seq")
            if type(boundary) is not int or boundary < 0:
                raise ValueError(
                    "Result cache hit requires a physical request boundary"
                )
            source = event.get("cache_source_seq")
            if source is not None:
                if type(source) is not int or not 0 < source <= boundary:
                    raise ValueError("Result cache hit has an invalid source identity")
                if source in sources:
                    raise ValueError("Result cache source identity is not unique")
                sources[source] = event
            yield event
        elif kind == "semantic_result_reuse":
            source = sources.get(event.get("source_request_seq"))
            if source is None or any(
                event.get(key) != source.get(key)
                for key in ("query_id", "model_id", "operator_kind")
            ):
                raise ValueError("Result reuse has no compatible preceding request")
            if event.get("parse_status") != "validated":
                raise ValueError("Result reuse must reference a validated result")
            if event.get("sampling_batch_size", 1) != source.get(
                "sampling_batch_size", 1
            ):
                raise ValueError("Result reuse changed the output batch contract")
            restored = dict(source, **event)
            restored.pop("request_seq", None)
            if event.get("sampling_batch_size", 1) == 1:
                mapping = event.get("display_to_input")
                old_mapping = source["display_to_input"]
                if not isinstance(mapping, list) or sorted(mapping) != list(
                    range(len(old_mapping))
                ):
                    raise ValueError(
                        "Result reuse has an invalid input display mapping"
                    )
                values = [None] * len(mapping)
                for display, original in enumerate(mapping):
                    values[original] = source["input_values"][old_mapping[display]]
                restored["input_values"] = values
                if restored["emitted"] != source["emitted"]:
                    raise ValueError("Result reuse changed the validated decision")
            yield restored
