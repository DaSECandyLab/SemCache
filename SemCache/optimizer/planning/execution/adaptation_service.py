"""Isolate runtime RPCs from DuckDB execution and nonblocking server teardown."""

import copy
import json
import os
import socketserver
import threading
import time

from optimizer_server import OptimizerServer

from planning.execution.adaptation import ChunkAdaptationSession


class RuntimeOptimizerServer(socketserver.ThreadingMixIn, OptimizerServer):
    daemon_threads = True
    block_on_close = False


class RuntimeExecutionPolicy:
    def __init__(
        self,
        initial,
        profile,
        model,
        config,
        settings,
        log_path=None,
        initial_references=None,
    ):
        self.initial = initial
        self.profile, self.model, self.config, self.settings = (
            profile,
            copy.deepcopy(model),
            config,
            settings,
        )
        self.adapter = None
        self.lock = threading.Lock()
        self.closed = False
        self.log_path = log_path
        self.initial_references = initial_references

    @property
    def query(self):
        return self.initial.query

    def optimize(self, request):
        if request.get("kind") != "RuntimeRequest":
            spec = self.initial.optimize(request)
            self.adapter = ChunkAdaptationSession(
                request,
                self.profile,
                self.model,
                self.config,
                self.settings,
                initial_references=self.initial_references,
            )
            return spec
        thread = threading.current_thread()
        thread.name = "semcache-runtime-optimizer"
        native_thread = threading.get_native_id()
        if hasattr(os, "setpriority"):
            os.setpriority(os.PRIO_PROCESS, native_thread, 5)
        started = time.monotonic()
        with self.lock:
            if self.closed or self.adapter is None:
                return {
                    "version": 1,
                    "kind": "RuntimeUpdate",
                    "query_id": request["query_id"],
                    "updates": [],
                    "diagnostics": {"status": "query_closed"},
                }
            try:
                updates, diagnostics = self.adapter.check(request)
                diagnostics.update(
                    background_thread=thread.name,
                    native_thread_id=native_thread,
                    optimization_seconds=time.monotonic() - started,
                    value_model_device=str(self.adapter.device),
                )
                if self.log_path:
                    with open(self.log_path, "a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "completed_chunks": request["completed_chunks"],
                                    "diagnostics": diagnostics,
                                    "updates": updates,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                return {
                    "version": 1,
                    "kind": "RuntimeUpdate",
                    "query_id": request["query_id"],
                    "updates": updates,
                    "diagnostics": diagnostics,
                }
            finally:
                if self.closed:
                    self.adapter.close()
                    self.adapter = None

    def close(self):
        self.closed = True
        if self.lock.acquire(blocking=False):
            try:
                if self.adapter is not None:
                    self.adapter.close()
                    self.adapter = None
            finally:
                self.lock.release()
