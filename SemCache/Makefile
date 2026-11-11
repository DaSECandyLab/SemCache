.DEFAULT_GOAL := release
# A single Python module links the selected mode; serialize mode selection.
.NOTPARALLEL:
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PATH := $(ROOT).venv-tools/bin:$(PATH)
PLANNING_PYTHON ?= $(ROOT).venv-optimizer/bin/python
TOOLS_PYTHON ?= $(ROOT).venv-tools/bin/python
PLANNING_BUILD_MODE ?= release
CONFIGURED_BUILDS := $(shell PYTHONPATH="$(ROOT)optimizer" "$(PLANNING_PYTHON)" -c 'from planning.settings.configuration import load_config,runtime_path; c=load_config(); print(runtime_path(c,"build"),runtime_path(c,"native_bindings").parent)')
PLANNING_DUCKDB_BUILD ?= $(if $(filter debug,$(PLANNING_BUILD_MODE)),$(ROOT)build/debug,$(or $(word 1,$(CONFIGURED_BUILDS)),$(ROOT)build/release))
PYTHON_BUILD ?= $(if $(filter debug,$(PLANNING_BUILD_MODE)),$(ROOT)build/debug-bindings,$(or $(word 2,$(CONFIGURED_BUILDS)),$(ROOT)build/release-bindings))
BUILD_JOBS ?= $(or $(CMAKE_BUILD_PARALLEL_LEVEL),8)
EXTENSION_STATIC_BUILD ?= 1
EXT_FLAGS ?=
SANITIZER_FLAGS = $(if $(filter debug,$(PLANNING_BUILD_MODE)),-DENABLE_SANITIZER=OFF -DENABLE_UBSAN=OFF,)

BUILD_MODES := debug release
.PHONY: all $(BUILD_MODES) clean-build rebuild duckdb-native planning-native $(addprefix planning-native-,$(BUILD_MODES))
all: release
debug: PLANNING_BUILD_MODE = debug
release: PLANNING_BUILD_MODE = release
release: planning-native-release
debug: planning-native-debug
planning-native-debug planning-native-release: planning-native-%:
	$(MAKE) planning-native PLANNING_BUILD_MODE=$*

clean-build:
	rm -rf "$(ROOT)build/release" "$(ROOT)build/release-bindings" \
	       "$(ROOT)build/debug" "$(ROOT)build/debug-bindings"

rebuild: clean-build
	$(MAKE) release

duckdb-native:
	cmake -S third_party/duckdb -B "$(PLANNING_DUCKDB_BUILD)" -G Ninja \
	  -DCMAKE_BUILD_TYPE=$(if $(filter debug,$(PLANNING_BUILD_MODE)),Debug,Release) \
	  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	  -DDUCKDB_EXTENSION_CONFIGS="$(ROOT)cmake/extensions.cmake" \
	  -DEXTENSION_STATIC_BUILD=$(EXTENSION_STATIC_BUILD) \
	  -DENABLE_EXTENSION_AUTOLOADING=0 -DENABLE_EXTENSION_AUTOINSTALL=0 \
	  -DBUILD_UNITTESTS=OFF -DENABLE_UNITTEST_CPP_TESTS=OFF $(SANITIZER_FLAGS) $(EXT_FLAGS)
	cmake --build "$(PLANNING_DUCKDB_BUILD)" --target shell src/libduckdb.so semantic_loadable_extension beam_search_loadable_extension tpch_loadable_extension semcache_statistics_build --parallel $(BUILD_JOBS)

planning-native: duckdb-native
	cmake -S optimizer/bindings -B "$(PYTHON_BUILD)" -G Ninja -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE="$(PLANNING_PYTHON)" \
	  -DSEM_DUCKDB_BUILD="$(PLANNING_DUCKDB_BUILD)"
	cmake --build "$(PYTHON_BUILD)" --parallel $(BUILD_JOBS)

NATIVE_SOURCE_DIRS := beam-search-duckdb/src statistics optimizer/bindings
PYTHON_SOURCE_DIRS := train.py optimizer experiments ../experiments $(wildcard ../benchmarks/scripts tools)
CMAKE_SOURCE_FILES := $(shell rg --files --no-ignore cmake optimizer/bindings beam-search-duckdb semantic-duckdb statistics -g '*.cmake' -g CMakeLists.txt)
.PHONY: format-check format-fix native-format-check native-format-fix \
        optimizer-format-check optimizer-format-fix optimizer-typecheck \
        cmake-format-check cmake-format-fix
format-check: native-format-check optimizer-format-check cmake-format-check
	$(TOOLS_PYTHON) third_party/duckdb/scripts/format.py --all --check --directories semantic-duckdb/src
format-fix: native-format-fix optimizer-format-fix cmake-format-fix
	$(TOOLS_PYTHON) third_party/duckdb/scripts/format.py --all --fix --noconfirm --directories semantic-duckdb/src

native-format-check:
	rg --files $(NATIVE_SOURCE_DIRS) -g '*.cpp' -g '*.hpp' -g '*.h' | xargs -r "$(ROOT).venv-tools/bin/clang-format" --dry-run --Werror
native-format-fix:
	rg --files $(NATIVE_SOURCE_DIRS) -g '*.cpp' -g '*.hpp' -g '*.h' | while IFS= read -r source; do \
	  "$(ROOT).venv-tools/bin/clang-format" --dry-run --Werror "$$source" >/dev/null 2>&1 || "$(ROOT).venv-tools/bin/clang-format" -i "$$source" || exit; \
	done
optimizer-format-check:
	$(TOOLS_PYTHON) -m ruff check --config pyproject.toml $(PYTHON_SOURCE_DIRS)
	$(TOOLS_PYTHON) -m ruff format --config pyproject.toml --check $(PYTHON_SOURCE_DIRS)
optimizer-format-fix:
	$(TOOLS_PYTHON) -m ruff check --config pyproject.toml --fix $(PYTHON_SOURCE_DIRS)
	$(TOOLS_PYTHON) -m ruff format --config pyproject.toml $(PYTHON_SOURCE_DIRS)
optimizer-typecheck:
	$(TOOLS_PYTHON) -m basedpyright --project pyproject.toml
cmake-format-check:
	$(TOOLS_PYTHON) -m cmakelang.format --check $(CMAKE_SOURCE_FILES)
cmake-format-fix:
	$(TOOLS_PYTHON) -m cmakelang.format -i $(CMAKE_SOURCE_FILES)
