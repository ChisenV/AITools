# AGENTS.md

Guidance for AI coding agents working in this repository. Keep this file up to date when the project structure, tooling, or conventions change.

## Project overview

**AITools** (`aitools`) is a highly customizable Python library for AI model deployment / inference and dataset processing, with a focus on computer-vision workflows. It provides a config-driven, plugin-style component system so that datasets, preprocessors, models (ONNX / TensorRT backends), post-processors, evaluators, and savers can be assembled from YAML workflow files instead of hard-coded scripts.

- Package name: `aitools` (imported as `AITools`)
- Author: ChisenV — https://github.com/ChisenV/AITools
- Python: `>=3.8` (classifiers cover 3.8–3.11)
- License: see `LICENSE`
- Version: source of truth is the `VERSION_NUMBER` file (read by `setup.py`)

## Repository layout

```
AITools/            # Main package
  base/             # Abstract base classes / protocol definitions (dataset, model, process, vision, plugin)
  core/             # Framework core: Config, Builder, ComponentManager, logging
  comp/             # Concrete components
    backend/        # Inference backends (trt.py -> TensorRTModel, onnx.py)
    dataset.py, functions.py, parser.py, processor.py, saver.py, evaluator.py
  apps/             # Higher-level applications (apps/pipeline/vision.py)
  utils/            # Helpers: plotting, property, compatibility, image2video, stitch_images
crawler/            # Standalone paper/data crawling scripts (not part of the package)
docs/               # Design docs (written in Chinese)
example/            # Runnable usage examples (deploy, dataset processing, etc.)
tests/              # pytest tests (comp/, core/, web/, workflow/)
libs/               # Git submodules (CGraph, cocoapi)
setup.py, requirements.txt, VERSION_NUMBER
install.sh / install.bat, build_wheel.sh / build_wheel.bat
```

## Core architecture

The framework is built around three cooperating pieces in `AITools/core`:

- **`ComponentManager`** (`manager.py`): a registry. Components register themselves as a class or function, keyed by their class name (via `register_component`, usable as a decorator). Managers can auto-append to the global `COMPONENT_MANAGERS` list. Example: `BACKENDS = ComponentManager("backends")`.
- **`Config`** (`config.py`): loads YAML/JSON, supports multi-file inheritance via `_base_`, dynamic overrides via `opts`, and uses special keys `BASE_KEY='_base_'`, `INHERIT_KEY='_inherited_'`, `TYPE_KEY='_type_'`.
- **`Builder`** (`builder.py`): lazily and recursively instantiates components from a config dict. It looks up each `_type_` string in the registered managers, resolves nested/sub-component configs, and supports post-build hooks.

Workflows are expressed as YAML (see `tests/workflow/workflow_1.yml`) where each block has a `type:` and dependencies, e.g. `dataset`, `preprocessor`, `model`. Prefer extending this config-driven pattern over adding one-off scripts.

## Setup, build & test commands

Install for development (installs deps + editable package):

```bash
./install.sh            # Linux/macOS  (install.bat on Windows)
# equivalent to:
pip install -r requirements.txt
pip install -e .
```

Build a wheel:

```bash
./build_wheel.sh        # build_wheel.bat on Windows -> runs `python -m build --wheel`
```

Run tests (pytest is the test runner; `pytest` + `pytest-benchmark` are in requirements):

```bash
pytest                  # run the suite
pytest tests/core       # run a subset
pytest tests/core/test_config.py::<test_name>
```

Submodules (only needed if you touch `libs/`):

```bash
git submodule update --init --recursive
```

## Important environment caveats for agents

- **Heavy / hardware-specific dependencies.** `requirements.txt` pins `tensorrt`, `cuda-python`, `onnxruntime`, `onnx-graphsurgeon`, and `pypiwin32` (Windows-only). A generic Linux CI/agent box without an NVIDIA GPU + CUDA/TensorRT cannot import the TensorRT backend or run GPU tests. Don't assume a full install succeeds; verify before relying on it.
- **Optional imports are guarded.** Code degrades gracefully when optional deps are missing (e.g. `base/__init__.py` fakes `torch`; `comp/backend/__init__.py` wraps the ONNX import in `try/except ImportError`). Preserve this pattern — never make an optional dependency a hard import at module top level.
- **Tests contain hard-coded local paths.** Many tests/examples reference Windows dataset paths like `E:\python_ai_dataset\...`. These are author-local and will fail in this environment. Do not treat those failures as regressions you introduced, and avoid adding new machine-specific absolute paths.

## Coding conventions

- **Language:** English for code, identifiers, and docstrings. Design docs under `docs/` are in Chinese — that's fine, follow the existing language of a file when editing it.
- **Type hints:** use them for public function/method signatures (the codebase relies heavily on `typing`).
- **Docstrings:** Google-style with `Args:` / `Returns:` / `Raises:` sections, matching existing files (see `core/builder.py`, `core/manager.py`).
- **Naming:** component classes are `PascalCase` and are referenced by that exact class name in YAML `_type_` fields. Constants are `UPPER_SNAKE_CASE`. Modules expose their public API via `__all__` and `from .x import *` in `__init__.py`.
- **Registering new components:** create/obtain the appropriate `ComponentManager` and use `@manager.register_component` on the new class so it becomes buildable from config. Do not overwrite an existing registered name unless you intentionally pass `allow_overwrite=True`.
- **Comments:** only for non-obvious intent; do not narrate what the code plainly does.

## Do / Don't

- Do keep the config-driven, registry + builder architecture when adding features.
- Do update `VERSION_NUMBER` (not a hard-coded literal) when bumping the release version.
- Do respect `.gitignore` — never commit datasets, models (`*.pt`, `*.pth`, `model*/`), build artifacts (`build/`, `dist/`, `*.egg-info`), or logs.
- Don't add hard top-level imports of optional/GPU-only dependencies.
- Don't commit personal absolute filesystem paths in code or tests.

## Git & PR workflow

- Follow the existing commit message style seen in history: a bracketed tag prefix, e.g. `[update] <short description>` / `[fix] ...` / `[add] ...`.
- Make one commit per logical change; do not force-push or amend unless explicitly asked.
- Stay on your working branch and open a PR against the default base branch.
