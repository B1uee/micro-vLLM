# Repository Guidelines

## Project Structure & Module Organization
`nanovllm/` is the core package. Keep runtime logic in the existing submodules:
- `nanovllm/engine/`: scheduling, sequence state, block management, and model execution.
- `nanovllm/layers/`: attention, norms, linear layers, embeddings, and sampler internals.
- `nanovllm/models/`: model-specific definitions (for example `qwen3.py`).
- `nanovllm/utils/`: shared helpers (loading, context utilities).

Top-level scripts are for usage and performance checks: `example.py` (quick start) and `bench.py` (throughput benchmark). Static assets live in `assets/`.

## Build, Test, and Development Commands
Use Python 3.10-3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python example.py
python bench.py
```

- `pip install -e .`: install package in editable mode for local development.
- `python example.py`: functional smoke run against a local model path.
- `python bench.py`: simple performance benchmark.

If you need a distributable package, run `python -m build` (after installing `build`).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear, short functions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and descriptive module names.
- Keep APIs consistent with vLLM-style usage where possible; avoid unnecessary interface drift.
- Prefer explicit data flow and minimal special-case branching in engine code.

## Testing Guidelines
There is currently no dedicated `tests/` directory. For now:
- Treat `example.py` as the minimum functional smoke test.
- Use `bench.py` to catch obvious performance regressions.
- Add targeted script-based checks when changing scheduling, caching, or sampling behavior.

When introducing non-trivial features, add reproducible test cases (new `tests/` package is encouraged) and document expected outputs.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages (for example: `support qwen2`, `simplify`, `fix(model_runner): correct position indexing`).

For pull requests:
- Explain what changed and why.
- Call out behavior changes and compatibility impact.
- Include exact run commands used for validation.
- Link related issues and attach benchmark deltas when performance-sensitive paths are touched.
