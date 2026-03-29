- [x] Choose isolated project layout under `~/tq-qwen`
- [x] Add a standalone TurboQuant runner for `mlx-community/Qwen3.5-9B-OptiQ-4bit`
- [x] Create local virtualenv and install runtime dependencies
- [x] Run a smoke test that loads the model and generates a short response
- [x] Extract shared runtime logic so the CLI and web app use the same generation path
- [x] Add a local browser chat UI with stop/reset/state endpoints
- [x] Add a practical streaming frontend for the local chat flow

## Review

Verified with `mlx-lm==0.31.1`, `mlx-optiq==0.0.2`, and a successful cached 9B run returning:
`TurboQuant KV cache is active.`
