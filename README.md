# Qwen3.5 9B TurboQuant Runner

Minimal local runner for `mlx-community/Qwen3.5-9B-OptiQ-4bit` with TurboQuant KV cache on Apple Silicon.

## Setup

```bash
cd ~/tq-qwen
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd ~/tq-qwen
source .venv/bin/activate
python run_qwen_turboquant.py "Explain how recurrent and self-attention layers differ."
```

Interactive chat:

```bash
cd ~/tq-qwen
source .venv/bin/activate
python run_qwen_turboquant.py --offline --chat
```

Streaming is enabled by default. If you want full-response mode instead:

```bash
python run_qwen_turboquant.py --offline --chat --no-stream
```

Chat commands:

- `/clear` resets the conversation history.
- `/exit` or `/quit` leaves chat mode.

After the first download, force local-only inference:

```bash
cd ~/tq-qwen
source .venv/bin/activate
python run_qwen_turboquant.py --offline "Explain how recurrent and self-attention layers differ."
```

## Browser UI

Start the local server:

```bash
cd ~/tq-qwen
source .venv/bin/activate
python serve_qwen_ui.py --offline
```

Then open `http://127.0.0.1:8015`.

UI behavior:

- The UI shows a live `Thinking...` state with a token count while the model generates.
- The assistant bubble renders the final answer in a formatted way after generation completes.
- Only one response can generate at a time.
- `Stop` interrupts the current response between token steps.
- `Reset` clears the local conversation history on the server.

Exact-answer smoke test:

```bash
cd ~/tq-qwen
source .venv/bin/activate
python run_qwen_turboquant.py --max-tokens 16 --temp 0.0 --raw-prompt \
  "Answer with exactly: TurboQuant KV cache is active."
```

## Notes

- The model weights come from Hugging Face the first time you run the script.
- After the first download, `--offline` loads from the local Hugging Face cache only.
- TurboQuant is applied only to layers exposing `self_attn`; the recurrent path remains untouched.
- Use `--bits 3` or `--bits 4` to change KV cache compression.
- Use `--verbose` to print which cache slots were replaced.
- By default the script uses the tokenizer chat template when available. Use `--raw-prompt` to bypass it.
- Use `--chat` for a terminal REPL instead of a single prompt/response.
- The browser UI is served by `serve_qwen_ui.py` and uses the same runtime module as the CLI.
