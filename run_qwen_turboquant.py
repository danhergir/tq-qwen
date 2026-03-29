#!/usr/bin/env python3

from __future__ import annotations

import argparse
from threading import Event

from tq_qwen_runtime import DEFAULT_MODEL, RuntimeConfig, TurboQuantRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 9B OptiQ with TurboQuant KV cache on MLX."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain why KV cache compression matters for long context.",
        help="Prompt to send to the model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, choices=(3, 4), default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


def emit_stream(chunks, stream: bool) -> str:
    full_text = ""
    for chunk in chunks:
        full_text += chunk
        if stream:
            print(chunk, end="", flush=True)
    if stream:
        print()
    return full_text


def run_chat(args: argparse.Namespace, runtime: TurboQuantRuntime) -> int:
    if args.raw_prompt:
        raise RuntimeError("--raw-prompt is not supported together with --chat.")

    messages: list[dict[str, str]] = []
    print("Chat mode. Commands: /exit, /quit, /clear")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = emit_stream(
            runtime.stream_chat(messages, stop_event=Event()),
            stream=not args.no_stream,
        )
        messages.append({"role": "assistant", "content": reply})
        if args.no_stream:
            print(f"Assistant> {reply}")

    return 0


def main() -> int:
    args = parse_args()
    runtime = TurboQuantRuntime(
        RuntimeConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            temp=args.temp,
            seed=args.seed,
            bits=args.bits,
            offline=args.offline,
            verbose=args.verbose,
        )
    )

    if args.chat:
        return run_chat(args, runtime)

    response = emit_stream(
        runtime.stream_single(args.prompt, raw_prompt=args.raw_prompt, stop_event=Event()),
        stream=not args.no_stream,
    )
    if args.no_stream:
        print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
