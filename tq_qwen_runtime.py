from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from threading import Event
from typing import Generator, Iterable

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import hf_repo_to_path
from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention

from tq_qwen_common import clean_response

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. "
    "Provide only the final answer to the user. "
    "Do not reveal chain-of-thought, internal reasoning, scratchpad notes, "
    "or a thinking process. "
    "If the user asks for reasoning, provide a short direct explanation instead."
)


@dataclass(slots=True)
class RuntimeConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = 200
    temp: float = 0.7
    seed: int = 42
    bits: int = 4
    offline: bool = False
    verbose: bool = False


def iter_self_attention_layers(model: object) -> Iterable[tuple[int, object]]:
    for index, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            yield index, layer


def resolve_model_source(model_ref: str, offline: bool) -> str:
    if os.path.exists(model_ref):
        return model_ref

    try:
        return str(hf_repo_to_path(model_ref))
    except Exception:
        if offline:
            raise RuntimeError(
                f"Model '{model_ref}' is not present in the local Hugging Face cache."
            )
        return model_ref


class TurboQuantRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        patch_attention()
        model_source = resolve_model_source(config.model, config.offline)
        self.model, self.tokenizer = load(model_source)
        self.model_source = model_source

    def make_cache(self):
        cache = self.model.make_cache()
        patched_layers = 0
        for index, layer in iter_self_attention_layers(self.model):
            cache[index] = TurboQuantKVCache(
                head_dim=layer.self_attn.head_dim,
                bits=self.config.bits,
                seed=self.config.seed + index,
            )
            patched_layers += 1
            if self.config.verbose:
                print(
                    f"patched layer {index} with TurboQuantKVCache(bits={self.config.bits})",
                    file=sys.stderr,
                )

        if patched_layers == 0:
            raise RuntimeError("No self-attention layers found to patch.")

        if self.config.verbose:
            print(
                f"patched {patched_layers} self-attention cache slots",
                file=sys.stderr,
            )

        return cache

    def build_single_prompt(self, prompt: str, raw_prompt: bool = False) -> str:
        if raw_prompt or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("This tokenizer does not support chat templates.")

        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def stream_prompt_events(
        self,
        prompt: str,
        stop_event: Event | None = None,
    ) -> Generator[dict[str, int | float | str], None, None]:
        cache = self.make_cache()
        sampler = make_sampler(self.config.temp) if self.config.temp > 0 else None

        full_text = ""
        printed_text = ""
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            sampler=sampler,
            prompt_cache=cache,
        ):
            if stop_event is not None and stop_event.is_set():
                break

            full_text += response.text
            cleaned = clean_response(full_text)
            if len(cleaned) > len(printed_text):
                delta = cleaned[len(printed_text) :]
                printed_text = cleaned
                yield {
                    "delta": delta,
                    "token_count": response.generation_tokens,
                    "tokens_per_second": round(response.generation_tps, 2),
                    "text": cleaned,
                }

    def stream_prompt(
        self,
        prompt: str,
        stop_event: Event | None = None,
    ) -> Generator[str, None, None]:
        for event in self.stream_prompt_events(prompt, stop_event=stop_event):
            yield str(event["delta"])

    def stream_single(
        self,
        prompt: str,
        raw_prompt: bool = False,
        stop_event: Event | None = None,
    ) -> Generator[str, None, None]:
        built_prompt = self.build_single_prompt(prompt, raw_prompt=raw_prompt)
        yield from self.stream_prompt(built_prompt, stop_event=stop_event)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        stop_event: Event | None = None,
    ) -> Generator[str, None, None]:
        built_prompt = self.build_chat_prompt(messages)
        yield from self.stream_prompt(built_prompt, stop_event=stop_event)

    def stream_chat_events(
        self,
        messages: list[dict[str, str]],
        stop_event: Event | None = None,
    ) -> Generator[dict[str, int | float | str], None, None]:
        built_prompt = self.build_chat_prompt(messages)
        yield from self.stream_prompt_events(built_prompt, stop_event=stop_event)
