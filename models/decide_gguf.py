# Adapted from Mapika/decider-2b-GGUF's decide_gguf.py and decider.prompt
# (Apache-2.0; see LICENSE-APACHE-2.0.txt and models/README.md).
"""Mapika Decider GGUF on llama.cpp, adapted to use the bundled tokenizer directly.

The prompt layout, option labels, and temperature handling follow Mapika's Apache-2.0
GGUF helper. Using the tokenizer.json through the `tokenizers` package keeps this
runtime compatible with EchoMind's pinned ChromaDB dependency without importing
Transformers or the PyTorch inference stack.

    from models.decide_gguf import GGUFDecider
    d = GGUFDecider("decider-2b-v11-Q4_K_M.gguf")          # tokenizer and decider_config.json from the same folder
    d.decide("My card was charged twice for the same purchase.",
             [{"question": "Which department should handle this?", "options": ["billing", "technical", "sales"]}])
"""
import ctypes
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import llama_cpp as L
from tokenizers import Tokenizer

MAX_OPTIONS = 255
NARROW_OPTIONS = 10
LETTERS = "ABCDEFGHIJ"


@dataclass
class _Question:
    text: str
    options: list
    gold: int = 0


@dataclass
class _Example:
    context: str
    qs: list


class _NoShuffle:
    def shuffle(self, values):
        return None

    def sample(self, values, count):
        return values[:count]


class _TokenizerAdapter:
    """Expose the small tokenizer API used by Mapika's prompt builder."""

    def __init__(self, path: Path):
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def _label_ids(tokenizer):
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    labels.extend(a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    selected = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1:
            selected.append((label, ids[0]))
        if len(selected) == MAX_OPTIONS:
            break
    if len(selected) != MAX_OPTIONS or len({token_id for _, token_id in selected}) != MAX_OPTIONS:
        raise ValueError("tokenizer_does_not_support_decider_option_labels")
    for index, label in enumerate(LETTERS):
        if tokenizer.encode(label, add_special_tokens=False) != [selected[index][1]]:
            raise ValueError("tokenizer_decider_letter_ids_mismatch")
    return [token_id for _, token_id in selected]


def _build_prompt(example, tokenizer, max_ctx_tokens):
    """Build the upstream state-first prompt and one answer slot per question."""
    context_ids = tokenizer.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    ids = list(context_ids)
    slots = []
    option_counts = []
    rng = _NoShuffle()
    multiple = len(example.qs) > 1
    label_ids = None
    open_option_ids = None

    for question_index, question in enumerate(example.qs):
        options = list(range(len(question.options)))
        if len(options) > MAX_OPTIONS:
            forced = {question.gold} | {
                index for index, option in enumerate(question.options)
                if option.strip().lower().startswith((
                    "none of the above", "none of these", "not listed", "no suitable",
                    "does not apply", "cannot tell",
                )) or option.strip().lower() in {
                    "other", "unsure", "something else", "neither of these", "other / not covered",
                }
            }
            others = [index for index in options if index not in forced]
            options = rng.sample(others, MAX_OPTIONS - len(forced)) + list(forced)
        rng.shuffle(options)
        option_counts.append(len(options))
        head = f"\n\nQuestion{' ' + str(question_index + 1) if multiple else ''}: {question.text}\nOptions:"
        tail = f"\nAnswer{' ' + str(question_index + 1) if multiple else ''}: ("

        if len(options) <= NARROW_OPTIONS:
            prompt_piece = head + "".join(
                f"\n({LETTERS[option_index]}) {question.options[option_index_in_question]}"
                for option_index, option_index_in_question in enumerate(options)
            ) + tail
            ids.extend(tokenizer.encode(prompt_piece, add_special_tokens=False))
        else:
            if label_ids is None:
                label_ids = _label_ids(tokenizer)
                open_option_ids = tokenizer.encode("\n(", add_special_tokens=False)
            ids.extend(tokenizer.encode(head, add_special_tokens=False))
            for option_index, option_index_in_question in enumerate(options):
                ids.extend(open_option_ids)
                ids.append(label_ids[option_index])
                ids.extend(tokenizer.encode(
                    f") {question.options[option_index_in_question]}", add_special_tokens=False
                ))
            ids.extend(tokenizer.encode(tail, add_special_tokens=False))

        slots.append(len(ids) - 1)

    return {"ids": ids, "slots": slots, "nopts": option_counts}


def _choice_temperature(config):
    by_type = config.get("temperature_by_type", {})
    if by_type is not None and not isinstance(by_type, dict):
        raise ValueError("decider_config_temperature_by_type_must_be_a_map")
    value = (by_type or {}).get("choice", config.get("temperature", 1.0))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("decider_config_choice_temperature_must_be_positive")
    return float(value)

_QUIET = L.llama_log_callback(lambda level, text, data: None)


class GGUFDecider:
    def __init__(self, gguf_path, folder=None, n_ctx=32768, n_gpu_layers=-1, n_threads=None, verbose=False):
        folder = folder or os.path.dirname(os.path.abspath(gguf_path))
        cfg = json.load(open(os.path.join(folder, "decider_config.json")))
        self.temperature = _choice_temperature(cfg)
        self.name = "decider-" + str(cfg.get("version", "dev"))
        self.tok = _TokenizerAdapter(Path(folder) / "tokenizer.json")
        self.letters = np.asarray(_label_ids(self.tok))
        if not verbose:
            L.llama_log_set(_QUIET, ctypes.c_void_p(0))
        L.llama_backend_init()
        mp = L.llama_model_default_params(); mp.n_gpu_layers = n_gpu_layers
        self.model = L.llama_model_load_from_file(os.fsencode(gguf_path), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf_path}")
        cp = L.llama_context_default_params()
        cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = min(2048, n_ctx); cp.n_seq_max = 1
        if n_threads:
            cp.n_threads = cp.n_threads_batch = n_threads
        self.ctx = L.llama_init_from_model(self.model, cp)
        self.n_ctx = n_ctx
        self.n_vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(self.model))
        self.batch = L.llama_batch_init(n_ctx, 0, 1)

    def _slot_logits(self, ids, slots):
        if len(ids) > self.n_ctx:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds n_ctx {self.n_ctx}")
        L.llama_memory_clear(L.llama_get_memory(self.ctx), True)
        b, want = self.batch, set(slots)
        for i, t in enumerate(ids):
            b.token[i] = t; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i in want
        b.n_tokens = len(ids)
        if L.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        rows = []
        for s in slots:
            p = ctypes.cast(L.llama_get_logits_ith(self.ctx, s), ctypes.POINTER(ctypes.c_float))
            rows.append(np.ctypeslib.as_array(p, shape=(self.n_vocab,))[self.letters].astype(np.float64))
        return rows

    def decide(self, context, questions, max_ctx_tokens=1536):
        """questions: [{"question": str, "options": [str, ...]}, ...] (2..255 options).
        -> [{"choice", "confidence", "probs"}, ...], one per question, as decider.infer.Decider.decide returns them."""
        for q in questions:
            assert 2 <= len(q["options"]) <= MAX_OPTIONS, f"2..{MAX_OPTIONS} options required"
        item = _build_prompt(
            _Example(context, [_Question(q["question"], list(q["options"]), 0) for q in questions]),
            self.tok,
            max_ctx_tokens,
        )
        out = []
        for q, lg, n in zip(questions, self._slot_logits(item["ids"], item["slots"]), item["nopts"]):
            z = lg[:n] / self.temperature; p = np.exp(z - z.max()); p /= p.sum()
            j = int(p.argmax())
            out.append(dict(choice=q["options"][j], confidence=float(p[j]), probs=dict(zip(q["options"], p.tolist()))))
        return out
