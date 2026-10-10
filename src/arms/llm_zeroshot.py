"""Zero-shot small-LLM anomaly scorer. Sample-based (seed 0 only); never in the main ranking table."""
import re

import numpy as np
import pandas as pd
import torch

from dataset_paths import processed_path

MODEL_NAME_PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LINES = 40

PROMPT_TEMPLATE = (
    "You are a log anomaly detector. Given this sequence of log templates from a system,"
    " estimate the probability (0-100) that it represents an anomaly.\n"
    "Log templates:\n{text}\n\n"
    "Respond with only a single integer from 0 to 100."
)


class LLMZeroShotArm:
    name = "llm_zeroshot"

    def fit(self, ctx, train_sessions):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.templates_ = pd.read_parquet(
            processed_path(ctx.dataset, "templates.parquet", ctx.selftest)
        ).set_index("template_id")["template"]

        # This machine's GPU is shared with the OS display / other apps, so VRAM headroom
        # varies run to run, and a CPU fp32 load of the 1.5B model can exceed the pagefile.
        # Try progressively smaller/cheaper options; see DECISIONS.md, Step 04.
        candidates = [
            (MODEL_NAME_PRIMARY, "cuda", torch.float16),
            (MODEL_NAME_FALLBACK, "cuda", torch.float16),
            (MODEL_NAME_FALLBACK, "cpu", torch.float32),
        ]
        candidates = [c for c in candidates if c[1] != "cuda" or torch.cuda.is_available()]

        last_err = None
        for model_name, device, dtype in candidates:
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                self.tokenizer_ = AutoTokenizer.from_pretrained(model_name)
                self.model_ = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
                self.device_ = device
                self.model_name_used_ = model_name
                break
            except (torch.OutOfMemoryError, OSError) as e:
                last_err = e
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
        else:
            raise RuntimeError(f"llm_zeroshot: all load candidates failed, last error: {last_err}")

        self.model_.eval()
        return self

    def _render(self, seq):
        lines = [self.templates_.get(t, "<unknown>") for t in list(seq)[:MAX_LINES]]
        return "\n".join(lines)

    def score(self, ctx, sessions):
        scores = np.zeros(len(sessions), dtype="float32")
        for i, seq in enumerate(sessions["template_seq"]):
            prompt = PROMPT_TEMPLATE.format(text=self._render(seq))
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer_.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer_(prompt_text, return_tensors="pt").to(self.device_)
            with torch.no_grad():
                out = self.model_.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=self.tokenizer_.eos_token_id,
                )
            gen = self.tokenizer_.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+", gen)
            scores[i] = min(100.0, float(m.group())) if m else 0.0
        return scores
