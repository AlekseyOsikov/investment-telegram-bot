"""Слияние LoRA-адаптера фактов с базой Qwen2.5-1.5B-Instruct в обычный HF-каталог.

Запуск (после обучения в Colab):
    python scripts/merge_adapter.py \
      --adapter ../artifacts/boundaries/adapter_facts \
      --out ../artifacts/boundaries/merged_facts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print("загружаю базу:", args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, low_cpu_mem_usage=True)
    print("загружаю адаптер:", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("сливаю адаптер...")
    model = model.merge_and_unload()
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print("готово:", out)


if __name__ == "__main__":
    main()
