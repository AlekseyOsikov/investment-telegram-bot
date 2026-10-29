"""Слияние LoRA-адаптера roleplay_lora с базой Qwen2.5-1.5B-Instruct.

Адаптер и merged-модель — тяжёлые артефакты, они лежат в
`finetuning/artifacts/roleplay_lora/` и не коммитятся.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "artifacts" / "roleplay_lora"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--adapter", default=str(ARTIFACTS / "adapter_roleplay_ru"))
    parser.add_argument("--out", default=str(ARTIFACTS / "merged"))
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16" if torch.cuda.is_available() else "float32",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print("база:", args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, low_cpu_mem_usage=True
    )
    print("адаптер:", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("сливаю...")
    model = model.merge_and_unload()
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print("готово:", out)


if __name__ == "__main__":
    main()
