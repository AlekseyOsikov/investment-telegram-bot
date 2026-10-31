"""LoRA-обучение Qwen2.5-1.5B-Instruct на метках учителя (дистилляция).

Логика повторяет ``watch_parser/training/train_lora.py``: лосс только по ответу
ассистента (системный промпт и вопрос замаскированы -100), те же гиперпараметры
(r=16/α=32, 2 эпохи, T4). Отличие — данные: ``data/train_distill.jsonl`` (метки
учителя, отфильтрованные по форме) или ``data/train_golden.jsonl`` (строгая
фильтрация по смыслу). Скрипт самодостаточен, чтобы Colab нужна была только
папка ``distillation/``.

Запуск в Colab (T4):
    !pip -q install -U "transformers>=4.56" "peft>=0.13" accelerate
    !pip -q uninstall -y torchao
    # Runtime -> Restart session
    !python training/train_lora.py --variant distill --out /content/adapter_distill
    !python training/train_lora.py --variant golden  --out /content/adapter_golden
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["distill", "golden"], default="distill")
    parser.add_argument("--train", default=None, help="по умолчанию data/train_<variant>.jsonl")
    parser.add_argument("--valid", default=None, help="по умолчанию data/valid_<variant>.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="/content/adapter_distill")
    parser.add_argument("--run-dir", default="/content/run_distill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode(messages: list[dict], tok, max_len: int) -> dict:
    """Токенизирует диалог и оставляет labels только на ответе ассистента."""
    full = tok.apply_chat_template(messages, tokenize=False)
    assistant = messages[-1]["content"]
    start_char = full.rfind(assistant)
    if start_char < 0:
        raise ValueError("не нашли ответ ассистента в chat template")

    enc = tok(
        full,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    labels = [
        token_id if offset_start >= start_char else -100
        for token_id, (offset_start, _offset_end) in zip(enc["input_ids"], enc["offset_mapping"], strict=True)
    ]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


class EncodedDataset(Dataset):
    """Список уже токенизированных примеров; без pyarrow и datasets."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def build_encoded(rows: list[dict], tok, max_len: int) -> list[dict]:
    items = []
    for row in rows:
        item = encode(row["messages"], tok, max_len)
        if any(label != -100 for label in item["labels"]):
            items.append(item)
    return items


def main() -> None:
    args = parse_args()
    train_path = args.train or str(ROOT / "data" / f"train_{args.variant}.jsonl")
    valid_path = args.valid or str(ROOT / "data" / f"valid_{args.variant}.jsonl")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(), "Нет GPU: включите T4 в настройках среды выполнения"

    gpu = torch.cuda.get_device_properties(0)
    env = {
        "gpu": gpu.name,
        "vram_gb": round(gpu.total_memory / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    import peft
    import transformers

    env.update(transformers=transformers.__version__, peft=peft.__version__)

    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    print(f"variant={args.variant} train={len(train_rows)} valid={len(valid_rows)}")
    print("пример:", train_rows[0]["messages"][1]["content"], "→", train_rows[0]["messages"][2]["content"])

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "right"

    train_ds = EncodedDataset(build_encoded(train_rows, tok, args.max_len))
    valid_ds = EncodedDataset(build_encoded(valid_rows, tok, args.max_len))
    print(f"после токенизации: train={len(train_ds)} valid={len(valid_ds)}")

    lengths = [len(item["input_ids"]) for item in train_ds.items]
    if lengths:
        print("токенов: среднее", sum(lengths) // len(lengths), "макс", max(lengths))

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()

    steps_per_epoch = -(-len(train_ds) // (args.batch * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    print(f"шагов оптимизатора: {total_steps} | разогрев: {warmup_steps}")

    training_kwargs = dict(
        output_dir=args.run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        optim="adamw_torch",
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    started = time.time()
    trainer.train()
    train_seconds = round(time.time() - started)
    print(f"Время обучения: {train_seconds // 60} мин {train_seconds % 60} с")
    print("Пик памяти GPU, ГБ:", round(torch.cuda.max_memory_allocated() / 2**30, 1))

    history = trainer.state.log_history
    train_losses = [(entry["step"], entry["loss"]) for entry in history if "loss" in entry]
    eval_losses = [(entry["step"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    run = {
        "variant": args.variant,
        "cfg": vars(args),
        "env": env,
        "train_examples": len(train_ds),
        "valid_examples": len(valid_ds),
        "train_seconds": train_seconds,
        "loss_first": train_losses[0][1] if train_losses else None,
        "loss_last": train_losses[-1][1] if train_losses else None,
        "loss_history": train_losses,
        "eval_loss_history": eval_losses,
    }
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "run_params.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3.5))
        plt.plot([x for x, _ in train_losses], [y for _, y in train_losses], label="train")
        if eval_losses:
            plt.plot([x for x, _ in eval_losses], [y for _, y in eval_losses], marker="o", label="valid")
        plt.xlabel("шаг")
        plt.ylabel("loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(f"Лосс по ходу обучения ({args.variant})")
        plt.savefig(run_path / "loss.png", dpi=120, bbox_inches="tight")
    except Exception as exc:  # matplotlib может отсутствовать
        print("график не построен:", exc)

    print("адаптер сохранён в", out)
    print("параметры запуска:", run_path / "run_params.json")


if __name__ == "__main__":
    main()
