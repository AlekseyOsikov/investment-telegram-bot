# Задачи проекта «Дистилляция».
# Обучение — в Colab (training/train_lora.py), здесь — данные, разметка учителем и eval.

BASE := qwen2.5:1.5b-instruct-fp16
TEACHER := qwen2.5:7b
DISTILL := qwen25-distill-v1
GOLDEN := qwen25-golden-v1

PY := python3

.PHONY: corpus label filter eval eval-all compare zip clean

corpus: ## неразмеченный корпус
	$(PY) data/make_inputs.py

label: ## учитель размечает корпус
	$(PY) scripts/teacher_label.py --model $(TEACHER)

filter: ## проверка ответов учителя и сборка train/valid
	$(PY) scripts/filter_labels.py

eval: ## замер базы и учителя на тесте
	$(PY) scripts/eval_three.py --variants base,teacher

eval-all: ## замер всех вариантов (дистиллированная — после Colab)
	$(PY) scripts/eval_three.py --variants base,teacher,distilled,golden

compare: ## сводная таблица и анализ «зон замены»
	$(PY) scripts/compare.py

zip: ## архив для Colab
	cd ../.. && rm -f finetuning/distillation.zip
	cd ../.. && zip -qr finetuning/distillation.zip finetuning/distillation \
		-x '*/__pycache__/*' '*/.ruff_cache/*' '*/results/*' '*/teacher_raw.jsonl'

clean:
	rm -rf data/teacher_raw.jsonl data/train_distill.jsonl data/valid_distill.jsonl \
		data/train_golden.jsonl data/valid_golden.jsonl data/filter_report.json \
		data/corpus.jsonl data/corpus_stats.json scripts/__pycache__ data/__pycache__
