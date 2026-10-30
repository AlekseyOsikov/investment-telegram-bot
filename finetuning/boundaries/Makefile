# Задачи проекта «Границы файнтюнинга».
# Обучение фактов — в Colab (training/colab_boundaries.ipynb), здесь — данные и eval.

BASE := qwen2.5:1.5b-instruct-fp16
TUNED_FACTS := qwen25-facts-v1
ROLEPLAY := qwen25-roleplay
WATCH := qwen25-watch-v3

PY := python3

.PHONY: data facts-none facts-prompt facts-tuned ood-base ood-tuned style-none style-prompt style-tuned compare console zip clean

data: ## собрать train/valid/eval фактов + наборы вне домена и стиля
	$(PY) data/build_facts.py
	$(PY) data/out_of_domain.py
	$(PY) data/style_eval.py

facts-none: ## факты: база без фактов
	$(PY) scripts/eval_facts.py --arm base_none --model $(BASE) --out results/facts/base_none.json

facts-prompt: ## факты: база + факт в промпте (baseline RAG)
	$(PY) scripts/eval_facts.py --arm base_facts --model $(BASE) --out results/facts/base_facts.json

facts-tuned: ## факты: дообученная facts-модель
	$(PY) scripts/eval_facts.py --arm tuned --model $(TUNED_FACTS) --out results/facts/tuned.json

ood-base: ## задачи вне домена: база
	$(PY) scripts/eval_out_of_domain.py --model $(BASE) --out results/ood/base.json

ood-tuned: ## задачи вне домена: дообученная facts-модель
	$(PY) scripts/eval_out_of_domain.py --model $(TUNED_FACTS) --out results/ood/tuned.json

style-none: ## стиль: база без роли
	$(PY) scripts/eval_style.py --arm base_none --model $(BASE) --out results/style/base_none.json

style-prompt: ## стиль: база + ролевой промпт
	$(PY) scripts/eval_style.py --arm base_prompt --model $(BASE) --out results/style/base_prompt.json

style-tuned: ## стиль: дообученная roleplay-модель
	$(PY) scripts/eval_style.py --arm tuned --model $(ROLEPLAY) --out results/style/tuned.json

compare: ## сводная таблица трёх типов задач
	$(PY) scripts/compare_three.py \
		--facts results/facts/base_none.json results/facts/base_facts.json results/facts/tuned.json \
		--style results/style/base_none.json results/style/base_prompt.json results/style/tuned.json

console: ## консольное сравнение трёх типов (ответы моделей)
	$(PY) scripts/compare_console.py

zip: ## архив для Colab: содержит пути finetuning/boundaries/...
	cd ../.. && rm -f finetuning/boundaries.zip
	cd ../.. && zip -qr finetuning/boundaries.zip finetuning/boundaries \
		-x '*/__pycache__/*' '*/.ruff_cache/*'

clean:
	rm -rf results/ data/train.jsonl data/valid.jsonl data/eval_facts.json data/out_of_domain.json data/style_eval.json data/__pycache__ scripts/__pycache__
