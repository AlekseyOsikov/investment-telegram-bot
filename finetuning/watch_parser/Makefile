.PHONY: data eval-set check v2-check v3-check holdout-check ollama-eval all clean

data:
	python3 scripts/build_dataset.py

eval-set:
	python3 scripts/make_eval.py

check:
	python3 scripts/check_dataset.py
	python3 scripts/check_bot_compat.py

v2-check:
	python3 scripts/check_dataset.py \
		--train data/v2/train.jsonl \
		--valid data/v2/valid.jsonl \
		--eval data/eval.jsonl \
		--report data/v2/check_report.json
	python3 scripts/check_bot_compat.py \
		--train data/v2/train.jsonl \
		--valid data/v2/valid.jsonl \
		--eval data/eval.jsonl

v3-check:
	python3 scripts/check_dataset.py \
		--train data/v3/train.jsonl \
		--valid data/v3/valid.jsonl \
		--eval data/eval.jsonl \
		--report data/v3/check_report.json
	python3 scripts/check_bot_compat.py \
		--train data/v3/train.jsonl \
		--valid data/v3/valid.jsonl \
		--eval data/eval.jsonl

holdout-check:
	python3 scripts/check_dataset.py \
		--train data/v3/train.jsonl \
		--valid data/v3/valid.jsonl \
		--eval data/holdout.jsonl \
		--report data/holdout_check.json

ollama-eval:
	python3 scripts/ollama_eval.py --eval data/eval.jsonl \
		--model qwen25-watch-v3 --out results/v3/ollama.json

all: data eval-set check

clean:
	rm -rf data/v1
