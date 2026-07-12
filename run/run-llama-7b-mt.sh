export CUDA_VISIBLE_DEVICES=0
export TREE_UPDATE_METHOD=argmax
export MAX_CHILD_NUM=4
export MAX_DEPTH=6


python main.py --temperature 0.5 --model-path /root/autodl-fs/Llama-2-7b-chat-hf --run-mode draft  --sample-number -1 --question-file data/mt-bench/mt-bench.jsonl | tee res/log-sample/DT-Llama-2-7b-chat-hf-mt--1.log
python main.py --temperature 0.5 --model-path /root/autodl-fs/Llama-2-7b-chat-hf --run-mode ar  --sample-number -1 --question-file data/mt-bench/mt-bench.jsonl | tee res/log-sample/AR-Llama-2-7b-chat-hf-mt--1.log