export CUDA_VISIBLE_DEVICES=0
export TREE_UPDATE_METHOD=argmax
export MAX_CHILD_NUM=4
export MAX_DEPTH=6


python main.py --temperature 0.5 --model-path /root/autodl-fs/Llama-2-7b-chat-hf --run-mode draft  --sample-number 100 --question-file data/gsm/test.jsonl | tee res/log-sample/DT-Llama-2-7b-chat-hf-gsm-100.log
python main.py --temperature 0.5 --model-path /root/autodl-fs/Llama-2-7b-chat-hf --run-mode ar  --sample-number 100 --question-file data/gsm/test.jsonl | tee res/log-sample/AR-Llama-2-7b-chat-hf-gsm-100.log