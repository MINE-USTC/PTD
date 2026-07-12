export CUDA_VISIBLE_DEVICES=0
export TREE_UPDATE_METHOD=argmax
export MAX_CHILD_NUM=4
export MAX_DEPTH=6


python main.py --temperature 0.5 --model-path /root/autodl-fs/Qwen2___5-7B-Instruct --run-mode draft  --sample-number 100 --question-file data/gsm/test.jsonl | tee res/log-sample/DT-qwen-2-7b-chat-hf-gsm-100.log
python main.py --temperature 0.5 --model-path /root/autodl-fs/Qwen2___5-7B-Instruct --run-mode ar  --sample-number 100 --question-file data/gsm/test.jsonl | tee res/log-sample/AR-qwen-2-7b-chat-hf-gsm-100.log