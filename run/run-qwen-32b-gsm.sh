export CUDA_VISIBLE_DEVICES=0,1
export TREE_UPDATE_METHOD=argmax
export MAX_CHILD_NUM=4
export MAX_DEPTH=6


python main.py --temperature 0.5 --model-path /root/autodl-fs/Qwen2.5-32B-Instruct --run-mode draft  --sample-number 100 --question-file data/gsm/test.jsonl \
--use-pp 1 --num-gpus-per-model 2 --num-gpus-total 2 | tee res/log-sample/DT-qwen-32b-gpus-2-chat-hf-gsm-100.log
python main.py --temperature 0.5 --model-path /root/autodl-fs/Qwen2.5-32B-Instruct --run-mode ar  --sample-number 100 --question-file data/gsm/test.jsonl \
--use-pp 1 --num-gpus-per-model 2 --num-gpus-total 2 | tee res/log-sample/AR-qwen-32b-gpus-2-chat-hf-gsm-100.log