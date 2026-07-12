export CUDA_VISIBLE_DEVICES=0
export TREE_UPDATE_METHOD=argmax
export MAX_CHILD_NUM=4
export MAX_DEPTH=6


python main.py --temperature 0.5 --model-path /root/autodl-fs/CodeLlama-7b-Instruct-hf --run-mode draft  --sample-number 100 --question-file data/mbpp/ | tee res/log-sample/DT-CodeLlama-7b-Instruct-hf-mbpp-100.log
python main.py --temperature 0.5 --model-path /root/autodl-fs/CodeLlama-7b-Instruct-hf --run-mode ar  --sample-number 100 --question-file data/mbpp/ | tee res/log-sample/AR-CodeLlama-7b-Instruct-hf-mbpp-100.log