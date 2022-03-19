python run_mlm.py \
--model_type bert \
--tokenizer_name **path_to_tokenizer**\
--config_name **path_to_config** \
--train_file **path_to_train_file* *\
--do_train \
--max_seq_length 128\
--line_by_line True\
--load_best_model_at_end False\
--per_device_train_batch_size 32\
--gradient_accumulation_steps 1 \
--num_train_epochs 1\
--lr_scheduler_type linear\
--warmup_ratio 0.1\
--learning_rate 5e-05

