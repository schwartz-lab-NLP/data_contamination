tokenizer=**path_to_tokenizer**
config=**path_to_config**
corpus=**path_to_corpus** 

# pretrain and get mem metric
python pretrain/run_mlm.py \
--model_type bert \
--tokenizer_name ${tokenizer}\
--config_name  ${config}\
--train_file ${corpus} \
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

bert_model=**path_to_prertained_model**
results_path=**path_to_results_folder**
id=**model_identifier**

python pretrain/test_mem.py \
--bert_model ${bert_model}\
--tokenizer ${tokenizer}\
--results_path ${results_path}\
--model_id ${id}

# finetune and get expl metric
for seed in 0 1 42 483 494 210 799 852 464 398
do
python fintune/SST5_finetune.py\
--bert_model ${bert_model}\
--epoch 3\
--seed ${seed}\
--tokenizer ${tokenizer}\
--batch 8\
--results_path ${results_path}\
--model_id ${id}
done

python finetune/results_parser.py\
--seen ${results_path}${id}_metrics_seen.txt\
--unseen ${results_path}${id}_metrics_seen.txt\
--results_path ${results_path}






