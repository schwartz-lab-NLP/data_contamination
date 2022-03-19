bert_model=**path_to_prertained_model**
seed=**seed**
tokenizer=**path_to_tokenizer**
results_path=**path_to_results_folder**
id=**model_identifier**

python SST5_finetune.py\
--bert_model ${bert_model}\
--epoch 3\
--seed ${seed}\
--tokenizer ${tokenizer}\
--batch 8\
--results_path ${results_path}\
--model_id ${id}

# get combined results file (expl)
python results_parser.py\
--seen ${results_path}${id}_metrics_seen.txt\
--unseen ${results_path}${id}_metrics_seen.txt\
--results_path ${results_path}