import argparse
import pandas as pd
from transformers import AutoModelForMaskedLM, BertTokenizer
import torch.nn as nn
import numpy as np

LABELS = [0, 1, 2, 3, 4]

def test_mlm(id=1):

    # get masked lines for inference and the data set with labels
    if id == 1 or id == 2:
        with open(f'data/SST5/sst5_test_set_{id}_review_mask.txt', 'r') as f:
            masked_lines = f.read().splitlines()
        test_df = pd.read_csv(f"data/SST5/sst5_test_set_{id}.csv")
    else:
        return

    # get tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    softmax = nn.Softmax(dim=0)

    # inference
    top_label = []
    labels_probs = {}
    for i in LABELS:
        labels_probs[f'label_{i}_prob'] = []

    for index, line in enumerate(masked_lines):
        input = tokenizer.encode(line, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        token_logits = model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        # get label tokens probabilities
        probs = []
        for i in LABELS:
            label_token = str(i)
            label_token_id = tokenizer.encode(label_token, add_special_tokens=False)[0]
            token_score = mask_token_logits[:, label_token_id]
            token_score = token_score.detach().numpy()[0]
            probs.append(token_score)
        probs = softmax(torch.Tensor(probs))

        for i in LABELS:
            labels_probs[f'label_{i}_prob'].append(probs[i].item())

        top_label.append(np.argmax(labels_probs).item())

    # add preds to df
    for for i in LABELS:
        test_df[f'label_{i}_prob'] = labels_probs[f'label_{i}_prob']
    test_df['top_label'] = top_label
    test_df['correct_pred_by_label'] = (test_df['top_label'] == test_df['sentiment']).astype(int)

    file_name = ""
    if id == 1 or id == 2:
        file_name = f'{args.results_path}{args.name}_sst5_test_set_{id}_preds.csv'
    test_df.to_csv(file_name, index=False)
    
    print(f"mlm accuracy: {test_df['correct_pred_by_label'].mean()}")
    return test_df['correct_pred_by_label'].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bert_model", help="path to bert model (e.g., bert-base-uncased)")
    parser.add_argument("-t", "--tokenizer", help="path to tokenizer folder")
    parser.add_argument("--results_path", default='./', help="path to folder for results. i.e: experiment/results/")
    parser.add_argument("--model_id", default='model', help="identifier for model. i.e.: example: bert_base")

    args = parser.parse_args()

    with open(f'{args.results_path}{args.name}_mem.txt', 'w') as f:
        mlm_acc_1 = test_mlm(1)
        print(f"test_set_1: ", file=f)
        print(f"mlm_accuracy", test_mlm(1), file=f)

    with open(f'{args.results_path}{args.name}_mem.txt', 'a') as f:
        mlm_acc_2 = test_mlm(2)
        print(f"test_set_2: ", file=f)
        print(f"mlm_accuracy", test_mlm(2), file=f)

        print(f"diff: ", file=f)
        print(f"diff_mlm_accuracy (mem)", mlm_acc_1 - mlm_acc_2, file=f)



