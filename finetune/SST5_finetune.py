import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

from SST5DataModule import SST5DataModule
from SSTDataset5Labels import SSTDataset5Labels
from SST5Classifier import SST5Classifier


RANDOM_SEED = int(args.seed)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = int(args.batch)
N_EPOCHS = int(args.epoch)

def fine_tune():
    train_df = pd.read_csv(f"datasets/SST5/sst5_train_set.csv")
    test_df = pd.read_csv(f"datasets/SST5/sst5_test_set_1.csv")
    test2_df = pd.read_csv(f"datasets/SST5/sst5_test_set_2.csv")

    # create data module (train and test)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    data_module = SST5DataModule(train_df, test_df, tokenizer, batch_size=BATCH_SIZE, max_token_len=128)
    data_module.setup()

    # create model
    model = SST5Classifier(args.bert_model)
    checkpoint_callback = ModelCheckpoint(save_last=True)

    # train + test (seen test set)
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, gpus=1, max_epochs=N_EPOCHS)
    trainer.fit(model, data_module)
    trainer.test(model, ckpt_path=None)

    results_dict = model.last_test_results_dict
    results_df = pd.DataFrame(data=results_dict)
    results_df.to_csv(f"{args.results_path}{args.model_id}_{args.seed}_seen.csv", index=False)
    with open(f"{args.results_path}{args.model_id}_metrics_seen.txt", 'a') as f:
        print(trainer.callback_metrics, file=f)

    # test (unseen test set)
    test2_dataset = SST5Dataset(test2_df, tokenizer, max_token_len=128)
    test2 = DataLoader(test2_dataset, batch_size=BATCH_SIZE, num_workers=72)
    trainer.test(test_dataloaders=test2, ckpt_path=None)

    results_dict = model.last_test_results_dict
    results_df = pd.DataFrame(data=results_dict)
    results_df.to_csv(f"{args.results_path}{args.model_id}_{args.seed}_unseen.csv", index=False)
    with open(f"{args.results_path}{args.model_id}_metrics_unseen.txt", 'a') as f:
        print(trainer.callback_metrics, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--bert_model", help="path to bert model (e.g., bert-base-uncased)")
    parser.add_argument("-e", "--epoch", help="number of epochs")
    parser.add_argument("-s", "--seed", help="seed")
    parser.add_argument("-t", "--tokenizer", help="path to tokenizer folder")
    parser.add_argument("-b", "--batch", default=8, help="batch size")
    parser.add_argument("--results_path", default='./', help="path to folder for results. i.e: experiment/results/")
    parser.add_argument("--model_id", default='model', help="identifier for model. i.e.: example: bert_base")
    args = parser.parse_args()

    fine_tune()
    
