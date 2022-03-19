from torch.utils.data import DataLoader
import pytorch_lightning as pl

from SST5Dataset import SST5Dataset


class SST5DataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, tokenizer, batch_size, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = SST5Dataset(self.train_df,
                                         self.tokenizer,
                                         self.max_token_len)

        self.test_dataset = SST5Dataset(self.test_df,
                                        self.tokenizer,
                                        self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=72)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=72)
