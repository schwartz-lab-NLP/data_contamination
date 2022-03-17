import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

LABEL_COLUMNS = ['sentiment']


class SST5Dataset(Dataset):

  def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int=128):
    self.data = data
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    review = data_row.review
    label = data_row[LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True, 
      return_attention_mask=True,
      return_tensors="pt" # pt = pytorch tensor
    )

    return dict(
        review = review,
        input_ids = encoding['input_ids'].flatten(),
        attention_mask = encoding['attention_mask'].flatten(),
        label = torch.LongTensor(label)
    )
  