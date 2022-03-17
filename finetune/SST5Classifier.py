import torch
import torchmetrics
import torch.nn as nn
from transformers import BertModel, AdamW
import pytorch_lightning as pl


class SST5Classifier(pl.LightningModule):

    def __init__(self, bert_path):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 5)  # 5 for 5 labels

        self.accuracy = torchmetrics.Accuracy()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        
        self.last_test_results_dict = {}

    def forward(self, input_ids, attention_mask, label=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        loss = 0
        if label is not None:  # label is None in inference mode
            loss = self.loss(output, label.flatten())
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, output = self(batch['input_ids'], batch['attention_mask'], batch['label'])
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss, 'prediction': output, 'label': batch['label']}

    def test_step(self, batch, batch_idx):
        loss, output = self(batch['input_ids'], batch['attention_mask'], batch['label'])
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.accuracy(self.softmax(output), label.flatten())
        self.log('test_acc_step', self.accuracy)
        return {'loss': loss, 'prediction': output, 'review': batch['review'], 'label': batch['label']}

    def test_epoch_end(self, outputs):
        """ This function saves the test's predictions in a dictionary. """
        correct_count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        label_count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        test_dict = {'review': [], 'label': [], 'pred': [], 'confidence_score': [], 'acc': []}
        for batch in range(len(outputs)):
            reviews = outputs[batch]['review']
            preds = torch.argmax(self.softmax(outputs[batch]['prediction']), dim=1)
            confidence_score= torch.max(self.softmax(outputs[batch]['prediction']), dim=1).values
            true_labels = outputs[batch]['label'].flatten()

            for i in range(len(true_labels)):
                test_dict['review'].append(reviews[i])
                test_dict['label'].append(true_labels[i].item())
                test_dict['pred'].append(preds[i].item())
                test_dict['confidence_score'].append(confidence_score[i].item())
                test_dict['acc'].append(int(preds[i] == true_labels[i]))

                label_count_dict[true_labels[i].item()] += 1
                if preds[i] == true_labels[i]:
                    correct_count_dict[true_labels[i].item()] += 1

        for key in label_count_dict.keys():
            print(f"acc for {key}: ", correct_count_dict[key]/label_count_dict[key])
        print(f"total acc: ", sum(correct_count_dict.values())/sum(label_count_dict.values()))
        self.last_test_results_dict = test_dict

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return [optimizer]
