import torch.nn as nn
from transformers import BertModel, BertConfig


config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)


class BertClassifier(nn.Module):

    def __init__(self, label_num):
        super(BertClassifier, self).__init__()

        self.label_num = label_num
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, label_num)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        _, pooled_output = self.bert(input_ids=x, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.classifier(dropout_output)
        return linear_output
