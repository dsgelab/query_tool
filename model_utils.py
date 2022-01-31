import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)


# class BertClassifier(nn.Module):
#
#     def __init__(self, label_num):
#         super(BertClassifier, self).__init__()
#
#         self.label_num = label_num
#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, label_num)
#         #         self.relu = nn.ReLU()
#         nn.init.xavier_normal_(self.classifier.weight)
#
#     def forward(self, x):
#         _, pooled_output = self.bert(input_ids=x, return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.classifier(dropout_output)
#         return linear_output


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


class QandADataset(Dataset):
    def __init__(self, questions, answers, max_length):
        self.vocab_size = len(words)
        self.max_len = max_length
        self.block_size = max_length*2 - 1
        self.q = questions
        self.a = answers
        
    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        # grab a chunk of words from the data (q and a)
        q_chunk = self.q[idx]
        a_chunk = self.a[idx]
        # encode every word to an integer
        q_vec_chunk = [word2vec.get(s, word2vec['<na>']) for s in q_chunk]
        a_vec_chunk = [word2vec.get(s, word2vec['<na>']) for s in a_chunk]
        
        vec_chunk = q_vec_chunk + a_vec_chunk
        
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(vec_chunk[:-1], dtype=torch.long)
        y = torch.tensor(vec_chunk[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.max_len-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y


# class NLG_Dataset(Dataset):
#     def __init__(self, x_y_list, max_length):
#         self.x_y_list = x_y_list
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.x_y_list[0])

#     def __getitem__(self, index):
#         # grab a chunk of words from the data (q and a)
#         question = self.x_y_list[0][index]
#         q_vec_chunk = tokenizer(question, padding='max_length', max_length=self.max_length,
#                                 truncation=True, return_tensors="pt")['input_ids']

#         answer = self.x_y_list[1][index]
#         a_vec_chunk = tokenizer(answer, padding='max_length', max_length=self.max_length,
#                                 truncation=True, return_tensors="pt")['input_ids']

#         vec_chunk = q_vec_chunk.squeeze(0).tolist() + a_vec_chunk.squeeze(0).tolist()

#         # x will be input to GPT and y will be the associated expected outputs
#         X = torch.tensor(vec_chunk[:-1], dtype=torch.long)
#         y = torch.tensor(vec_chunk[1:], dtype=torch.long)  # predict the next token in the sequence
#         y[:self.max_length - 1] = -100  # we will only train in the output locations. -100 will mask loss to zero
#         return X, y


