import re
import tqdm
import copy
import random as rnd
import numpy as np
from collections import Counter
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
logger = logging.getLogger(__name__)


class CLF_Dataset(Dataset):
    def __init__(self, x_y_list, max_seq_len):
        self.x_y_list = x_y_list
        self.max_length = max_seq_len

    def __getitem__(self, index):
        question = self.x_y_list[0][index]
        X = tokenizer(question, padding='max_length', max_length=self.max_length,
                      truncation=True, return_tensors="pt")['input_ids']

        group = self.x_y_list[1][index]
        y = torch.from_numpy(np.array(group))

        return X, y

    def __len__(self):
        return len(self.x_y_list[0])



class TrainerConfig:
    # optimization parameters
    epochs = 10
    batch_size = 128
    learning_rate = 3e-4
    max_seq_length = 64
    # betas = (0.9, 0.95)
    # grad_norm_clip = 1.0
    # weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # lr_decay = False
    # warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    # final_tokens = 260e9 # (at what point it reaches 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, data_dict, config):
        self.model = model
        self.data_dict = data_dict
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        train_x_1 = self.data_dict['train_x_1']
        train_x_0 = self.data_dict['train_x_0']
        valid_x_1 = self.data_dict['valid_x_1']
        valid_x_0 = self.data_dict['valid_x_0']

        len_train = len(train_x_1) * 2
        len_val = len(valid_x_1) * 2

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        for epoch_num in range(config.epochs):
            this_train_X = train_x_1 + [i[0] for i in rnd.sample(train_x_0, len(train_x_1))]
            this_train_y = [1] * len(train_x_1) + [0] * len(train_x_1)
            this_valid_X = valid_x_1 + [i[0] for i in rnd.sample(valid_x_0, len(valid_x_1))]
            this_valid_y = [1] * len(valid_x_1) + [0] * len(valid_x_1)

            train_dataset = CLF_Dataset([this_train_X, this_train_y], config.max_seq_length)
            valid_dataset = CLF_Dataset([this_valid_X, this_valid_y], config.max_seq_length)

            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            val_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm.tqdm(train_dataloader):
                train_label = train_label.to(device)
                input_id = train_input.squeeze(1).to(device)
                output = model(input_id)
                #             print('output',output)
                output = torch.sigmoid(output)[:, 1]

                #             print('output',output)
                #             print('true  ',train_label)

                batch_loss = criterion(output, train_label.float())
                total_loss_train += batch_loss.item()

                train_pred = torch.tensor([1 if i >= 0.5 else 0 for i in output])

                acc = (train_pred == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    input_id = val_input.squeeze(1).to(device)
                    output = model(input_id)
                    output = torch.sigmoid(output)[:, 1]

                    batch_loss = criterion(output, val_label.float())
                    total_loss_val += batch_loss.item()
                    val_pred = torch.tensor([1 if i >= 0.5 else 0 for i in output])

                    acc = (val_pred == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len_train: .4f} \
                | Train Accuracy: {total_acc_train / len_train: .4f} \
                | Val Loss: {total_loss_val / len_val: .4f} \
                | Val Accuracy: {total_acc_val / len_val: .4f}')





