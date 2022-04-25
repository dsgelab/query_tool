class BioBertNER(nn.Module):

    def __init__(self, label_num, config, state_dict):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.load_state_dict(state_dict, strict=False)
        self.dropout = nn.Dropout(p=0.3)
        self.linear_output = nn.Linear(self.bert.config.hidden_size, label_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_layer = outputs[0] # torch.Size([1, max_len, hidden_weights])
#         pool_layer = outputs[1]
        output = self.dropout(encoded_layer)
        output = self.linear_output(output)
        return output.argmax(-1)