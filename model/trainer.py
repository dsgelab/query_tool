max_seq_length = 64

tokenizer = BertTokenizer(vocab_file='/Users/feiwang/Documents/Projects/biobert/biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

# check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_json_file('biobert_v1.1_pubmed/config.json')

# load the pretained biobert model
tmp_d = torch.load('biobert_v1.1_pubmed/pytorch_model.bin', map_location=device)
state_dict = OrderedDict()
for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]


def text_to_tokens(sentence):
    # convert a question to a list of words
    word_list = nltk.word_tokenize(sentence)

    # convert a list of words to a (longer) list of tokens defined by vocab_file
    tokens = []
    for word in word_list:
        tokenized_word = tokenizer.tokenize(word)
        tokens.extend(tokenized_word)

    # drop if token is longer than max_seq_length
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]

    return tokens


def process_sentences(sentences):
    list_of_token_lists = [text_to_tokens(sentence) for sentence in sentences]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(token_list) for token_list in list_of_token_lists],
                              maxlen=max_seq_length, dtype="long", value=0.0,
                              truncating="post", padding="post")
    # attention masks make explicit reference to which tokens are actual words vs padded words
    # e.g. [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] a list of a 6-word sentence and 3 pads
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    tensor_inputs = torch.tensor(input_ids)
    tensor_masks = torch.tensor(attention_masks)
    return tensor_inputs, tensor_masks

