import numpy as np
import torch
import re
import pickle, json
from trainer import QandADataset
from utils import sample

max_len = 40
words = pickle.load(open('latest_words.pickle', 'rb'))
word2vec = { word:i for i,word in enumerate(words) }
vec2word = { i:word for i,word in enumerate(words) }



def preprocess_data(data, max_len):
    data = [i.lower() for i in data]
    q, a = zip(*[(i.split('\n')[0], i.split('\n')[1]) for i in data])
    q = [re.findall(r"\w+|[^\w\s]", i, re.UNICODE) for i in q]
    a = [re.findall(r"\w+|[^\w\s]", i, re.UNICODE) for i in a]

    m = np.array([0 if len(i) <= max_len else 1 for i in q])
    n = np.array([0 if len(i) <= max_len else 1 for i in a])
    qna_filter = [True if i == 0 else False for i in m + n]
    q = np.array(q)[qna_filter]
    a = np.array(a)[qna_filter]

    # so suppose the max of each is 50
    q = [word_list+['<na>']*(max_len-len(word_list)) for word_list in q]
    a = [word_list+['<na>']*(max_len-len(word_list)) for word_list in a]
    
    return q, a

def write_query(elements):
    part2nd = elements[1].split('-')
    part3rd = elements[2]
    for i in re.findall('((endpoint|outcome|prior)=(\w|\!))',part3rd):
        part3rd = re.sub(i[0],i[1]+' = '+i[2],part3rd)
    for i in re.findall('(endpoint|outcome|prior) = ([\w\s\-\!\']+)(,|$)',part3rd):
        part3rd = re.sub(i[1],'"'+i[1]+'"',part3rd)
    return 'INPUT: '+elements[0]+'\nOUTPUT: SELECT '+part2nd[0]+' FROM '+part2nd[1]+' WHERE '+part3rd.replace(', ',' AND ')

def load_data():
    data_train = json.load(open('custom_data_train.txt','r'))
    data_valid = json.load(open('custom_data_valid.txt','r')) 

    data_train = [write_query(i) for i in data_train]
    data_valid = [write_query(i) for i in data_valid]

    train_q, train_a = preprocess_data(data_train, max_len)
    valid_q, valid_a = preprocess_data(data_valid, max_len)
    train_dataset = QandADataset(train_q, train_a, max_len)
    valid_dataset = QandADataset(valid_q, valid_a, max_len)
    return train_dataset, valid_dataset    

def disease_replace(sentence):
    for i in re.findall('((endpoint|outcome|prior) = " ([\w\s\-\!\']+) ")( |$)',sentence):
        sentence = re.sub(i[0],i[1]+' = " disease "',sentence)
    return sentence

def main():
    # load data
    train_dataset, valid_dataset = load_data()

    # load pre-trained model
    trainer = pickle.load(open('pre_trainer.pickle', 'rb'))

    # run the model
    trainer.train_dataset = train_dataset
    trainer.config.max_epochs = 10
    trainer.train()

    # test model
    out_file = open('results.txt', 'x')
    res1, res2 = [], []
    for i in range(len(valid_q)):
        x = torch.tensor([word2vec[s] for s in valid_q[i]], dtype=torch.long)[None,...].to(trainer.device)
        y = sample(trainer.model, x, steps = max_len, temperature=0.1, sample=True, top_k=10)[0]
        answer = re.sub(' <na>', '', ' '.join([vec2word[int(i)] for i in y][max_len:]))
        true_answer = re.sub(' <na>', '', ' '.join(valid_a[i]))

        # write the result to the output file
        print(i, file=out_file)
        print('The     question: ', re.sub(' <na>', '', ' '.join(valid_q[1])), file=out_file)
        print('Predicted answer: ', answer, file=out_file)
        print('True      answer: ', true_answer, file=out_file)
        print('',file=out_file)

        if true_answer == answer:
            res1.append(1)
        else:
            res1.append(0)
        
        answer = disease_replace(answer)
        true_answer = disease_replace(true_answer)
        if true_answer == answer:
            res2.append(1)
        else:
            res2.append(0)

            # print errors
            print('The     question: ',re.sub(' <na>','',' '.join(valid_q[i])))
            print('Predicted answer: ',answer)
            print('True      answer: ',true_answer)
            print()

    out_file.close()

    print('Accuracy: ',np.sum(res1)/len(res1))
    print('Accuracy after removing disease name: ',np.sum(res2)/len(res2))

if __name__ == "__main__":
    main()
