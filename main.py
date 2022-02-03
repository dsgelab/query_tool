import sqlite3
import pandas as pd
import numpy as np
import pickle, json
from utility import *
from model_utils import *
# from model_utils import BertClassifier
import torch
import re
from fuzzywuzzy import process
from flask import Flask, request, jsonify
# from ontoma import OnToma
from transformers import BertTokenizer
import warnings
warnings.filterwarnings("ignore")

# define a list of paths
db_path = '/home/fey/Projects/query_tool_contents/registry.sl3'
trainer_clf_path = '/home/fey/Projects/query_tool_contents/new_clf_trainer.pickle'
trainer_nlg_path = '/home/fey/Projects/query_tool_contents/trainer_nlg.pickle'
word_list_path = '/home/fey/Projects/query_tool_contents/latest_words.pickle'
ep_path = '/home/fey/Projects/query_tool_contents/ep_'
onto_mapping_path = '/home/fey/Projects/query_tool_contents/out_ontology_r6v1.json'
efo_data_path = '/home/fey/Projects/query_tool_contents/efo_data'

with open(word_list_path, 'rb') as f:
    words = pickle.load(f)
word2vec = {word: i for i, word in enumerate(words)}
vec2word = {i: word for i, word in enumerate(words)}
del words

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

with open(trainer_clf_path, 'rb') as f:
    trainer_clf = pickle.load(f)
with open(trainer_nlg_path, 'rb') as f:
    trainer_nlg = pickle.load(f)

action_mapping = {
    'select avg ( age ) from long_registry': 'SELECT age,counts FROM fevent_mean_age WHERE',
    'select count ( * ) from long_registry': 'SELECT counts FROM fevent_count WHERE'
}

# prepare dictionaries for finngen ep mapping
ep_full = json.load(open(ep_path + 'full.json', 'r'))
ep_omit = json.load(open(ep_path + 'omit.json', 'r'))

otmap = pd.read_csv(efo_data_path+'/synonyms.tsv', sep='\t')
terms = pd.read_csv(efo_data_path+'/terms.tsv', sep='\t')

onto = json.load(open(onto_mapping_path, 'r'))
onto_efo, onto_desc = {}, {}
for i in onto.keys():
    onto_result = onto[i].get('EFO', 'None')
    if onto_result != 'None':
        if len(onto_result) == 1:
            onto_efo[i] = onto[i]['EFO'][0]
        else:
            for j in range(len(onto[i]['EFO'])):
                onto_efo[i + '-' + str(j)] = onto[i]['EFO'][j]
    if onto[i].get('description'):
        onto_desc[i] = onto[i]['description']

app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route("/translate", methods=["POST"])
def translate():
    txt = ''
    prompt = request.json["prompt"]

    res = clf(prompt, trainer_clf)
    if res == 0:
        res = 'No answer'
        ep_list, txt = {}, 'Sorry, this is an irrelevant question. '
    else:
        res = nlg('input : ' + prompt, trainer_nlg)
        print(res)
        if 'long_registry' in res:
            ep_list, txt = capture_ep(res, txt)
        else:
            ep_list, txt = {}, 'Sorry, this is an irrelevant question. '

    print(ep_list)
    print(txt)
    msg = {
        'answer': res,
        'ep_list': ep_list,
        'text': txt
    }
    # print(msg['answer'])
    # print(msg['question_index'])

    return jsonify(message=msg)


def clf(prompt, trainer):
    input_id = tokenizer(prompt, padding='max_length', max_length=64,
                         truncation=True, return_tensors="pt")['input_ids']
    output = trainer.model(input_id)
    output = torch.sigmoid(output)[:, 1]
    output = torch.tensor([1 if i >= 0.5 else 0 for i in output])

    return output.item()


def nlg(prompt, trainer, max_len=40):
    if re.search('^\w[\?\.]$', prompt[-2:]):
        prompt = prompt[:-1] + ' ?'
    prompt = prompt.split(' ')
    prompt += ['<na>'] * (max_len - len(prompt))
    x = torch.tensor([word2vec.get(s, word2vec['<unknown>']) for s in prompt], dtype=torch.long)[None, ...].to(
        trainer.device)
    y = sample(trainer.model, x, steps=40, temperature=0.01, sample=True, top_k=10)[0]
    output = re.sub(' <na>', '', ' '.join([vec2word[int(i)] for i in y]))
    return output


def capture_ep(nlg_res, txt):  # todo: replaced by lstm model
    ep = re.findall('endpoint = " (.+) "', nlg_res)
    if len(ep) == 1:
        eps = map_onto(ep[0])
        if type(eps) == str:
            return [], txt + txt_dict['ep_failed_to_find']
        else:
            return eps, txt
    else:
        return [], txt + txt_dict['ep_failed_to_find']


def map_onto(disease):
    try:
        onto = otmap[otmap.normalised_synonym == disease]
        if len(onto) == 0:
            mapped_eps = process.extract(disease, ep_full.values())[:3]
            if mapped_eps[1][1] < 75:
                return 'fail'
            else:
                mapped_ep_id = [list(ep_full.keys())[list(ep_full.values()).index(i[0])] for i in mapped_eps]
                return mapped_ep_id
        mapped_id = onto.iloc[0, 1].split(':')
        if mapped_id[0] != 'EFO':
            label = terms[terms.normalised_id == onto.iloc[0, 1]].normalised_label.tolist()[0]
            mapped_eps = process.extract(label, ep_full.values())[:3]
            mapped_ep_id = [list(ep_full.keys())[list(ep_full.values()).index(i[0])] for i in mapped_eps]
            return mapped_ep_id
        else:
            efo = mapped_id[1]
            return get_keys_by_value(onto_efo, efo)
    except Exception as e:
        print(e)
        return 'error'


def write_query(nlg_res, ep):
    txt = ''
    query_parts = re.findall('output : ([\w\s\(\)*]+) where (.+)$', nlg_res)
    print(query_parts)
    query = action_mapping[query_parts[0][0]]
    query += ' ep = "' + ep + '"'
    if 'year' in query_parts[0][1]:
        yr_range = re.findall('year between (\d+) and (\d+)', query_parts[0][1])
        if yr_range == []:
            yr_end = re.findall('year < (\d+)', query_parts[0][1])
            if yr_end == []:
                yr_start = re.findall('year > (\d+)', query_parts[0][1])
                if yr_start == []:
                    yr_start = re.findall('year >= (\d+)', query_parts[0][1])
                    if yr_start == []:
                        yr_end = re.findall('year <= (\d+)', query_parts[0][1])
                        if yr_end == []:
                            yr_range = re.findall('year = (\d+)', query_parts[0][1])
                            if yr_range == []:
                                query_yr, txt_yr = '', ''
                            else:
                                query_yr, txt_yr = find_yr_range(int(yr_range[0]), int(yr_range[0]), txt)
                        else:
                            query_yr, txt_yr = find_yr_range(2001, int(yr_end[0]), txt)
                    else:
                        query_yr, txt_yr = find_yr_range(int(yr_start[0]), 2020, txt)
                else:
                    query_yr, txt_yr = find_yr_range(int(yr_start[0]) + 1, 2020, txt)
            else:
                query_yr, txt_yr = find_yr_range(2001, int(yr_end[0]) - 1, txt)
        else:
            query_yr, txt_yr = find_yr_range(int(yr_range[0][0]), int(yr_range[0][1]), txt)
        query += query_yr
        txt += txt_yr
        print(query_yr)
    if 'sex' in query_parts[0][1]:
        if re.findall('sex = female', query_parts[0][1]):
            query += ' AND sex = 2'
        elif re.findall('sex = male', query_parts[0][1]):
            query += ' AND sex = 1'
    if 'age' in query_parts[0][1]:
        age_range = re.findall('age between (\d+) and (\d+)', query_parts[0][1])
        if age_range == []:
            age_end = re.findall('age < (\d+)', query_parts[0][1])
            if age_end == []:
                age_start = re.findall('age > (\d+)', query_parts[0][1])
                if age_start == []:
                    query_age = ''
                else:
                    query_age, txt_age = find_age_range(int(age_start[0]) + 1, 111, txt)
            else:
                query_age, txt_age = find_age_range(0, int(age_end[0]) - 1, txt)
        else:
            query_age, txt_age = find_age_range(int(age_range[0][0]), int(age_range[0][1]), txt)
        query += query_age
        txt += txt_age
    return query, txt


def query_data(query):
    conn = sqlite3.connect(db_path)
    table = pd.read_sql_query(query, conn)
    zeros = table.counts.isin([0]).sum()
    if 'fevent_mean_age' in query:
        if zeros != 0:
            table['counts'] = table.counts.replace(0, 1)
        res = round(np.dot(table.age.astype(float), table.counts.astype(int)) / table.counts.astype(int).sum(), 2)
        output = 'Mean age of your target group is ' + str(res) + '.'
    elif 'fevent_count' in query:
        res = table.counts.astype(int).sum()
        output = 'Total number of your target group is ' + str(res) + '.'
        if zeros != 0:
            min_num = zeros + res
            max_num = 4 * zeros + res
            output = 'Total number of your target group is between ' + str(min_num) + ' and ' + str(max_num) + '.'

    return output


@app.route("/getResult", methods=["POST"])
def getResult():
    nlg_res = request.json["nlg_res"]
    ep = request.json["ep"]
    # print(ep)
    # print('nlg_res: ',nlg_res)
    query, txt = write_query(nlg_res, ep)
    output = query_data(query)

    msg = {
        'output': output,
        'query': query,
        'text': txt
    }

    return jsonify(message=msg)

    # try:
    #     write_query(nlg_res, ep, txt)
    #     return jsonify(
    #         message={"part1": "Your risk of having " + outcome + " is ", "part2": "{0:.2%}".format(abs_risk)})
    # except IndexError as e1:
    #     print('Error type:' + str(e1))
    #     return jsonify(message={"part1": "No record is found with the input: \n", "part2": str(
    #         [prior, outcome, datetime.datetime.today().year - int(result[2]), result[3]])})


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
