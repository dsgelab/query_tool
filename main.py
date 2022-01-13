import sqlite3
import pandas as pd
import numpy as np
import pickle, json
from utility import *
from mingpt.trainer import QandADataset
from mingpt.utils import sample
import torch
import re
from fuzzywuzzy import process
from flask import Flask, request, jsonify
from ontoma import OnToma

# define a list of paths
db_path = '/Users/feiwang/Documents/Projects/query_tool_contents/registry.sl3'
trainer_nlg_path = '/Users/feiwang/Documents/Projects/query_tool_contents/trainer_nlg.pickle'
chrome_path = '/Users/feiwang/Documents/Projects/query_tool_contents/chromedriver'
word_list_path = '/Users/feiwang/Documents/Projects/query_tool_contents/latest_words.pickle'
ep_path = '/Users/feiwang/Documents/Projects/query_tool_contents/FINNGEN_ENDPOINTS_DF8_Final_2021-09-02.xlsx'
onto_mapping_path = '/Users/feiwang/Documents/Projects/query_tool_contents/out_ontology_r6v1.json'

max_len = 40
words = pickle.load(open(word_list_path, 'rb'))
word2vec = {word: i for i, word in enumerate(words)}
vec2word = {i: word for i, word in enumerate(words)}
trainer_nlg = pickle.load(open(trainer_nlg_path, 'rb'))
action_mapping = {
    'select avg ( age ) from long_registry': 'SELECT age,counts FROM fevent_mean_age WHERE',
    'select count ( * ) from long_registry': 'SELECT counts FROM fevent_count WHERE'
}

# prepare a dictionary for finngen ep mapping
ep_df = pd.read_excel(ep_path, sheet_name='Sheet 1', usecols=['NAME', 'LONGNAME', 'OMIT'])
ep_df = ep_df[ep_df.LONGNAME.notna()]
ep_full = dict(zip(ep_df.NAME, ep_df.LONGNAME))
ep_df = ep_df[ep_df.OMIT.isna()][['NAME', 'LONGNAME']]
ep_df = ep_df.dropna()
ep_omit = dict(zip(ep_df.NAME, ep_df.LONGNAME))

otmap = OnToma()
onto = json.load(open(onto_mapping_path, 'r'))
onto_efo, onto_desc = {}, {}
for i in onto.keys():
    onto_result = onto[i].get('EFO', 'None')
    if onto_result != 'None':
        if len(onto_result) == 1:
            onto_efo[i] = onto[i]['EFO'][0]
        else:
            for j in range(len(onto[i]['EFO'])):
                onto_efo[i+'-'+str(j)] = onto[i]['EFO'][j]
    if onto[i].get('description'):
        onto_desc[i] = onto[i]['description']

app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route("/translate", methods=["POST"])
def translate():
    txt = ''
    prompt = 'input : ' + request.json["prompt"]
    nlg_res = nlg(prompt, trainer_nlg)
    print(nlg_res)
    if 'long_registry' in nlg_res:
        ep_list, txt = capture_ep(nlg_res, txt)
    else:
        ep_list, txt = {}, 'Sorry, this is an irrelevant question. '
    print(ep_list)
    print(txt)
    msg = {
        'answer': nlg_res,
        'ep_list': ep_list,
        'text': txt
    }
    # print(msg['answer'])
    # print(msg['question_index'])

    return jsonify(message=msg)


def nlg(prompt, trainer):
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
    ontos = otmap.find_term(disease)
    try:
        mapped_id = ontos[0].id_normalised.split(':')
        if len(mapped_id) == 0:
            return 'fail'
        if mapped_id[0] != 'EFO':
            mapped_eps = process.extract(ontos[0].label, ep_full.values())[:3]
            mapped_ep_id = [list(ep_full.keys())[list(ep_full.values()).index(i[0])] for i in mapped_eps]
            return mapped_ep_id
        else:
            efo = mapped_id[1]
            return get_keys_by_value(onto_efo, efo)
    except Exception as e:
        print(e)
        return 'error'
#     if len(this) == 1:
#         ontos[0].id_normalised.split(':')[1]
#     else:
#         [ontos[i].id_normalised.split(':')[1] for i in ontos]


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