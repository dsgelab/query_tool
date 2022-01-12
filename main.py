import sqlite3
import pandas as pd
import numpy as np
import pickle
from utility import find_age_range, find_yr_range, txt_dict
from mingpt.trainer import QandADataset
from mingpt.utils import sample
import torch
import re
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from fuzzywuzzy import process
from flask import Flask, request, jsonify

db_path = '/Users/feiwang/Documents/Projects/query_tool_contents/registry.sl3'
trainer_nlg_path = '/Users/feiwang/Documents/Projects/query_tool_contents/trainer_nlg.pickle'
chrome_path = '/Users/feiwang/Documents/Projects/query_tool_contents/chromedriver'
mesh_path = 'https://www.ncbi.nlm.nih.gov/mesh'
word_list_path = '/Users/feiwang/Documents/Projects/query_tool_contents/latest_words.pickle'
ep_path = '/Users/feiwang/Documents/Projects/query_tool_contents/FINNGEN_ENDPOINTS_DF8_Final_2021-09-02.xlsx'

max_len = 40
words = pickle.load(open(word_list_path, 'rb'))
word2vec = {word: i for i, word in enumerate(words)}
vec2word = {i: word for i, word in enumerate(words)}
trainer_nlg = pickle.load(open(trainer_nlg_path, 'rb'))

# Using Chrome to access web
driver = webdriver.Chrome(chrome_path)
driver.get(mesh_path)

action_mapping = {
    'select avg ( age ) from long_registry': 'SELECT age,counts FROM fevent_mean_age WHERE',
    'select count ( * ) from long_registry': 'SELECT counts FROM fevent_count WHERE'
}

# prepare a pedigree
ep_df = pd.read_excel(ep_path, sheet_name='Sheet 1', usecols=['NAME','INCLUDE'])
ep_df = ep_df.dropna()
parents, children = [], []
for i,row in ep_df.iterrows():
    kids = row.INCLUDE.split('|')
    children += kids
    parents += [row.NAME]*len(kids)
ep_tree = pd.DataFrame({'parents': parents, 'children': children})

# prepare a dictionary for finngen ep mapping
ep_df = pd.read_excel(ep_path, sheet_name='Sheet 1', usecols=['NAME','LONGNAME','OMIT'])
ep_df = ep_df[ep_df.OMIT.isna()][['NAME', 'LONGNAME']]
ep_df = ep_df.dropna()
ep_display = dict(zip(ep_df.LONGNAME, ep_df.NAME))

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
        ep_dict, txt = capture_ep(nlg_res, driver, txt)
    else:
        ep_dict, txt = {}, 'Sorry, this is an irrelevant question. '
    print(ep_dict)
    print(txt)
    msg = {
        'answer': nlg_res,
        'ep_dict': ep_dict,
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


def capture_ep(nlg_res, browser, txt):  # todo: replaced by lstm model
    ep = re.findall('endpoint = " (.+) "', nlg_res)
    if len(ep) == 1:
        browser.find_element_by_xpath('//*[@id="term"]').clear()
        browser.find_element_by_xpath('//*[@id="term"]').send_keys(ep[0])
        browser.find_element_by_xpath('//*[@id="search"]').click()
        try:
            mesh_ep = browser.find_element_by_xpath(
                '/html/body/div[1]/div[1]/form/div[1]/div[4]/div/div[5]/div/h1/span').text
        except NoSuchElementException:
            try:
                browser.find_element_by_xpath(
                    '/html/body/div[1]/div[1]/form/div[1]/div[4]/div/div[5]/div[1]/div[2]/p/a').click()
                mesh_ep = browser.find_element_by_xpath(
                    '/html/body/div[1]/div[1]/form/div[1]/div[4]/div/div[5]/div/h1/span').text
            except NoSuchElementException:
                mesh_ep = ep
        ep_options = process.extract(mesh_ep, ep_display.keys())
        ep_options = list(set(list(zip(*ep_options))[0]))[:3]
        ep_codes = [ep_display[i] for i in ep_options]
        children_long = [get_children(ep_code) for ep_code in ep_codes]
        parent_long = [get_parent(ep_code) for ep_code in ep_codes]
        res = dict(zip(['name', 'parent', 'children'], list(zip(ep_options, parent_long, children_long))))
        return dict(zip(ep_codes, res)), txt
    else:
        return {}, txt + txt_dict['ep_failed_to_find']


def get_children(ep):
    children_short = ep_tree[ep_tree.parents == ep].children.tolist()
    children_long = []
    for i in children_short:
        try:
            children_long.append(list(ep_display.keys())[list(ep_display.values()).index(i)])
        except:
            children_long.append('* endpoint omited')
    return children_long


def get_parent(ep):
    parent_short = ep_tree[ep_tree.children == 'F5_BEHAVE'].parents.tolist()[0]
    try:
        parent_long = list(ep_display.keys())[list(ep_display.values()).index(parent_short)]
    except:
        parent_long = '* endpoint omited'
    return parent_long


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