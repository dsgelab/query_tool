import numpy as np
import pandas as pd
import time, tqdm
import gc
import matplotlib.pyplot as plt
import re

# a = '/data/processed_data/detailed_longitudinal/detailed_longitudinal.csv'
first_event_file = '/data/processed_data/endpointer/wide_first_events_endpoints_2021-09-04_densified.txt'
demo_file = '/data/processed_data/minimal_phenotype/minimal_phenotype_file.csv'
# d = '/data/processed_data/endpointer/main/finngen_endpoints_longitudinal_04-09-2021_v3.txt.ALL'
death_file = '/data/processed_data/sf_death/thl2019_1776_ksyy_vuosi.csv.finreg_IDsp'
'thl2019_1776_ksyy_tutkimus.csv.finreg_IDsp'

demo = pd.read_csv(demo_file)
death = pd.read_csv('/home/fwang/cod.txt', sep='\t', header=None)
death.columns = ['FINREGISTRYID', 'EVENT_TYPE', 'EVENT_AGE', 'EVENT_YEAR', 'ICDVER', 'ENDPOINT']
death['age'] = np.floor(death.EVENT_AGE)
cod = pd.read_csv(death_file)
death_df = pd.merge(death, cod[['TNRO','VUOSI','TPKS','TPKSAIKA']], how='inner',left_on=['FINREGISTRYID','EVENT_YEAR'], right_on=['TNRO','VUOSI'])
ep = pd.read_csv(first_event_file)

death_first = pd.merge(death_df[['FINREGISTRYID','age','sex','EVENT_YEAR','ENDPOINT']], ep, how='inner',left_on=['FINREGISTRYID','ENDPOINT'], right_on=['FINNGENID','ENDPOINT'])
death_first = death_first[['FINNGENID','ENDPOINT', 'sex','age','EVENT_YEAR','AGE','YEAR']]
death_first.columns = ['id', 'endpoint', 'sex', 'death_age', 'death_year', 'first_age', 'first_year']
death_first['first_age'] = np.floor(death_first['first_age'])
res = death_first.groupby(['endpoint','sex','death_age','death_year','first_age','first_year']).count()
res = pd.DataFrame(res.to_records())
res.columns = ['endpoint', 'sex', 'death_age', 'death_year', 'first_age', 'first_year', 'nevt']
res.to_csv('mortality_first_event.txt',index=False)

cod = death_df[['sex','ENDPOINT','EVENT_YEAR','age','FINREGISTRYID']].groupby(['ENDPOINT','age','EVENT_YEAR','sex']).count()
cod = pd.DataFrame(cod.to_records())
cod.columns = ['endpoint','age','year','sex','nevt']
cod = cod.sort_values(['endpoint','sex','year','age'])
cod = cod[['endpoint','sex','year','age','nevt']]
cod.to_csv('mortality_cause.txt',index=False)

ep = pd.merge(ep, demo[['FINREGISTRYID','sex']], how='left', left_on='FINNGENID', right_on='FINREGISTRYID')
ep['age'] = np.floor(ep.AGE)
ep = ep[['sex','ENDPOINT','YEAR','age','NEVT']].groupby(['ENDPOINT','age','YEAR','sex']).count()
ep = pd.DataFrame(ep.to_records())
ep.columns = ['endpoint','age','year','sex','nevt']
ep = ep.sort_values(['endpoint','sex','year','age'])
ep = ep[['endpoint','sex','year','age','nevt']]
ep.to_csv('first_event.txt',index=False)
