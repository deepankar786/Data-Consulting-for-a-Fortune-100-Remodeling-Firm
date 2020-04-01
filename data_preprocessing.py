# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:55:35 2020

@author: deepa
"""

import pandas as pd
import numpy as np
import zipfile
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
%matplotlib inline
from matplotlib import*
import warnings
warnings.filterwarnings('ignore')
import psutil
import copy

leads = pd.read_csv('leads.csv')
lead_sources = pd.read_csv('lead_sources.csv')

### Creating dataframes

homes = pd.read_csv("homes.csv", usecols = ['id', 'city','state','zip_code','house_style','year_home_built'])
homes.columns = ['id','home_city','home_state','home_zip_code','house_style','year_home_built']

lead_products = pd.read_csv("lead_products.csv", usecols = ['id','lead_id','product_id'])

lead_sources = pd.read_csv("lead_sources.csv", usecols = ['id','home_id','owner_id','taken_by','taken_at','status','created_at'])
lead_sources.columns = ['id','home_id','owner_id','marketter_id','taken_at','ls_status','ls_created_at']

leads = pd.read_csv("leads.csv", usecols = ['id','set_for','result','status','zip_code_id','created_at','set_type','setter_id','confirmer_id','source_id','project_id','lead_source_id'])
leads.columns = ['lead_id','set_for','result','status','owner_zip_id','created_at','set_type','setter_id','confirmer_id','source_id','project_id','lead_source_id']

products = pd.read_csv("products.csv", usecols = ['id','code','name'])
products.columns = ['id','product_code','product_name']

projects = pd.read_csv("projects.csv", usecols = ['id','current_state','price','ceiling_price'])

source_categories = pd.read_csv("source_categories.csv", usecols = ['id','description'])
source_categories.columns = ['id','src_catg_desc']

source_groups = pd.read_csv("source_groups.csv", usecols = ['id','description'])
source_groups.columns = ['id','src_grp_desc']

sources = pd.read_csv("sources.csv", usecols = ['id','source','source_category_id','source_group_id'])
sources.columns = ['id','src_desc','source_category_id','source_group_id']

territories = pd.read_csv("territories.csv", usecols = ['id','name','time_zone'])
territories.columns = ['id','territory_name','time_zone']

zip_codes = pd.read_csv("zip_codes.csv", usecols = ['id','zip_code', 'state', 'territory_id'])
zip_codes.columns = ['id','zip_code', 'zip_state', 'territory_id']

users = pd.read_csv("users.csv", usecols = ['id','started_on'])

### Merging
#### Lead - Lead_source

base_table = pd.merge(leads, lead_sources, how='left', on=None, left_on='lead_source_id', right_on='id',
         suffixes=('', '_ls'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Source - Source_categories

base_table_source_1 = pd.merge(sources, source_categories, how='left', on=None, left_on='source_category_id', right_on='id',
         suffixes=('', '_src_catg'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Source - Source_categories - Source_groups

base_table_source_2 = pd.merge(base_table_source_1, source_groups, how='left', on=None, left_on='source_group_id', right_on='id',
         suffixes=('', '_src_grp'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Leads - Source

base_table_1 = pd.merge(base_table, base_table_source_2, how='left', on=None, left_on='source_id', right_on='id',
         suffixes=('', '_sources'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Leads - Source - Homes

base_table_2 = pd.merge(base_table_1, homes, how='left', on=None, left_on='home_id', right_on='id',
         suffixes=('', '_homes'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Leads - Source - Homes - Projects

base_table_3 = pd.merge(base_table_2, projects, how='left', on=None, left_on='project_id', right_on='id',
         suffixes=('', '_projects'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Zip_codes - Territory

base_zip = pd.merge(zip_codes, territories, how='left', on=None, left_on='territory_id', right_on='id',
         suffixes=('', '_territory'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Leads - Source - Homes - Projects - Zip_Terr

base_table_4 = pd.merge(base_table_3, base_zip, how='left', on=None, left_on='owner_zip_id', right_on='id',
         suffixes=('', '_zip_terr'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

#### Leads - Source - Homes - Projects - Zip_Terr - Products

lead_products['product_id'] = lead_products.apply(lambda lead_products: 3 if (lead_products['product_id'] == 6) else lead_products['product_id'], axis=1)

lead_products['product_id'] = lead_products.apply(lambda lead_products: 7 if (lead_products['product_id'] == 9) else lead_products['product_id'], axis=1)

lead_products['product_id'] = lead_products.apply(lambda lead_products: 7 if (lead_products['product_id'] == 10) else lead_products['product_id'], axis=1)

lp = lead_products.loc[lead_products.groupby('lead_id').product_id.idxmin()].reset_index(drop=True)

def product_group(product_id):
    if product_id == 1:
        return 'WINDOW'
    elif product_id == 2:
        return 'SIDINGS'
    elif product_id == 3:
        return 'DOORS'
    elif product_id == 4:
        return 'GUTTERS'
    elif product_id == 5:
        return 'ROOFING'
    elif product_id == 7:
        return 'SOLAR'
    else:
        return 'OTHERS'

lp['product'] = lp.apply(lambda lp: product_group(lp['product_id']),axis=1)

lead_products = lp[['lead_id','product']]

base_table_5 = pd.merge(base_table_4, lead_products, how='left', on=None, left_on='lead_id', right_on='lead_id',
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

### Processing

base_table_5.drop(base_table_5[base_table_5['set_type'] == 'ProjectVisit'].index, inplace = True)

base_table_5['result'] = base_table_5['result'].astype(str)

def target_function(result, status):
    if result == 'No Pitch' and status == 'Issued':
        return 1
    elif result == 'Pitch Miss' and status == 'Issued':
        return 1
    elif result == 'Sold' and status == 'Issued':
        return 1
    elif status == 'Conf Canx' and result == 'Canceled':
        return 0
    elif status == 'Disp Canx' and result == 'Canceled':
        return 0
    elif status == 'Issued Canx' and result == 'Canceled':
        return 0
    elif status == 'Confirmed' and result == 'Unconf':
        return 0
    elif status == 'Not Conf' and result == 'Unconf':
        return 0
    elif status == 'Confirmed' and result == 'nan':
        return 2
    elif status == 'Dispatched' and result == 'nan':
        return 2
    elif status == 'Disp Review' and result == 'nan':
        return 2
    elif status == 'Not Conf' and result == 'nan':
        return 2
    else:
        return 3

base_table_5['target'] = base_table_5.apply(lambda base_table_5: target_function(base_table_5['result'],base_table_5['status']),axis=1)

base_table_5 = base_table_5.loc[base_table_5['target']<2]

base_table['taken_at'] = base_table.apply(lambda base_table: str(base_table['taken_at']),axis = 1)

base_table['taken_at'] = base_table.apply(lambda base_table: '9'*19 if (base_table['taken_at']=='nan') else base_table['taken_at'],axis=1)

base_table['taken_year_str'] = base_table['taken_at'].str[:4]

indexes = base_table[(base_table['taken_year_str'].astype(int) >= 2000) & (base_table['taken_year_str'].astype(int) <= 2019)].index

base_table = base_table.iloc[indexes,:].reset_index(drop=True)

base_table.shape

base_table_5 = base_table_5.drop(base_table_5.index[base_table_5.set_for == '6469-07-01 23:30:00'].tolist()).reset_index(drop=True)
base_table_5['set_for'] = pd.to_datetime(base_table_5['set_for'], format="%Y-%m-%d %H:%M:%S")
base_table_5['created_at'] = pd.to_datetime(base_table_5['created_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['taken_at'] = pd.to_datetime(base_table_5['taken_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['ls_created_at'] = pd.to_datetime(base_table_5['created_at'], format="%Y-%m-%d %H:%M:%S")

base_table_5['set_type'] = base_table_5.apply(lambda base_table_5: 'Rehash' if (base_table_5['set_type'] in ['Rehash','Rehash Res']) else base_table_5['set_type'], axis=1)

base_table_5.dropna(subset = ['set_for'], inplace=True)

base_table_5['set_for'] = pd.to_datetime(base_table_5['set_for'], format="%Y-%m-%d %H:%M:%S")
base_table_5['created_at'] = pd.to_datetime(base_table_5['created_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['taken_at'] = pd.to_datetime(base_table_5['taken_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['ls_created_at'] = pd.to_datetime(base_table_5['created_at'], format="%Y-%m-%d %H:%M:%S")

def zone_change(t,z):
    if t.month in range(3,11) and z == 'Eastern Time (US & Canada)':
        return (t+DateOffset(hours=-4))
    elif t.month in range(3,11) and z == 'Central Time (US & Canada)':
        return (t+DateOffset(hours=-5))
    elif t.month in range(3,11) and z == 'Mountain Time (US & Canada)':
        return (t+DateOffset(hours=-6))
    elif t.month in (1,2,11,12) and z == 'Eastern Time (US & Canada)':
        return (t+DateOffset(hours=-5))
    elif t.month in (1,2,11,12) and z == 'Central Time (US & Canada)':
        return (t+DateOffset(hours=-6))
    elif t.month in (1,2,11,12) and z == 'Mountain Time (US & Canada)':
        return (t+DateOffset(hours=-7))

base_table_5['set_for'] = base_table_5.apply(lambda base_table_5: zone_change(base_table_5['set_for'],base_table_5['time_zone']),axis=1)

base_table_5.dropna(subset = ['set_for'], inplace=True)

base_table_5['created_at'] = base_table_5.apply(lambda base_table_5: zone_change(base_table_5['created_at'],base_table_5['time_zone']),axis=1)

base_table_5.dropna(subset = ['created_at'], inplace=True)

base_table_5['taken_at'] = base_table_5.apply(lambda base_table_5: zone_change(base_table_5['taken_at'],base_table_5['time_zone']),axis=1)

base_table_5['ls_created_at'] = base_table_5.apply(lambda base_table_5: zone_change(base_table_5['ls_created_at'],base_table_5['time_zone']),axis=1)

base_table_5['set_for'] = pd.to_datetime(base_table_5['set_for'], format="%Y-%m-%d %H:%M:%S")
base_table_5['created_at'] = pd.to_datetime(base_table_5['created_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['taken_at'] = pd.to_datetime(base_table_5['taken_at'], format="%Y-%m-%d %H:%M:%S")
base_table_5['ls_created_at'] = pd.to_datetime(base_table_5['ls_created_at'], format="%Y-%m-%d %H:%M:%S")

base_table_5['set_for_year'] = base_table_5.apply(lambda base_table_5: int(base_table_5['set_for'].year), axis = 1)

base_table_5['taken_at_year'] = base_table_5.apply(lambda base_table_5: int(base_table_5['taken_at'].year), axis = 1)

base_table_5['lead_gap'] = base_table_5.apply(lambda base_table_5: (base_table_5['set_for'] - base_table_5['created_at']).days,axis=1)

base_table_5['month_of_year'] = pd.DatetimeIndex(base_table_5['set_for']).month.fillna(99).astype(int)

base_table_5['day_of_month'] = pd.DatetimeIndex(base_table_5['set_for']).day.fillna(99).astype(int)

base_table_5['hour_of_day'] = pd.DatetimeIndex(base_table_5['set_for']).hour.fillna(99).astype(int)

def quarter_function(month):
    if month in ([1,2,3]):
        return 1
    if month in ([4,5,6]):
        return 2
    if month in ([7,8,9]):
        return 3
    if month in ([10,11,12]):
        return 4

base_table_5['quarter'] = base_table_5.apply(lambda base_table_5: quarter_function(base_table_5['month_of_year']),axis=1)

base_table_5['set_for_week_of_year'] = base_table_5.apply(lambda base_table_5: base_table_5['set_for'].isocalendar()[1],axis=1)

base_table_5['taken_at_week_of_year'] = base_table_5.apply(lambda base_table_5: base_table_5['taken_at'].isocalendar()[1],axis=1)

#### Derived Metrics

#### Experience

setter_exp = base_table_5[['lead_id','setter_id','set_for']].merge(users, left_on = 'setter_id', right_on = 'id', copy = True)

setter_exp['started_on'] = pd.to_datetime(setter_exp['started_on'], format="%Y-%m-%d %H:%M:%S")

setter_exp['experience'] = setter_exp.apply(lambda setter_exp: ((setter_exp['set_for'] - setter_exp['started_on']).days)/30,axis=1)

marketter_exp = base_table_5[['lead_id','marketter_id','taken_at']].merge(users, left_on = 'marketter_id', right_on = 'id', copy = True)

marketter_exp['started_on'] = pd.to_datetime(marketter_exp['started_on'], format="%Y-%m-%d %H:%M:%S")

marketter_exp['experience'] = marketter_exp.apply(lambda marketter_exp: ((marketter_exp['taken_at'] - marketter_exp['started_on']).days)/30,axis=1)

#### Performance = Sum(Issues per week)(For 4 weeks period) / Total leads (For 4 weeks period)

setter_leads = base_table_5[['lead_id','setter_id','target','set_for_year','set_for_week_of_year']]

setter_lds = pd.DataFrame(setter_leads.groupby(['setter_id','set_for_year','set_for_week_of_year']).agg({'lead_id':'count', 'target':'sum'}).reset_index())
setter_lds.columns = ['setter_id','set_for_year','set_for_week_of_year','lead_count','issue_count']

setter_temp_lead_4w = pd.DataFrame(setter_lds.groupby('setter_id')['lead_count'].rolling(4,min_periods=1).sum().reset_index())
setter_temp_issue_4w = pd.DataFrame(setter_lds.groupby('setter_id')['issue_count'].rolling(4,min_periods=1).sum().reset_index())

setter_temp_lead_6m = pd.DataFrame(setter_lds.groupby('setter_id')['lead_count'].rolling(24,min_periods=1).sum().reset_index())
setter_temp_issue_6m = pd.DataFrame(setter_lds.groupby('setter_id')['issue_count'].rolling(24,min_periods=1).sum().reset_index())

setter_temp_lead_4w.columns = ['setter_id','level_1','setter_lead_count_4w']
setter_temp_issue_4w.columns = ['setter_id','level_1','setter_issue_count_4w']
setter_temp_lead_6m.columns = ['setter_id','level_1','setter_lead_count_6m']
setter_temp_issue_6m.columns = ['setter_id','level_1','setter_issue_count_6m']

setter_lds = pd.concat([setter_lds.reset_index(drop=True),setter_temp_lead_4w['setter_lead_count_4w']], axis = 1)
setter_lds = pd.concat([setter_lds.reset_index(drop=True),setter_temp_issue_4w['setter_issue_count_4w']], axis = 1)
setter_lds = pd.concat([setter_lds.reset_index(drop=True),setter_temp_lead_6m['setter_lead_count_6m']], axis = 1)
setter_lds = pd.concat([setter_lds.reset_index(drop=True),setter_temp_issue_6m['setter_issue_count_6m']], axis = 1)

setter_lds['setter_performance_4w'] = setter_lds['setter_issue_count_4w']/setter_lds['setter_lead_count_4w']
setter_lds['setter_performance_6m'] = setter_lds['setter_issue_count_6m']/setter_lds['setter_lead_count_6m']

marketter_leads = base_table_5[['lead_id','marketter_id','target','taken_at_year','taken_at_week_of_year']]

marketter_lds = pd.DataFrame(marketter_leads.groupby(['marketter_id','taken_at_year','taken_at_week_of_year']).agg({'lead_id':'count', 'target':'sum'}).reset_index())
marketter_lds.columns = ['marketter_id','taken_at_year','taken_at_week_of_year','lead_count','issue_count']

marketter_temp_lead_4w = pd.DataFrame(marketter_lds.groupby('marketter_id')['lead_count'].rolling(4,min_periods=1).sum().reset_index())
marketter_temp_issue_4w = pd.DataFrame(marketter_lds.groupby('marketter_id')['issue_count'].rolling(4,min_periods=1).sum().reset_index())
marketter_temp_lead_6m = pd.DataFrame(marketter_lds.groupby('marketter_id')['lead_count'].rolling(24,min_periods=1).sum().reset_index())
marketter_temp_issue_6m = pd.DataFrame(marketter_lds.groupby('marketter_id')['issue_count'].rolling(24,min_periods=1).sum().reset_index())
marketter_temp_lead_4w.columns = ['marketter_id','level_1','marketter_lead_count_4w']
marketter_temp_issue_4w.columns = ['marketter_id','level_1','marketter_issue_count_4w']
marketter_temp_lead_6m.columns = ['marketter_id','level_1','marketter_lead_count_6m']
marketter_temp_issue_6m.columns = ['marketter_id','level_1','marketter_issue_count_6m']

marketter_lds = pd.concat([marketter_lds.reset_index(drop=True),marketter_temp_lead_4w['marketter_lead_count_4w']], axis = 1)
marketter_lds = pd.concat([marketter_lds.reset_index(drop=True),marketter_temp_issue_4w['marketter_issue_count_4w']], axis = 1)
marketter_lds = pd.concat([marketter_lds.reset_index(drop=True),marketter_temp_lead_6m['marketter_lead_count_6m']], axis = 1)
marketter_lds = pd.concat([marketter_lds.reset_index(drop=True),marketter_temp_issue_6m['marketter_issue_count_6m']], axis = 1)

marketter_lds['marketter_performance_4w'] = marketter_lds['marketter_issue_count_4w']/marketter_lds['marketter_lead_count_4w']
marketter_lds['marketter_performance_6m'] = marketter_lds['marketter_issue_count_6m']/marketter_lds['marketter_lead_count_6m']

#### Repeat lead - Repeat Customer

lead_owner = base_table_5[['lead_id','set_for','set_type','project_id','lead_source_id','owner_id','current_state']]

lead_owner['owner_id'] = lead_owner.apply(lambda lead_owner: str(lead_owner['owner_id']),axis = 1)

lead_owner['owner_type_act'] = lead_owner['owner_id']+'-'+lead_owner['set_type']

leads_rep = lead_owner[['lead_id','set_for','set_type','lead_source_id','owner_id']]

indexNames = leads_rep[(leads_rep['set_type'] >= 'Reset') | (leads_rep['set_type'] <= 'Add On') | (leads_rep['set_type'] <= 'Cancel Save') | (leads_rep['set_type'] <= 'Retry') | (leads_rep['set_type'] <= 'FLAG-Credit') | (leads_rep['set_type'] <= 'FLAG-BadPhon') | (leads_rep['set_type'] <= 'Flag-No Eng') | (leads_rep['set_type'] <= 'Flag-Renter') | (leads_rep['set_type'] <= '--- :reset\n') | (leads_rep['set_type'] <= 'FLAG-Dead')].index
leads_rep.drop(indexNames , inplace=True)

leads_rep['set_type_2'] = leads_rep.apply(lambda leads_rep: 'New' if (leads_rep['set_type'] == 'Rehash') else leads_rep['set_type'], axis=1)

leads_rep['owner_type'] = leads_rep['owner_id']+'-'+leads_rep['set_type_2']

leads_rep['owner_type_cnt'] = leads_rep.groupby('owner_type').cumcount() + 1

lead_owner_2 =pd.merge(lead_owner[['lead_id','set_type','owner_id','owner_type_act']], 
             leads_rep[['lead_id','set_type_2','owner_type','owner_type_cnt']], 
             how='left', on=None, left_on='lead_id', right_on='lead_id',
             left_index=False, right_index=False, sort=True,
             suffixes=('', '_lr'), copy=True, indicator=False,
             validate=None)

lead_owner_2['owner_cnt'] = lead_owner_2.groupby('owner_type_act')['owner_type_cnt'].ffill()

lead_owner_2['lead_rep'] = lead_owner_2.apply(lambda lead_owner_2: 1 if (lead_owner_2['owner_cnt'] > 1) else 0, axis=1)

lr = lead_owner_2[['lead_id','lead_rep']]

# customer repeat

lead_owner_proj = lead_owner[['lead_id','set_for','project_id','owner_id','current_state']]

lead_owner_proj['lead_proj_time'] =  lead_owner_proj.apply(lambda lead_owner_proj: lead_owner_proj['set_for'] if (lead_owner_proj['current_state'] == 'closed') else (datetime.now()+DateOffset(years=-20)), axis=1)

lead_owner_proj['lead_proj_owner'] =  lead_owner_proj.apply(lambda lead_owner_proj: lead_owner_proj.owner_id if (lead_owner_proj['current_state'] == 'closed') else 0, axis=1)

lead_owner_proj['cust_rep'] =  lead_owner_proj.apply(lambda lead_owner_proj: 1 if (lead_owner_proj['lead_proj_time'] > lead_owner_proj['set_for'] and lead_owner_proj['lead_proj_owner'] == lead_owner_proj['owner_id']) else 0, axis=1)

cp = lead_owner_proj[['lead_id','cust_rep']]

#### Metrics per Zip/Territory

zip_terr_leads = base_table[['lead_id','owner_zip_id','territory_id','target','set_for_year','set_for_week_of_year']]

zip_lds = pd.DataFrame(zip_terr_leads.groupby(['owner_zip_id','set_for_year','set_for_week_of_year']).agg({'lead_id':'count', 'target':'sum'}).reset_index())
zip_lds.columns = ['owner_zip_id','set_for_year','set_for_week_of_year','lead_count','issue_count']
terr_lds = pd.DataFrame(zip_terr_leads.groupby(['territory_id','set_for_year','set_for_week_of_year']).agg({'lead_id':'count', 'target':'sum'}).reset_index())
terr_lds.columns = ['territory_id','set_for_year','set_for_week_of_year','lead_count','issue_count']

zip_lead_4w = pd.DataFrame(zip_lds.groupby('owner_zip_id')['lead_count'].rolling(4,min_periods=1).sum().reset_index())
zip_issue_4w = pd.DataFrame(zip_lds.groupby('owner_zip_id')['issue_count'].rolling(4,min_periods=1).sum().reset_index())
zip_lead_6m = pd.DataFrame(zip_lds.groupby('owner_zip_id')['lead_count'].rolling(24,min_periods=1).sum().reset_index())
zip_issue_6m = pd.DataFrame(zip_lds.groupby('owner_zip_id')['issue_count'].rolling(24,min_periods=1).sum().reset_index())
zip_lead_4w.columns = ['owner_zip_id','level_1','leads_per_zip_4w']
zip_issue_4w.columns = ['owner_zip_id','level_1','issues_per_zip_4w']
zip_lead_6m.columns = ['owner_zip_id','level_1','leads_per_zip_6m']
zip_issue_6m.columns = ['owner_zip_id','level_1','issues_per_zip_6m']

terr_lead_4w = pd.DataFrame(terr_lds.groupby('territory_id')['lead_count'].rolling(4,min_periods=1).sum().reset_index())
terr_issue_4w = pd.DataFrame(terr_lds.groupby('territory_id')['issue_count'].rolling(4,min_periods=1).sum().reset_index())
terr_lead_6m = pd.DataFrame(terr_lds.groupby('territory_id')['lead_count'].rolling(24,min_periods=1).sum().reset_index())
terr_issue_6m = pd.DataFrame(terr_lds.groupby('territory_id')['issue_count'].rolling(24,min_periods=1).sum().reset_index())
terr_lead_4w.columns = ['territory_id','level_1','leads_per_territory_4w']
terr_issue_4w.columns = ['territory_id','level_1','issues_per_territory_4w']
terr_lead_6m.columns = ['territory_id','level_1','leads_per_territory_6m']
terr_issue_6m.columns = ['territory_id','level_1','issues_per_territory_6m']

zip_lds = pd.concat([zip_lds.reset_index(drop=True),zip_lead_4w['leads_per_zip_4w']], axis = 1)
zip_lds = pd.concat([zip_lds.reset_index(drop=True),zip_issue_4w['issues_per_zip_4w']], axis = 1)
zip_lds = pd.concat([zip_lds.reset_index(drop=True),zip_lead_6m['leads_per_zip_6m']], axis = 1)
zip_lds = pd.concat([zip_lds.reset_index(drop=True),zip_issue_6m['issues_per_zip_6m']], axis = 1)
terr_lds = pd.concat([terr_lds.reset_index(drop=True),terr_lead_4w['leads_per_territory_4w']], axis = 1)
terr_lds = pd.concat([terr_lds.reset_index(drop=True),terr_issue_4w['issues_per_territory_4w']], axis = 1)
terr_lds = pd.concat([terr_lds.reset_index(drop=True),terr_lead_6m['leads_per_territory_6m']], axis = 1)
terr_lds = pd.concat([terr_lds.reset_index(drop=True),terr_issue_6m['issues_per_territory_6m']], axis = 1)

zip_lds['issue_rate_per_zip_4w'] = zip_lds['issues_per_zip_4w']/zip_lds['leads_per_zip_4w']
terr_lds['issue_rate_per_territory_4w'] = terr_lds['issues_per_territory_4w']/terr_lds['leads_per_territory_4w']
zip_lds['issue_rate_per_zip_6m'] = zip_lds['issues_per_zip_6m']/zip_lds['leads_per_zip_6m']
terr_lds['issue_rate_per_territory_6m'] = terr_lds['issues_per_territory_6m']/terr_lds['leads_per_territory_6m']

year_leads = pd.DataFrame(base_table.groupby('set_for_year')['lead_id'].count().reset_index())
year_issues = pd.DataFrame(base_table.groupby('set_for_year')['target'].sum().reset_index())
year_quarter_leads = pd.DataFrame(base_table.groupby(['set_for_year','quarter'])['lead_id'].count().reset_index())
year_quarter_issues = pd.DataFrame(base_table.groupby(['set_for_year','quarter'])['target'].sum().reset_index())

year_leads.columns = ['set_for_year','year_lead_count']
year_issues.columns = ['set_for_year','year_issue_count']
year_quarter_leads.columns = ['set_for_year','quarter','quarter_lead_count']
year_quarter_issues.columns = ['set_for_year','quarter','quarter_issue_count']

#### Lead Repetition

leads = leads.drop(leads.index[leads.set_for == '6469-07-01 23:30:00'].tolist()).reset_index(drop=True)
leads.drop(leads[leads['set_type'] == 'ProjectVisit'].index, inplace = True)

leads['set_type'] = leads.apply(lambda leads: 'Rehash' if (leads['set_type'] in ['Rehash','Rehash Res']) else leads['set_type'], axis=1)

lead_owner = pd.merge(leads[['id','set_for','set_type','project_id','lead_source_id']],
             lead_sources[['id','owner_id']], 
             how='left', on=None, left_on='lead_source_id', right_on='id',
             left_index=False, right_index=False, sort=True,
             suffixes=('', '_ls'), copy=True, indicator=False,
             validate=None)

leads_rep = copy.deepcopy(lead_owner)

indexNames = leads_rep[(leads_rep['set_type'] == 'Reset') | (leads_rep['set_type'] == 'Add On') | (leads_rep['set_type'] == 'Cancel Save') | (leads_rep['set_type'] == 'Retry') | (leads_rep['set_type'] == 'FLAG-Credit') | (leads_rep['set_type'] == 'FLAG-BadPhon') | (leads_rep['set_type'] == 'Flag-No Eng') | (leads_rep['set_type'] == 'Flag-Renter') | (leads_rep['set_type'] == '--- :reset\n') | (leads_rep['set_type'] == 'FLAG-Dead')].index
leads_rep.drop(indexNames , inplace=True)

leads_rep['set_type_2'] = leads_rep.apply(lambda leads_rep: 'New' if (leads_rep['set_type'] == 'Rehash') else leads_rep['set_type'], axis=1)

owner_first_inst = leads_rep.groupby(['owner_id'],as_index=False)['set_for'].first()

leads_rep = pd.merge(leads_rep,
             owner_first_inst[['owner_id', 'set_for']], 
             how='left', on=None, left_on='owner_id', right_on='owner_id',
             left_index=False, right_index=False, sort=True,
             suffixes=('', '_ls'), copy=True, indicator=False,
             validate=None)

leads_rep['set_for'] = pd.to_datetime(leads_rep['set_for'], format="%Y-%m-%d %H:%M:%S")
leads_rep['set_for_ls'] = pd.to_datetime(leads_rep['set_for_ls'], format="%Y-%m-%d %H:%M:%S")

leads_rep['lead_rep'] = leads_rep.apply(lambda leads_rep: 1 if leads_rep.set_for > leads_rep.set_for_ls else 0, axis=1)

leads_owner_2 = pd.merge(lead_owner,
             leads_rep, 
             how='left', on=None, left_on='id', right_on='id',
             left_index=False, right_index=False, sort=True,
             suffixes=('', '_lr'), copy=True, indicator=False,
             validate=None)

owner_first_final = leads_owner_2.groupby(['owner_id'],as_index=False)['set_for'].first()

leads_owner_3 = pd.merge(leads_owner_2,
             owner_first_final[['owner_id', 'set_for']], 
             how='left', on=None, left_on='owner_id', right_on='owner_id',
             left_index=False, right_index=False, sort=True,
             suffixes=('', '_lrf'), copy=True, indicator=False,
             validate=None)

leads_owner_3['lead_rep'] = leads_owner_3.apply(lambda leads_owner_3: str(leads_owner_3['lead_rep']),axis = 1)

def lead_rep_treat(x,t,t_f):
    if x=='nan':
        if t>t_f:
            return '1.0'
        else:
            return '0.0'
    else:
        return x

leads_owner_3['set_for'] = pd.to_datetime(leads_owner_3['set_for'], format="%Y-%m-%d %H:%M:%S")
leads_owner_3['set_for_lrf'] = pd.to_datetime(leads_owner_3['set_for_lrf'], format="%Y-%m-%d %H:%M:%S")

leads_owner_3['lead_rep'] = leads_owner_3.apply(lambda leads_owner_3: lead_rep_treat(leads_owner_3['lead_rep'],leads_owner_3['set_for'],leads_owner_3['set_for_lrf']),axis=1)

leads_owner_3['lead_rep'] = leads_owner_3.apply(lambda leads_owner_3: float(leads_owner_3['lead_rep']),axis = 1)

lead_id_rep = leads_owner_3[['id','lead_rep']]

####Discount

price_discount = base_table[['lead_id','owner_zip_id','set_for','price','ceiling_price','set_for_year','set_for_week_of_year']]

estimates = pd.read_csv('estimates.csv',usecols=['lead_id','price','created_at','project_id','ceiling_price'])
estimates.columns = ['lead_id','price','created_at','project_id','ceiling_price']

est_uni = estimates.groupby(['lead_id'],as_index=False).last()

est_uni.info()

est_uni.isna().sum()

price_discount = pd.merge(price_discount, est_uni[['lead_id','price','ceiling_price']], how='left', on=None, 
            left_on='lead_id',right_on='lead_id',suffixes=('', '_est'), left_index=False, right_index=False, 
                sort=True, copy=True, indicator=False, validate=None)

price_discount.info()

price_discount.isna().sum()

price_discount['price'] = price_discount.price.fillna(0)
price_discount['ceiling_price'] = price_discount.ceiling_price.fillna(0)
price_discount['price_est'] = price_discount.price_est.fillna(0)
price_discount['ceiling_price_est'] = price_discount.ceiling_price_est.fillna(0)

price_discount['price'] = price_discount.apply(lambda price_discount: price_discount['price_est'] if price_discount['price'] == 0 else price_discount['price'],axis=1)
price_discount['ceiling_price'] = price_discount.apply(lambda price_discount: price_discount['ceiling_price_est'] if price_discount['ceiling_price'] == 0 else price_discount['ceiling_price'],axis=1)

def discount(x,y):
    try:
        return (1-x/y)
    except ZeroDivisionError:
        return 0

price_discount['discount'] = price_discount.apply(lambda price_discount: discount(price_discount['price'],price_discount['ceiling_price']),axis=1)

zip_discount = price_discount[['owner_zip_id','set_for_year', 'set_for_week_of_year', 'discount']]

zip_disc = pd.DataFrame(zip_discount.groupby(['owner_zip_id','set_for_year','set_for_week_of_year']).agg({'discount':'mean'}).reset_index())

zip_disc_4w = pd.DataFrame(zip_disc.groupby('owner_zip_id')['discount'].rolling(4,min_periods=1).mean().reset_index())
zip_disc_6m = pd.DataFrame(zip_disc.groupby('owner_zip_id')['discount'].rolling(24,min_periods=1).mean().reset_index())
zip_disc_1y = pd.DataFrame(zip_disc.groupby('owner_zip_id')['discount'].rolling(48,min_periods=1).mean().reset_index())

zip_disc_4w.columns = ['owner_zip_id', 'level_1', 'discount_4w']
zip_disc_6m.columns = ['owner_zip_id', 'level_1', 'discount_6m']
zip_disc_1y.columns = ['owner_zip_id', 'level_1', 'discount_1y']

zip_disc = pd.concat([zip_disc.reset_index(drop=True),zip_disc_4w['discount_4w']], axis = 1)
zip_disc = pd.concat([zip_disc.reset_index(drop=True),zip_disc_6m['discount_6m']], axis = 1)
zip_disc = pd.concat([zip_disc.reset_index(drop=True),zip_disc_1y['discount_1y']], axis = 1)



#### Merging metrics to base

# Experience
base_table = pd.merge(base_table_5, setter_exp[['lead_id','experience']], how='left', on=None, left_on='lead_id', right_on='lead_id',
         suffixes=('', '_setter_exp'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, marketter_exp[['lead_id','experience']], how='left', on=None, left_on='lead_id', right_on='lead_id',
         suffixes=('', '_marketter_exp'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

# Performance
base_table = pd.merge(base_table, setter_lds, how='left', on=None, left_on=['setter_id','set_for_year','set_for_week_of_year'], right_on=['setter_id','set_for_year','set_for_week_of_year'],
         suffixes=('', '_setter'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, marketter_lds, how='left', on=None, left_on=['marketter_id','taken_at_year','taken_at_week_of_year'], right_on=['marketter_id','taken_at_year','taken_at_week_of_year'],
         suffixes=('', '_marketter'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

# Repeat
base_table = pd.merge(base_table, lr, how='left', on=None, left_on='lead_id', right_on='lead_id',
         suffixes=('', '_lrep'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, cp, how='left', on=None, left_on='lead_id', right_on='lead_id',
         suffixes=('', '_crep'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

# Zip-Terr
base_table = pd.merge(base_table, zip_lds, how='left', on=None, left_on=['owner_zip_id','set_for_year','set_for_week_of_year'], right_on=['owner_zip_id','set_for_year','set_for_week_of_year'],
         suffixes=('', '_zip'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, terr_lds, how='left', on=None, left_on=['territory_id','set_for_year','set_for_week_of_year'], right_on=['territory_id','set_for_year','set_for_week_of_year'],
         suffixes=('', '_territory'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

# Year/Quarter metrics
base_table = pd.merge(base_table, year_leads, how='left', on=None, left_on='set_for_year', right_on='set_for_year',
         suffixes=('', '_year'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, year_issues, how='left', on=None, left_on='set_for_year', right_on='set_for_year',
         suffixes=('', '_year'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, year_quarter_leads, how='left', on=None, left_on=['set_for_year','quarter'], right_on=['set_for_year','quarter'],
         suffixes=('', '_quarter'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = pd.merge(base_table, year_quarter_issues, how='left', on=None, left_on=['set_for_year','quarter'], right_on=['set_for_year','quarter'],
         suffixes=('', '_quarter'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

# Discount metrics
base_table = pd.merge(base_table, zip_disc, how='left', on=None, left_on=['owner_zip_id','set_for_year','set_for_week_of_year'], right_on=['owner_zip_id','set_for_year','set_for_week_of_year'],
         suffixes=('', '_disc'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

base_table = base_table.loc[base_table['set_for_year']>2014]

base_table = base_table.loc[base_table['set_for_year']<2020]

base_table = pd.read_csv('base_table.csv')

base_table = base_table[['lead_id','set_for','owner_zip_id','set_type','setter_id','confirmer_id','marketter_id','taken_at','ls_status','src_desc','src_catg_desc','src_grp_desc','home_city','home_state','home_zip_code','house_style','year_home_built','current_state','price','ceiling_price','zip_code','zip_state','territory_name','time_zone','product','target','set_for_year','lead_gap','month_of_year','day_of_month','hour_of_day','quarter','set_for_week_of_year','experience','lead_count','issue_count','setter_lead_count_4w','setter_issue_count_4w','setter_lead_count_6m','setter_issue_count_6m','setter_performance_4w','setter_performance_6m','lead_rep','cust_rep','lead_count_zip','issue_count_zip','leads_per_zip_4w','issues_per_zip_4w','leads_per_zip_6m','issues_per_zip_6m','issue_rate_per_zip_4w','issue_rate_per_zip_6m','lead_count_territory','issue_count_territory','leads_per_territory_4w','issues_per_territory_4w','leads_per_territory_6m','issues_per_territory_6m','issue_rate_per_territory_4w','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','taken_at_year','taken_at_week_of_year','experience','lead_count','issue_count','marketter_lead_count_4w','marketter_issue_count_4w','marketter_lead_count_6m','marketter_issue_count_6m','marketter_performance_4w','marketter_performance_6m']]

base_table.columns = ['lead_id','set_for','owner_zip_id','set_type','setter_id','confirmer_id','marketter_id','taken_at','ls_status','src_desc','src_catg_desc','src_grp_desc','home_city','home_state','home_zip_code','house_style','year_home_built','current_state','price','ceiling_price','zip_code','zip_state','territory_name','time_zone','product','target','set_for_year','lead_gap','month_of_year','day_of_month','hour_of_day','quarter','set_for_week_of_year','setter_experience','setter_lead_count_1w','setter_issue_count_1w','setter_lead_count_4w','setter_issue_count_4w','setter_lead_count_6m','setter_issue_count_6m','setter_performance_4w','setter_performance_6m','lead_rep','cust_rep','leads_per_zip_1w','issues_per_zip_1w','leads_per_zip_4w','issues_per_zip_4w','leads_per_zip_6m','issues_per_zip_6m','issue_rate_per_zip_4w','issue_rate_per_zip_6m','lead_count_per_territory_1w','issue_count_per_territory_1w','leads_per_territory_4w','issues_per_territory_4w','leads_per_territory_6m','issues_per_territory_6m','issue_rate_per_territory_4w','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','taken_at_year','taken_at_week_of_year','marketter_experience','marketter_lead_count_1w','marketter_issue_count_1w','marketter_lead_count_4w','marketter_issue_count_4w','marketter_lead_count_6m','marketter_issue_count_6m','marketter_performance_4w','marketter_performance_6m']

base_table.to_csv("base_table_new.csv")

base_table = pd.read_csv("base_table_new.csv")

pd.DataFrame(base_table.isna().sum()).to_csv('na.csv')

model_data = base_table[['lead_id','set_type','ls_status','src_catg_desc','src_grp_desc','home_city','home_state','home_zip_code','territory_name','product','target','set_for_year','lead_gap','month_of_year','day_of_month','hour_of_day','quarter','set_for_week_of_year','setter_experience','setter_lead_count_4w','setter_issue_count_4w','setter_lead_count_6m','setter_issue_count_6m','setter_performance_4w','setter_performance_6m','leads_per_zip_4w','issues_per_zip_4w','leads_per_zip_6m','issues_per_zip_6m','issue_rate_per_zip_4w','issue_rate_per_zip_6m','leads_per_territory_4w','issues_per_territory_4w','leads_per_territory_6m','issues_per_territory_6m','issue_rate_per_territory_4w','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','taken_at_year','taken_at_week_of_year','marketter_experience','marketter_lead_count_1w','marketter_issue_count_1w','marketter_lead_count_4w','marketter_issue_count_4w','marketter_lead_count_6m','marketter_issue_count_6m','marketter_performance_4w','marketter_performance_6m']]

model_data['product'] = model_data['product'].fillna('OTHERS')
model_data.dropna(how='any',inplace=True)

model_data = pd.read_csv('model_data_new.csv')

model_data.drop(columns = ['Unnamed: 0'],inplace=True)

model_data = model_data.merge(lead_id_rep, how='left', on=None, left_on='lead_id', right_on='id',
         suffixes=('', 'lrep'), left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

model_data.drop(columns = ['id'],inplace=True)

X_new = ['ls_status','src_catg_desc','src_grp_desc','home_state','territory_name','product','set_for_year','lead_gap','month_of_year','hour_of_day','quarter','setter_experience','setter_performance_6m','issue_rate_per_zip_6m','issue_rate_per_territory_6m','year_lead_count','year_issue_count','quarter_lead_count','quarter_issue_count','taken_at_year','marketter_experience','marketter_performance_6m']
test = model_data[X_new]

model_data.groupby('territory_name')['lead_id'].count()

new_terr = ['Austin','Charlotte','Dallas','Denver','Nashville','Tampa']
model_data_stable = model_data.loc[~model_data.territory_name.isin(new_terr)]

model_data_stable.info()