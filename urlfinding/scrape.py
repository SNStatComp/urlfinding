#!/usr/bin/env python
# coding: utf-8

from yaml import load, FullLoader
import pandas as pd
from collections import OrderedDict
import re
import os
#import googlesearch
from urllib.parse import urlparse
import requests
import fire
import time
from googlesearch import GoogleSearch

cwd = os.getcwd()
STREETNAME = 'Streetname'
COMPANIES = f'{cwd}/data/companies.csv'
MAPPINGS = f'{cwd}/config/mappings.yml'

def getConfig(file):
    with open(file,'r') as f:
        config = load(f, Loader=FullLoader)
    mapping = {val:key for key, val in config['columns'].items() if (val != None) and (type(val) != list)}
    computed = {key:val for key, val in config['columns'].items() if type(val) == list}
    search = config['search']
    features = config['features']
    return mapping, computed, search, features

def getFeatures(file, mapping, computed, features):
    feat = pd.read_csv(file, sep=';', dtype=str).fillna('')
    feat.rename(columns=mapping, inplace=True)
    addFeatures(feat, mapping, computed)
    feat = feat[['Id'] + features]
    return feat

def addFeatures(df, mapping, computed):
    for key, val in computed.items():
        df[key] = ''
        for x in val:
            if mapping.get(x):
                if mapping[x] == STREETNAME:
                    df[key] += df[mapping[x]].apply(lambda x: editStreet(x))
                else:
                    df[key] += df[mapping[x]]
            else:
                df[key] += str(x)

def createTerm(rec, features, search):
    res = ''
    for x in search:
        if x in features:
            if x == STREETNAME:
                res += str(editStreet(rec[x]))
            else:
                res += str(rec[x])
        else:
            res += str(x)
    return res

def editStreet(name):
    if name:
        location = re.sub(r'STRWG$', 'STRAATWEG', name)
        location = re.sub(r'WG$', 'WEG', location)
        location = re.sub(r'STR$', 'STRAAT', location)
        location = re.sub(r'^PLN|PLN$', 'PLEIN', location)
        location = re.sub(r'^LN|LN$', 'LAAN', location)
        location = re.sub(r'SNGL$', 'SINGEL', location)
        location = re.sub(r'DK$', 'DIJK', location)
        location = re.sub(r'KD$', 'KADE', location)
    else:
        location = ''
    return location

def search(item, googleSearch):
    result = []
    if item['term']:
        result = googleSearch.search({
            'term': item['term'],
            'orTerm': item['orTerm'],
            'blacklist': item['blacklist'],
            'maxPages': 1
        })
        for i, _ in result.iterrows():
            result.loc[i, 'Id'] = item['Id']
            result.loc[i, 'queryType'] = item['queryType']
    return result
    
def main(fileIn, fileOut, blacklist, log, googleSearch, fstart, skiprows=0, nrows=None, config=MAPPINGS):
    mapping, computed, searchTerms, features = getConfig(config)
    # transform input file to file which can be processed
    if not os.path.isfile(COMPANIES):
        getFeatures(fileIn, mapping, computed, features).to_csv(COMPANIES, sep=';', index=False)
    if skiprows == 0:
        skip = 0
    else:
        skip = range(1, skiprows+1)
    companies = pd.read_csv(COMPANIES, sep=';', skiprows=skip, nrows=nrows)
    for index, company in companies.iterrows():
        print(f'\rAt record: {index}', end='')

        for i, term in enumerate(searchTerms):
            searchTerm = {
                'term': createTerm(company, features, term['term']),
                'orTerm': createTerm(company, features, term.get('orTerm', [])),
                'blacklist': blacklist,
                'Id': company['Id'],
                'queryType': i
            }
            res = search(searchTerm, googleSearch)
            if len(res) > 0:
                if not os.path.isfile(fileOut):
                    res.to_csv(fileOut, sep=';', index=False, float_format='%.0f')
                else:
                    res.to_csv(fileOut, sep=';', mode='a', header=False, index=False, float_format='%.0f')
            else:
                #log company with no results
                if not os.path.isfile(log):
                    company.to_frame().T.to_csv(log, sep=';', index=False, float_format='%.0f')
                else:
                    company.to_frame().T.to_csv(log, sep=';', mode='a', header=False, index=False, float_format='%.0f')
            time.sleep(1)
        skiprows += 1
        fstart.seek(0)
        fstart.write(str(skiprows))

def start(fileIn, googleconfig, blacklist, nrows):
    with open(googleconfig, 'r') as f:
        settings = load(f)
        googleSearch = GoogleSearch(settings)

    with open(blacklist, 'r') as f:
        blacklist = f.read().splitlines()

    today = time.strftime('%Y%m%d')

    if not os.path.isfile(f'{cwd}/maxrownum'):
        with open(f'{cwd}/maxrownum', 'w+') as f:
            f.write('0')

    fstart = open(f'{cwd}/maxrownum', 'r+')
    maxrownum = int(fstart.readline())

    fileOut = f'{cwd}/data/{today}searchResult.csv'
    log = f'{cwd}/data/missed_companies.csv'

    main(fileIn, fileOut, blacklist, log, googleSearch, fstart, maxrownum, nrows)
    fstart.close()
    print(f'Searchresults saved in {fileOut}')
