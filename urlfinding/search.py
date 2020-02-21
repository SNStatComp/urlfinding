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
import time
from urlfinding.googlesearch import GoogleSearch
from urlfinding.common import get_config

cwd = os.getcwd()
STREETNAME = 'Streetname'
COMPANIES = f'{cwd}/data/companies.csv'
MAPPINGS = f'{cwd}/config/mappings.yml'

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

def createTerm(rec, features, search_terms):
    res = ''
    for x in search_terms:
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

def search_item(item, googleSearch):
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
    mapping, computed, searchTerms, features, _ = get_config(config)
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

        for i, term in enumerate(searchTerms['queries']):
            searchTerm = {
                'term': createTerm(company, features, term['term']),
                'orTerm': createTerm(company, features, term.get('orTerm', [])),
                'blacklist': blacklist,
                'Id': company['Id'],
                'queryType': i
            }
            res = search_item(searchTerm, googleSearch)
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

def search(fileIn, googleconfig, blacklist, nrows):
    '''
    This function startes a Google search.

    Parameters:
    - base_file: A .csv file with a list of enterprises for which you want to find the webaddress.
    If you want to use the pretrained ML model provided (data/model.pkl) the file must at least include
    the following columns: id, tradename, legalname, address, postalcode and locality.
    The column names can be specified in a mapping file (see config/mappings.yml for an example)
    - googleconfig: This file contains your credentials for using the Google custom search engine API
    - blacklist: A file containing urls you want to exclude from your search
    - nrows: Number of enterprises you want to search for. Google provides 100 queries per day for free.
    For example if for every enterprise 6 queries are performed, then for 10 enterprises 6 * 10 = 60 queries are performed.
    Every query returns at most 10 search results.

    Returns:
    This function creates a file (<YYYYMMDD>searchResult.csv) in the 'data' folder containing the search results, where YYYYMMDD is the current date.
    '''
    with open(googleconfig, 'r') as f:
        settings = load(f, Loader=FullLoader)
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
    print(f'\nSearchresults saved in {fileOut}')
