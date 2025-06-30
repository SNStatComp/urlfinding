import pandas as pd
import re
from urllib.parse import urlparse
import json
import os
from flatten_json import flatten
import jellyfish
from nltk.tokenize import RegexpTokenizer
from urlfinding.url_finder import UrlFinder

cwd = os.getcwd()
Tokenizer = RegexpTokenizer(r'\w+')
POPULATION = f'{cwd}/data/companies.csv'

def read_datafiles(filelist, popFile, nqueries):
    pop = pd.read_csv(popFile, sep=';', dtype=str).fillna('')
    data = pd.DataFrame()
    for dataFile in filelist:
        data = data.append(pd.read_csv(dataFile, sep=';', dtype=str).fillna(''))
    data = data.reset_index(drop=True)
    data['host'] = data['url_se'].map(getHost)
    urlsToRemove = getFrequentUrls(data, nqueries * 10)
    data = data[~data['host'].isin(urlsToRemove)]

    moreUrlsToRemove = getFrequentUrls2(data, 1)
    data = data[~data['host'].isin(moreUrlsToRemove)].copy()
    data['seqno'] = data.sort_values(by=['Id', 'queryType', 'seqno'], axis=0).groupby(['Id', 'queryType']).cumcount() + 1
    data['seq_score'] = 11 - data['seqno']
    return data.reset_index(drop=True), pop

def isEqualish(s1, s2, threshold):
    return jellyfish.jaro_winkler(s1, s2) > threshold

def commonWordsRow(s1, s2):
    result = []
    w1 = [w for w in Tokenizer.tokenize(s1) if len(w) > 2]
    try:
        w2 = [w for w in Tokenizer.tokenize(s2) if len(w) > 2]
    except:
        w2 = []
    for w in w2:
        result += [jellyfish.jaro_winkler(v, w) for v in w1]
    maxJW = 0
    if len(result) > 0:
        maxJW = max(result)
    return maxJW

def commonWords(s):
    res = s.apply(lambda row: commonWordsRow(row[0], row[1]), axis=1)
    return res

def inBlacklist(host, blacklist):
    host = '.'.join(host.split('.')[-2:-1])
    return host in blacklist

def FeaturesFromVariable(data, var, cols_se):
    for col in cols_se:
        if col == 'pagemap':
            if not var in ['TradeName', 'LegalName']:
                data['eqPagemap' + var] = data['pagemap' + var].apply(lambda x: float(len(x) > 0) if x else 0.0)
        else:
            data['eq' + col.title() + var] = commonWords(data[[col, var]])
    return data

def Features(data, vars, cols_se):
    for var in vars:
        if var != 'Id':
            data = FeaturesFromVariable(data, var, cols_se)
    return data

def FeaturesFromPagemap(pagemap):
    address, postalcode, locality, telephone, urls = [], [], [], [], []
    if pagemap != '':
        pm = json.loads(pagemap)
        pagemapFlat = flatten(pm)
        for col in pagemapFlat:
            value = pagemapFlat[col]
            if 'streetaddress' in col:
                address.append(value.lower())
            elif 'postalcode' in col:
                postalcode.append(pagemapFlat[col].lower().replace(' ', ''))
            elif 'locality' in col:
                locality.append(pagemapFlat[col].lower())
            elif 'telephone' in col:
                telephone.append(re.sub(r'\D','',pagemapFlat[col]))
            elif 'url' in col:
                host = urlparse(pagemapFlat[col]).hostname
                if host:
                    urls.append(host.replace('www.', ''))
        address = list(set(address) - set(postalcode))
        address = list(set(address) - set(locality))
    return {
        'pagemapAddress': ', '.join(address),
        'pagemapPostalcode':  ', '.join(postalcode),
        'pagemapLocality': ', '.join(locality),
        'pagemapTelephone': ', '.join(telephone),
        'pagemapUrl': urls,
    }

def getFrequentUrls(df, threshold):
    counts = df['host'].value_counts()
    frac = counts > threshold
    return frac[frac].index.tolist()

def getFrequentUrls2(df, threshold):
    g = df.groupby('host')['Id'].nunique()
    return g[g > threshold].index

def addEqUrl(host, url, url_redirect):
    if url=='':
        res = False
    else:
        host = '.'.join(host.split('.')[-2:-1])
        url = '.'.join(url.split('.')[-2:-1])
        if url_redirect:
            url_redirect = '.'.join(url_redirect.split('.')[-2:-1])
            res = (host == url) or (host == url_redirect)
        else:
            res = (host == url)
            
    return res

def getHost(x):
    hostname = urlparse(x).hostname
    if hostname:
        res = '.'.join(hostname.replace('www.', '').split('.')[-2:])
    else:
        res = ''
    return res

def aggregate_urls(data):
    cols = [x for x in data.columns if (re.match("^eq.+", x)) and (x != 'eqUrl')]
    aggs = {x: ['min', 'mean', 'max'] for x in cols}
    aggs['zscore'] = 'max'
    aggs['percentage'] = 'max'
    aggs['seq_score_total'] = 'max'
    aggs['seq_score'] = 'sum'
    if 'eqUrl' in data.columns:
        aggs['eqUrl'] = 'max'
        sel = ['Id', 'host', 'zscore', 'percentage', 'seq_score', 'seq_score_total', 'eqUrl'] + cols
    else:
        sel = ['Id', 'host', 'zscore', 'percentage', 'seq_score', 'seq_score_total'] + cols
    res = data[sel].groupby(['Id', 'host']).agg(aggs).reset_index()
    res.columns = ['_'.join(tup).rstrip('_') for tup in res.columns.values]
    res['seq_score_perc'] = res['seq_score_sum']/res['seq_score_total_max']
    return res

def extract(date, dataFiles, blacklistFile, config=UrlFinder.MAPPINGS):
    '''
    This function creates a feature file to be used for training your Machine Learning model or predicting using your an already trained model.

    Parameters:
    - date: Used for adding a 'timestamp' to the name of the created feature file
    - data_files: list of files containing the search results
    - blacklist: see above

    Returns:
    This function creates the feature file <YYYYMMDD>features_agg.csv in the data folder
    '''
    mappings = UrlFinder.get_mappings_config(config)
    search = mappings.search
    allVars = mappings.features

    allVars += ['Id']
    
    with open(blacklistFile) as f:
        blacklist = f.read().splitlines()
    blacklist = ['.'.join(x.split('.')[:-1]) for x in blacklist]
    result, sample = read_datafiles(dataFiles, POPULATION, len(search['queries']))
    result = result.merge(sample, on='Id')
    usedVars = list(set(allVars).intersection(result.columns))
    result[usedVars] = result[usedVars].applymap(str.lower)
    if 'pagemap' in search['columns']:
        resultPagemap = pd.DataFrame([FeaturesFromPagemap(row) for row in result['pagemap']])
        result = pd.concat([result, resultPagemap], axis=1)
    result = Features(result, usedVars, search['columns'])
    result = result[result['host'].map(lambda x: not inBlacklist(x, blacklist))]

    # remove columns not needed for training/predicting
    colsToRemove = sample.columns.tolist()
    colsToRemove.remove('Id')
    colsToRemove += search['columns']

    # if input file is used for training:
    if 'Url' in sample.columns:
        colsToRemove = [e for e in colsToRemove if e not in ['Url', 'Url_redirect']]
    result.drop(columns=colsToRemove, inplace=True)
    
    # add column with distribution of hosts for a company
    count_url = result[['Id','host']].groupby(['Id','host']).agg({'Id':'count'})
    perc_url = count_url.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))    
    perc_url.columns = ['percentage']
    perc_url.reset_index(inplace=True)
    result = result.merge(perc_url, on=['Id', 'host'])
    
    # add zscore
    zscore = lambda x: (x - x.mean()) / x.std()
    eee = result.groupby(['Id', 'host'])['Id'].count().transform(zscore).to_frame()
    eee.columns = ['zscore']
    eee.reset_index(inplace=True)
    result = result.merge(eee, on=['Id', 'host'])
    
    # transform columns
    result['seqno'] = result['seqno'].apply(float) / 10
    
    seq_score_total = result[['Id', 'seq_score']].groupby('Id').agg({'seq_score': 'sum'}).reset_index()
    seq_score_total.columns = ['Id', 'seq_score_total']
    result = result.merge(seq_score_total, on='Id')

    # if input file is used for training:
    if 'Url' in sample.columns:
        if 'Url_redirect' in sample.columns:
            result['eqUrl'] = [addEqUrl(x, y, z) for x, y, z in zip(result['host'],result['Url'], result['Url_redirect'])]
        else:
            result['eqUrl'] = [addEqUrl(x, y, '') for x, y in zip(result['host'],result['Url'])]

    #result.to_csv(f'{cwd}/data/{date}features_searchresult.csv', sep=';', index=False)
    aggs = aggregate_urls(result)
    if 'Url' in sample.columns:
        aggs.rename(columns={'zscore_max': 'zscore', 'eqUrl_max': 'eqUrl', 'percentage_max': 'percentage'}, inplace=True)
    else:
        aggs.rename(columns={'zscore_max': 'zscore', 'percentage_max': 'percentage'}, inplace=True)        
    aggs.to_csv(f'{cwd}/data/{date}features.csv', sep=';', index=False)
    print(f'Created feature file {cwd}/data/{date}features.csv')
