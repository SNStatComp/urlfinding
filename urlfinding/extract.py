from pathlib import Path
import pandas as pd
import re
from urllib.parse import urlparse
import json
import os
from flatten_json import flatten
from rapidfuzz.distance import JaroWinkler
from nltk.tokenize import RegexpTokenizer
from typing import List

import logging
logger = logging.getLogger(__name__)

from urlfinding.common import UrlFindingDefaults, MappingsConfig
from urllib.parse import urlparse

class FeatureExtractor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or RegexpTokenizer(r'\w+')

    def common_words_row(self, sentence_1: str, sentence_2: str):
        words_1 = [w for w in self.tokenizer.tokenize(sentence_1) if len(w) > 2]
        try:
            words_2 = [w for w in self.tokenizer.tokenize(sentence_2) if len(w) > 2]
        except:
            words_2 = []
        scores = [JaroWinkler.normalized_similarity(v, w) for w in words_2 for v in words_1]
        return max(scores) if scores else 0

    def common_words_batch(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda row: self.common_words_row(row.iloc[0], row.iloc[1]), axis=1)

    def extract_from_pagemap(self, pagemap: str) -> dict:
        address, postalcode, locality, telephone, urls = [], [], [], [], []
        if pagemap:
            try:
                pm = json.loads(pagemap)
                pagemap_flat = flatten(pm)
                for col, value in pagemap_flat.items():
                    if 'streetaddress' in col:
                        address.append(value.lower())
                    elif 'postalcode' in col:
                        postalcode.append(value.lower().replace(' ', ''))
                    elif 'locality' in col:
                        locality.append(value.lower())
                    elif 'telephone' in col:
                        telephone.append(re.sub(r'\D', '', value))
                    elif 'url' in col:
                        host = urlparse(value).hostname
                        if host:
                            urls.append(host.replace('www.', ''))
                address = list(set(address) - set(postalcode) - set(locality))
            except json.JSONDecodeError:
                logger.info("Skipped parsing pagemap, invalid JSON.")
        return {
            'pagemapAddress': ', '.join(address),
            'pagemapPostalcode': ', '.join(postalcode),
            'pagemapLocality': ', '.join(locality),
            'pagemapTelephone': ', '.join(telephone),
            'pagemapUrl': urls,
        }
    
    def features_from_variable(self, data: pd.DataFrame, variable: str, columns: list) -> pd.DataFrame:
        for column in columns:
            if column == 'pagemap':
                if variable not in ['TradeName', 'LegalName']:
                    data[f'eqPagemap{variable}'] = data[f'pagemap{variable}'].apply(
                        lambda x: float(len(x) > 0) if x else 0.0
                    )
            else:
                data[f'eq{column.title()}{variable}'] = self.common_words_batch(data[[column, variable]])
        return data

    def features(self, data: pd.DataFrame, vars: list, columns: list) -> pd.DataFrame:
        for var in vars:
            if var != 'Id':
                data = self.features_from_variable(data, var, columns)
        return data

    def add_seq_score_total(self, data: pd.DataFrame) -> pd.DataFrame:
        seq_score_total = data[['Id', 'seq_score']].groupby('Id').agg({'seq_score': 'sum'}).reset_index()
        seq_score_total.columns = ['Id', 'seq_score_total']
        return data.merge(seq_score_total, on='Id')

    def add_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        zscore = lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0
        grouped = data.groupby(['Id', 'host'])['Id'].count()
        zscores = grouped.transform(zscore).to_frame(name='zscore').reset_index()
        return data.merge(zscores, on=['Id', 'host'])

    def aggregate_urls(self, data: pd.DataFrame) -> pd.DataFrame:
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
        res['seq_score_perc'] = res['seq_score_sum'] / res['seq_score_total_max']
        return res

class DataCleaner:
        def __init__(self, blacklist=None):
            self.blacklist = blacklist or []
        
        def in_blacklist(self, host: str) -> bool:
            return '.'.join(host.split('.')[-2:-1]) in self.blacklist
        
        @staticmethod
        def get_host(url_str: str) -> str:
            host = urlparse(url_str).hostname
            return '.'.join(host.replace('www.', '').split('.')[-2:]) if host else ''

        @staticmethod
        def get_frequent_urls(df, threshold):
            return df['host'].value_counts()[lambda s: s > threshold].index.tolist()

        @staticmethod
        def get_frequent_urls2(df, threshold):
            return df.groupby('host')['Id'].nunique()[lambda s: s > threshold].index.tolist()
        
        @staticmethod
        def extract_domain(url: str) -> str:
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            return domain.lower().removeprefix("www.")

        @staticmethod
        def add_eq_url(host, url, url_redirect):
            host =  DataCleaner.extract_domain(host)
            url =  DataCleaner.extract_domain(url)
            url_redirect =  DataCleaner.extract_domain(url_redirect)

            if url == '':
                return False
            host = '.'.join(host.split('.')[-2:-1])
            url = '.'.join(url.split('.')[-2:-1])
            if url_redirect:
                url_redirect = '.'.join(url_redirect.split('.')[-2:-1])
                return (host == url) or (host == url_redirect)
            return host == url

class Extract:

    def __init__(self, mappings: MappingsConfig, population_path:str=None, working_directory:str=None,
                 url_blacklist: List[str] | None = None):        
        self.cleaner = DataCleaner()
        self.extractor = FeatureExtractor()

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.working_directory = Path(working_directory or UrlFindingDefaults.CWD)
        self.mappings_config = mappings
        self.population_path = Path(population_path or UrlFindingDefaults.POPULATION)

        self.population = None        

        if url_blacklist:
            self.cleaner.blacklist = ['.'.join(x.split('.')[:-1]) for x in url_blacklist]

    @classmethod
    def from_paths(cls, mappings_path:str=None, population_path:str=None, working_directory:str=None,
                 blacklist_path:str=None):
        mappings_config = UrlFindingDefaults.get_mappings_config(mappings_path or UrlFindingDefaults.MAPPINGS)
        url_blacklist = UrlFindingDefaults.read_linesplit_list_csv(blacklist_path)
        return cls(mappings_config, population_path, working_directory, url_blacklist)       

    def load_population(self, overwrite:bool=False) -> None:
        if self.population is None or overwrite:
            self.population = pd.read_csv(self.population_path, sep=';', dtype=str).fillna('')

    def preprocess(self, files: list, nqueries: int) -> pd.DataFrame:
        data = pd.concat((pd.read_csv(f, sep=';', dtype=str).fillna('') for f in files)).reset_index(drop=True)
        data['host'] = data['url_se'].map(self.cleaner.get_host)

        data = self.filter_frequent_urls(data, nqueries)
        data['seqno'] = data.sort_values(by=['Id', 'queryType', 'seqno']).groupby(['Id', 'queryType']).cumcount() + 1
        data['seq_score'] = 11 - data['seqno']
        return data

    def filter_frequent_urls(self, df, nqueries):
        urls_to_remove = self.cleaner.get_frequent_urls(df, nqueries * 10)
        df = df[~df['host'].isin(urls_to_remove)]
        urls_to_remove2 = self.cleaner.get_frequent_urls2(df, 1)
        return df[~df['host'].isin(urls_to_remove2)].copy()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        config = self.mappings_config.search
        vars = list(set(self.mappings_config.features + ['Id']).intersection(df.columns))
        df[vars] = df[vars].map(str.lower)

        if 'pagemap' in config['columns']:
            pagemap_data = pd.DataFrame([self.extractor.extract_from_pagemap(row) for row in df['pagemap']])
            df = pd.concat([df, pagemap_data], axis=1)

        df = self.extractor.features(df, vars, config['columns'])
        return df

    def run(self, date: str, files: List[str], force_reload_population:bool = False):
        self.load_population(force_reload_population)
        data = self.preprocess(files, len(self.mappings_config.search['queries']))
        data = data.merge(self.population, on='Id')
        data = self.extract_features(data)

        data = data[data['host'].map(lambda x: not self.cleaner.in_blacklist(x))]

        # remove columns not needed for training/predicting
        cols_to_remove = self.population.columns.tolist()
        if 'Url' in self.population.columns:
            cols_to_remove = [e for e in cols_to_remove if e not in ['Url', 'Url_redirect', 'Id']]
        else:
            cols_to_remove.remove('Id')
        cols_to_remove += self.mappings_config.search['columns']
        data.drop(columns=[c for c in cols_to_remove if c in data.columns], inplace=True)

        # add column with distribution of hosts for a company
        count_url = data[['Id', 'host']].groupby(['Id', 'host']).size()
        # Convert to a named DataFrame
        count_url = count_url.to_frame('url_count')
        # Compute percentage per Id group
        perc_url = count_url.groupby(level='Id').transform(lambda x: x * 100 / x.sum()).rename(columns={'url_count': 'percentage'})
        perc_url.reset_index(inplace=True)
        data = data.merge(perc_url, on=['Id', 'host'])

        data = self.extractor.add_zscore(data)
        # Legacy, cant't explain the / 10 part exactly.
        data['seqno'] = data['seqno'].apply(float) / 10
        data = self.extractor.add_seq_score_total(data)

        if 'Url' in self.population.columns:
            if 'Url_redirect' in self.population.columns:
                data['eqUrl'] = [self.cleaner.add_eq_url(x, y, z) for x, y, z in zip(data['host'], data['Url'], data['Url_redirect'])]
            else:
                data['eqUrl'] = [self.cleaner.add_eq_url(x, y, '') for x, y in zip(data['host'], data['Url'])]

        aggs = self.extractor.aggregate_urls(data)
        if 'Url' in self.population.columns:
            aggs.rename(columns={'zscore_max': 'zscore', 'eqUrl_max': 'eqUrl', 'percentage_max': 'percentage'}, inplace=True)
        else:
            aggs.rename(columns={'zscore_max': 'zscore', 'percentage_max': 'percentage'}, inplace=True)

        out_path = Path(self.working_directory) / "data" / f"{date}features.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aggs.to_csv(out_path, sep=';', index=False)
        print(f'Created feature file {out_path}')
    