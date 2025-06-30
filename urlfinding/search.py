#!/usr/bin/env python
# coding: utf-8

import yaml
import pandas as pd
import re
import os
import time
from urlfinding.googlesearch import GoogleSearch
from urlfinding.duckduckgo import DuckSearch
from urlfinding.url_finder import UrlFinder
from typing import List, Dict

class Search:
    STREETNAME = 'Streetname'

    def __init__(self, 
                 population_path:str, 
                 mappings_path: str, 
                 current_working_directory: str = None, 
                 output_path: str = None,
                 log_path: str = None):
        self.population_path = population_path
        self.mappings_path = mappings_path         
        self.cwd = current_working_directory or os.getcwd()
        
        if output_path:
            self.output_path = output_path
        else:
            today = time.strftime('%Y%m%d')
            self.output_path = output_path or f'{self.cwd}/data/{today}searchResult.csv'
        
        self.log_path = log_path or f'{self.cwd}/data/missed_companies.csv'

        self.maxrownum = f'{self.cwd}/maxrownum'

    def get_features(self, file:str, mapping: Dict[str, str], computed: Dict[str, List[str]], features: List[str]):
        feat = pd.read_csv(file, sep=';', dtype=str).fillna('')
        feat.rename(columns=mapping, inplace=True)
        self.add_features(feat, mapping, computed)
        feat = feat[['Id'] + features]
        return feat

    def add_features(self, df:pd.DataFrame, mapping: Dict[str, str], computed: Dict[str, List[str]]):
        for key, val in computed.items():
            df[key] = ''
            for x in val:
                if mapping.get(x):
                    if mapping[x] == Search.STREETNAME:
                        df[key] += df[mapping[x]].apply(lambda x: self.edit_street(x))
                    else:
                        df[key] += df[mapping[x]]
                else:
                    df[key] += str(x)

    def create_term(self, record, features: List[str], search_terms) -> str:
        res = ''
        for x in search_terms:
            if x in features:
                if x == Search.STREETNAME:
                    res += str(self.edit_street(record[x]))
                else:
                    res += str(record[x])
            else:
                res += str(x)
        return res

    def edit_street(self, name: str) -> str:
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

    def search_item(self, item, search_engine):
        result = []
        message = ''
        if item['term']:
            result, message = search_engine.search({
                'term': item['term'],
                'orTerm': item['orTerm'],
                'blacklist': item['blacklist'],
                'maxPages': 1
            })
            for i, _ in result.iterrows():
                result.loc[i, 'Id'] = item['Id']
                result.loc[i, 'queryType'] = item['queryType']
        else:
            message = 'No query specified'
        return result, message

    @staticmethod
    def append_to_csv(df: pd.DataFrame, output_path: str, sep=';'):
        if not os.path.isfile(output_path):
            df.to_csv(output_path, sep=sep, index=False, float_format='%.0f')
        else:
            df.to_csv(output_path, sep=sep, mode='a', header=False, index=False, float_format='%.0f')

    def _process(self, 
                 input_path: str, 
                 blacklist: List[str], 
                 search_engine, 
                 maxrownum_f, 
                 skip_rows:int=0, 
                 nrows:int=None, 
                 config=UrlFinder.MAPPINGS):
        
        mappings_config = UrlFinder.get_mappings_config(config)
        features = mappings_config.features

        # transform input file to file which can be processed
        if not os.path.isfile(self.population_path):
            self.get_features(input_path, mappings_config.mapping, mappings_config.computed, features).to_csv(self.population_path, sep=';', index=False)

        skip = 0 if skip_rows == 0 else range(1, skip_rows + 1)

        companies = pd.read_csv(self.population_path, sep=';', skip_rows=skip, nrows=nrows)

        for index, company in companies.iterrows():
            print(f'\rAt record: {index}', end='')

            for i, term in enumerate(mappings_config.search_terms['queries']):
                search_term = {
                    'term': self.create_term(company, features, term['term']),
                    'orTerm': self.create_term(company, features, term.get('orTerm', [])),
                    'blacklist': blacklist,
                    'Id': company['Id'],
                    'queryType': i
                }
                res, message = self.search_item(search_term, search_engine)
                if len(res) > 0:
                    self.append_to_csv(res, self.output_path)
                else:
                    #log company with no results
                    missed = company.to_frame().T
                    missed['reason'] = message
                    self.append_to_csv(missed, self.log_path)
                time.sleep(1)
            skip_rows += 1
            maxrownum_f.seek(0)
            maxrownum_f.write(str(skip_rows))

    def _load_search_engine_config(self, config_path: str):
        with open(config_path, 'r') as f:
            settings = yaml.safe_load(f)
            engine = settings.get('search_engine', 'google')
            if engine == 'google':
                return GoogleSearch(settings)
            elif engine == 'duckduckgo':
                return DuckSearch(settings)
            else:
                print('You must specify a search_engine. Possible values: "google" or "duckduckgo"')
                raise

    def search(self, input_file: str, config_path: str, blacklist_path: str, nrows: int):
        '''
        This function startes a Google search.

        Parameters:
        - base_file: A .csv file with a list of enterprises for which you want to find the webaddress.
        If you want to use the pretrained ML model provided (data/model.pkl) the file must at least include
        the following columns: id, tradename, legalname, address, postalcode and locality.
        The column names can be specified in a mapping file (see config/mappings.yml for an example)
        - googleconfig: This file contains your credentials for using the Google custom search engine API
        - blacklist_path: A file containing urls you want to exclude from your search
        - nrows: Number of enterprises you want to search for. Google provides 100 queries per day for free.
        For example if for every enterprise 6 queries are performed, then for 10 enterprises 6 * 10 = 60 queries are performed.
        Every query returns at most 10 search results.

        Returns:
        This function creates a file (<YYYYMMDD>searchResult.csv) in the 'data' folder containing the search results, where YYYYMMDD is the current date.
        '''
        search_engine = self._load_search_engine_config(config_path)

        with open(blacklist_path, 'r') as f:
            blacklist = f.read().splitlines()

        if not os.path.isfile(self.maxrownum):
            with open(self.maxrownum, 'w+') as f:
                f.write('0')        
        
        with open(self.maxrownum, 'r+') as maxrownum_f:
            skip_rows = int(maxrownum_f.readline())
            self._process(input_file, blacklist, search_engine, maxrownum_f, skip_rows, nrows)
        print(f'\nSearchresults saved in {self.output_path}')
