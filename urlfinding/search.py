#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import os
import time
from urlfinding.search_engine import SearchEngine

from typing import List, Dict

from urlfinding.common import UrlFindingDefaults, MappingsConfig
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

class Search:
    STREETNAME = 'Streetname'

    def __init__(self, 
                 search_engine: SearchEngine,                 
                 mappings_config: MappingsConfig, 
                 population_path: str | Path = None, 
                 url_blacklist_path: List[str] | None = None,
                 working_directory: str | Path = None, 
                 output_path: str | Path = None,
                 log_path: str | Path = None):
        
        self.search_engine = search_engine
        self.mappings_config = mappings_config

        self.population_path = Path(population_path or UrlFindingDefaults.POPULATION)
        self.url_blacklist = url_blacklist_path or []

        self.working_directory = Path(working_directory or UrlFindingDefaults.CWD)

        if output_path:
            self.output_path = output_path
        else:
            today = time.strftime('%Y%m%d')
            self.output_path = Path(output_path or self.working_directory / f'data/{today}searchResult.csv')
        
        self.log_path = Path(log_path or self.working_directory / 'data/missed_companies.csv')

        self.maxrownum_path = self.working_directory / 'maxrownum'

    @classmethod
    def from_paths(cls, 
                 search_engine_config_path: str | Path,                 
                 mappings_path: str | Path = None, 
                 population_path:str | Path = None, 
                 url_blacklist_path: str | Path = None,
                 working_directory: str | Path = None, 
                 output_path: str | Path = None,
                 log_path: str | Path = None):
        search_engine = SearchEngine.from_config(search_engine_config_path)
        mappings_config = UrlFindingDefaults.get_mappings_config(mappings_path or UrlFindingDefaults.MAPPINGS)     
        url_blacklist = UrlFindingDefaults.read_linesplit_list_csv(url_blacklist_path)
        return cls(search_engine, mappings_config, population_path,
                 url_blacklist, working_directory, output_path, log_path)

    def get_features(self, file:str, mapping: Dict[str, str], computed: Dict[str, List[str]], features: List[str]):
        feat = pd.read_csv(file, sep=';', dtype=str).fillna('')
        feat.rename(columns=mapping, inplace=True)
        
        for key, val in computed.items():
            feat[key] = ''
            for x in val:
                if mapping.get(x):
                    if mapping[x] == Search.STREETNAME:
                        feat[key] += feat[mapping[x]].apply(lambda x: self.edit_street(x))
                    else:
                        feat[key] += feat[mapping[x]]
                else:
                    feat[key] += str(x)
            features += [key]
        feat = feat[['Id'] + features]
        return feat

    def create_term(self, record, features: List[str], search_terms) -> str:
        res = ''
        for search_term in search_terms:
            if search_term in features:
                if search_term == Search.STREETNAME:
                    res += str(self.edit_street(record[search_term]))
                else:
                    res += str(record[search_term])
            else:
                res += str(search_term)
        return res

    def edit_street(self, name: str) -> str:
        if name:
            location = re.sub(r'STRWG\b', 'STRAATWEG', name)
            location = re.sub(r'WG\b', 'WEG', location)
            location = re.sub(r'STR\b', 'STRAAT', location)
            location = re.sub(r'^PLN|PLN\b', 'PLEIN', location)
            location = re.sub(r'^LN|LN\b', 'LAAN', location)
            location = re.sub(r'SNGL\b', 'SINGEL', location)
            location = re.sub(r'DK\b', 'DIJK', location)
            location = re.sub(r'KD\b', 'KADE', location)
        else:
            location = ''
        return location

    def search_item(self, item):
        result = []
        message = ''
        if item['term']:
            result, message = self.search_engine.search({
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
        """Helper function to append rows of a DataFrame to an existing csv file

        Args:
            df (pd.DataFrame): Dataframes with rows to be added to the output file
            output_path (str): Path to the output file
            sep (str, optional): Seperator/delimitr of the csv file. Defaults to ';'.
        """
        if not os.path.isfile(output_path):
            df.to_csv(output_path, sep=sep, index=False, float_format='%.0f')
        else:
            df.to_csv(output_path, sep=sep, mode='a', header=False, index=False, float_format='%.0f')

    def create_population(self, input_urls_path: str, override_population:bool=True):
        """
        If you want to use the pretrained ML model provided (data/model.pkl) the file must at least include
        the following columns: id, tradename, legalname, address, postalcode and locality.
        The column names can be specified in a mapping file (see config/mappings.yml for an example)
        Args:
            input_urls_path (str): A .csv file with a list of enterprises for which you want to find the webaddress.
            override_population (bool, optional): If True, always override the population file. Defaults to True.
        """
        # transform input file to file which can be processed
        if override_population or not os.path.isfile(self.population_path):
            self.get_features(input_urls_path, 
                              self.mappings_config.mapping, 
                              self.mappings_config.computed, 
                              self.mappings_config.features).to_csv(self.population_path, sep=';', index=False)
        else:
            logger.info("Population already exists. If you want to override it, set override_population=True")
        
    def _process(self,  
                 blacklist: List[str], 
                 maxrownum_f, 
                 skip_rows:int=0, 
                 nrows:int=None):
        
        features = self.mappings_config.features

        skip = 0 if skip_rows == 0 else range(1, skip_rows + 1)

        try:
            companies = pd.read_csv(self.population_path, sep=';', skiprows=skip, nrows=nrows)
        except FileNotFoundError as e:
            logger.error(f"{self.population_path} is missing. Set the `input_urls_path` parameter to a csv with companies to create this file.")
            raise e

        for index, company in companies.iterrows():
            logger.info(f'\rAt record: {index}', end='')

            for i, term in enumerate(self.mappings_config.search['queries']):
                search_term = {
                    'term': self.create_term(company, features, term['term']),
                    'orTerm': self.create_term(company, features, term.get('orTerm', [])),
                    'blacklist': blacklist,
                    'Id': company['Id'],
                    'queryType': i
                }
                result, message = self.search_item(search_term)
                if len(result) > 0:
                    self.append_to_csv(result, self.output_path)
                else:
                    #log company with no results
                    missed = company.to_frame().T
                    missed['reason'] = message
                    self.append_to_csv(missed, self.log_path)
                time.sleep(1)
            skip_rows += 1
            maxrownum_f.seek(0)
            maxrownum_f.write(str(skip_rows))

    def run(self, nrows: int, input_urls_path: str = None):
        '''
        This function startes a Search.

        Parameters:
        - search_engine_config_path: This file contains your credentials for using the Google custom search engine API
        - blacklist_path: A file containing urls you want to exclude from your search
        - nrows: Number of enterprises you want to search for. Google provides 100 queries per day for free.
        For example if for every enterprise 6 queries are performed, then for 10 enterprises 6 * 10 = 60 queries are performed.
        Every query returns at most 10 search results.

        Returns:
        This function creates a file (<YYYYMMDD>searchResult.csv) in the 'data' folder containing the search results, where YYYYMMDD is the current date.
        '''
        if input_urls_path:            
            self.create_population(input_urls_path, override_population=True)

        if not os.path.isfile(self.maxrownum_path):
            with open(self.maxrownum_path, 'w+') as f:
                f.write('0')        

        with open(self.maxrownum_path, 'r+') as maxrownum_f:
            skip_rows = int(maxrownum_f.readline())
            self._process(self.url_blacklist, maxrownum_f, skip_rows, nrows)
        logger.info(f'\nSearchresults saved in {self.output_path}')
