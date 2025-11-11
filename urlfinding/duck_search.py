import pandas as pd
import json
import time
import re
import random
import requests
from urllib import parse
from urlfinding.search_engine import SearchEngine
from urlfinding.common import UrlFinderConfig
from typing import Tuple

@SearchEngine.register("duckduckgo")
class DuckSearch(SearchEngine):

    def __init__(self, settings: UrlFinderConfig):
        super().__init__(settings)
        self.domain = 'https://duckduckgo.com'
        self.language = settings.ddg_language or 'en-us'
        self.headers = {
            'User-Agent': settings.ddg_user_agent or '',
            'Cache-Control': 'no-cache'
        }

    def _load_search_item(self, search_item):
        self.term = search_item.get('term')
        self._blacklist = search_item.get('blacklist', [])

    def _get_response(self):
        query = parse.quote(self.term)
        url = f'{self.domain}/local.js?q={query}&cb=DDG.duckbar.add_local&tg=maps_places&l={self.language}&sf=low'
        req =requests.get(url, headers=self.headers)
        m = re.search(r'DDG.duckbar.add_local\((.*)\)', req.text)
        if m:
            vqd = list(json.loads(m.group(1))['vqd'].values())[0]
            if vqd:
                url = f'{self.domain}/d.js?q={query}&t=D&l=wt-wt&s=0&a=hk&ss_mkt=us&vqd={vqd}&p_ent=&ex=-1&sp=0'
                time.sleep(random.uniform(1,3))
                return requests.get(url, headers=self.headers).text
            else:
                return ''
        else:
            return ''

    def parse_response(self, response) -> Tuple[pd.DataFrame, str]:
        message = ''
        match = re.search(r"DDG\.pageLayout\.load\('d',(\[.*\])\)", response)
        if match:
            result = pd.DataFrame(json.loads(match.group(1)))[['u', 't', 'a']]
            if 'l' in result.columns:
                res0 = pd.DataFrame()
                for r in result.loc[result[['l']].any(axis=1), 'l']:
                    resnew = pd.DataFrame([{'u': x['targetUrl'], 't': x['text'], 'a': x['snippet']} for x in r])
                    res0 = pd.concat([res0, resnew])
                result = pd.concat([res0, result]).reset_index(drop=True)
            result.columns = ['url_se', 'title', 'snippet']
            result['seqno'] = result.index + 1
            result['date'] = time.strftime('%Y%m%d')
            result['query'] = self.term
            result['pagemap'] = '' # return this column because of compatibility with googlsearch.py
            return result[pd.notnull(result.url_se)][self.output_columns], message
        else:
            message = 'Expected javascript file (d.js) not found.'
            return pd.DataFrame(columns=self.output_columns), message
    
    def _process_query(self) -> Tuple[pd.DataFrame, str]:
        exclude = self.excluded_sites()
        if exclude:
            self.term = self.term.replace(' ', '+') + ' ' + exclude
        response = self._get_response()
        # Not sure why we sleep here, assuming rate-limiting... legacy code
        time.sleep(6)
        return self.parse_response(response)