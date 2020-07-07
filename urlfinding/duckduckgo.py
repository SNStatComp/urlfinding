import pandas as pd
import json
import time
import os
import re
import random
import requests
from urllib import parse

class DuckSearch:

    def __init__(self, settings):
        self.domain = 'https://duckduckgo.com'
        self.language = settings.get('DDGlanguage', 'en-us')
        self.headers = {
            'User-Agent': settings.get('DDGuser-agent', ''),
            'Cache-Control': 'no-cache'
        }

    def search(self, searchItem):
        self.message = ''
        self.term = searchItem.get('term')
        self._blacklist = searchItem.get('blacklist', [])
        return self._processQuery(), self.message

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
                self.response = requests.get(url, headers=self.headers).text
            else:
                self.response = ''
        else:
            self.response = ''
        time.sleep(6)

    def _parse_response(self):
        columns = ['date', 'seqno', 'query', 'title', 'snippet', 'url_se', 'pagemap']
        m = re.search(r"DDG\.pageLayout\.load\('d',(\[.*\])\)", self.response)
        if m:
            result = pd.DataFrame(json.loads(m.group(1)))[['u', 't', 'a']]
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
            return result[pd.notnull(result.url_se)][columns]
        else:
            self.message = 'Expected javascript file (d.js) not found.'
            return pd.DataFrame(columns=columns)
    
    def excludedSites(self):
        exclude = ' '.join(['-site:' + url for url in self._blacklist])
        return exclude

    def _processQuery(self):
        exclude = self.excludedSites()
        if exclude:
            self.term = self.term.replace(' ', '+') + ' ' + exclude
        self._get_response()
        return self._parse_response()