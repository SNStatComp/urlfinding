import pandas as pd
import json
import time
import os
import re
import pyderman
from urllib.parse import urlparse
from seleniumwire import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

class DuckSearch:

    def __init__(self, settings):
        self.GEOLOCATION = settings.get('geolocation', '')
        browser = settings.get('browser', 'firefox')
        
        if browser == 'firefox':
            options = FirefoxOptions()
            options.headless = True
            path = pyderman.install(browser=pyderman.firefox)
            self.driver = webdriver.Firefox(executable_path=path, options=options, seleniumwire_options={'max_threads': 3})
        if browser == 'chrome':
            options = ChromeOptions()
            options.headless = True
            path = pyderman.install(browser=pyderman.chrome)
            self.driver = webdriver.Chrome(path, options=options, seleniumwire_options={'max_threads': 3})

        self.last_term = ''
        self.last_response = ''

    def search(self, searchItem):
        self.message = ''
        self.term = searchItem.get('term')
        self._blacklist = searchItem.get('blacklist', [])
        return self._processQuery(), self.message

    def _get_response(self):
#         try:
#             del self.driver.requests
#         except:
#             pass
        self.driver.scopes = ['.*duckduckgo\.com/.*\.js.*']
        self.driver.get(f'https://duckduckgo.com/?q={self.term}&t=hj&ia=web')
#        request = None
        try:
            request = self.driver.wait_for_request('https://duckduckgo.com/d.js', timeout=30)
        except:
            request = None
            for r in self.driver.requests:
                if 'd.js' in r.path:
                    request = r
                    break
        
        if request:
            if request.response.body:
                self.last_response = request.response.body.decode('utf-8', 'ignore')
                self.last_term = self.term
                self.response = request.response.body.decode('utf-8', 'ignore')
            else:
                self.last_term = self.term
                self.last_response = ''
                self.response = ''
                self.message = str(request.response.status_code) + ': ' + request.response.reason
        else:
            self.message = 'No response from query'
            if self.last_term == self.term:
                self.response = self.last_response
            else:
                self.response = ''

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
    
    def quit(self):
        self.driver.quit()
