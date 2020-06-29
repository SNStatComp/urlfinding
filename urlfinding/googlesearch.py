from googleapiclient.discovery import build
from urllib.parse import urlparse
import pandas as pd
import json
import time
import os

class GoogleSearch:

    def __init__(self, settings):
        self.KEY_VALUE = settings.get('key')
        if not self.KEY_VALUE:
            raise Exception('no google api key provided')
        self.SEARCH_ENGINE_ID = settings.get('searchengineid')
        if not self.SEARCH_ENGINE_ID:
            raise Exception('no google search_engine_id provided')
        self.GEOLOCATION = settings.get('geolocation', '')
        self.LANGUAGE = settings.get('language', '')

    def search(self, searchItem):
        self.message = ''
        self.term = searchItem.get('term')
        self.orTerm = searchItem.get('orTerm', '')
        self.MAXPAGES = searchItem.get('maxPages', 1)
        self._blacklist = searchItem.get('blacklist', [])
        return self._processQuery(), self.message

    def excludedSites(self):
        exclude = ' '.join(['-site:' + url for url in self._blacklist])
        return exclude

    def _processQuery(self):
        pageNum = 1
        numResults = 10
        stop = (pageNum > self.MAXPAGES) or (numResults < 10)
        service = build("customsearch", "v1", developerKey=self.KEY_VALUE)
        result = pd.DataFrame(columns=['date', 'seqno', 'query', 'title', 'snippet', 'url_se', 'pagemap'])
        exclude = self.excludedSites()

        while not stop:
            if exclude:
                term = self.term + ' ' + exclude
            else:
                term = self.term
            offset = (pageNum - 1) * 10 + 1
            res = service.cse().list(
                q=term,
                cx=self.SEARCH_ENGINE_ID,
                gl=self.GEOLOCATION,
                start=offset,
                lr=f'lang_{self.LANGUAGE}'
            ).execute()
            items = res.get('items',[])
            for i, item in enumerate(items):
                if not urlparse(item.get('link', '')).hostname in self._blacklist:
                    result = result.append({
                        'date': time.strftime('%Y%m%d'),
                        'seqno': i + 1,
                        'query': self.term,
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url_se': item.get('link', ''),
                        'pagemap': json.dumps(item.get('pagemap'))
                    }, ignore_index=True)
            pageNum += 1
            numResults = len(items)
            stop = (pageNum > self.MAXPAGES) or (numResults < 10)
        if len(result) == 0:
            self.message = 'No results returned'
        return result
    
    def quit(self):
        pass

