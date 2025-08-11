from googleapiclient.discovery import build
from urllib.parse import urlparse
import pandas as pd
import json
import time
from urlfinding.search_engine import SearchEngine
from urlfinding.common import UrlFinderConfig
from typing import Tuple

@SearchEngine.register("google")
class GoogleSearch(SearchEngine):

    def __init__(self, settings: UrlFinderConfig):
        super().__init__(settings)
        self.KEY_VALUE = settings.key
        self.SEARCH_ENGINE_ID = settings.searchengineid
        self.GEOLOCATION = settings.geolocation or ''
        self.LANGUAGE = settings.language or ''

    def _load_search_item(self, search_item):
        self.term = search_item.get('term')
        self.orTerm = search_item.get('orTerm', '')
        self.MAXPAGES = search_item.get('maxPages', 1)
        self._blacklist = search_item.get('blacklist', [])

    def _process_query(self) -> Tuple[pd.DataFrame, str]:
        message = ''
        pageNum = 1
        numResults = 10
        stop = (pageNum > self.MAXPAGES) or (numResults < 10)
        service = build("customsearch", "v1", developerKey=self.KEY_VALUE)
        exclude = self.excluded_sites()

        rows = []
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
                    rows.append({
                        'date': time.strftime('%Y%m%d'),
                        'seqno': i + 1,
                        'query': self.term,
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url_se': item.get('link', ''),
                        'pagemap': json.dumps(item.get('pagemap'))
                    })            
            pageNum += 1
            numResults = len(items)
            stop = (pageNum > self.MAXPAGES) or (numResults < 10)
        result = pd.DataFrame(rows)
        if len(result) == 0:
            message = 'No results returned'
        return result, message

