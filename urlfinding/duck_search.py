import pandas as pd
import json
import time
import re
import random
from curl_cffi import requests
from urllib import parse
from urlfinding.search_engine import SearchEngine
from urlfinding.common import UrlFinderConfig
from typing import Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse,  parse_qs, unquote


@SearchEngine.register("duckduckgo")
class DuckSearch(SearchEngine):

    def __init__(self, settings: UrlFinderConfig):
        super().__init__(settings)
        
        self.domain = 'https://html.duckduckgo.com/html/'
        self.language = settings.language or 'en-us'
        self.session = self._load_new_session()

    def _load_new_session(self):
        session = requests.Session(
            impersonate='firefox'
        )
        session.headers.update({
            "Accept-Language": self.language
        })
        return session

    def _load_search_item(self, search_item):
        self.term = search_item.get('term')
        self._blacklist = search_item.get('blacklist', [])

    def _get_response(self):
        params = {
            "q": parse.quote(self.term),
            "kl": self.language
        }
        resp = self.session.get(
            self.domain,
            params=params,
        )
        resp.raise_for_status()
        return resp

    def parse_response(self, response) -> Tuple[pd.DataFrame, str]:
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        date = time.strftime('%Y%m%d')
        for index, result in enumerate(soup.select("div.result")):
            title_element = result.select_one("a.result__a")
            snippet_element = result.select_one("a.result__snippet, div.result_snippet")

            if not title_element:
                continue
                
            title = title_element.get_text(strip=True)
            url = self.strip_ddg_redirect(title_element.get("href"))
            snippet = snippet_element.get_text()

            results.append({
                'seqno': index + 1,
                'url_se': url,
                'snippet': snippet,
                'date': date,
                'query': self.term,
                'pagemap': '' # return this column because of compatibility with googlsearch.py
            })

        return pd.DataFrame(results), ''
    
    def _process_query(self) -> Tuple[pd.DataFrame, str]:
        exclude = self.excluded_sites()
        if exclude:
            self.term = self.term.replace(' ', '+') + ' ' + exclude
        response = self._get_response()
        # Not sure why we sleep here, assuming rate-limiting... legacy code
        time.sleep(3)
        return self.parse_response(response)

    @staticmethod
    def strip_ddg_redirect(url: str) -> str:
        parsed = urlparse(url)

        if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
            qs = parse_qs(parsed.query)
            if "uddg" in  qs:
                return unquote(qs["uddg"][0])
            return url