from abc import  ABC, abstractmethod
import pandas as pd
from typing import Tuple

class SearchEngine(ABC):

    def __init__(self, settings):
        self._blacklist = []
        self.columns = ['date', 'seqno', 'query', 'title', 'snippet', 'url_se', 'pagemap']
    @abstractmethod
    def _processQuery(self) -> Tuple[pd.DataFrame, str]:
        raise NotImplementedError       
    
    @abstractmethod
    def _load_search_item(self, search_item):
        raise NotImplementedError  

    def search(self, search_item) -> Tuple[pd.DataFrame, str]:
        self._load_search_item(search_item)
        return self._processQuery()
    
    def excludedSites(self) -> str:
        return ' '.join(['-site:' + url for url in self._blacklist])