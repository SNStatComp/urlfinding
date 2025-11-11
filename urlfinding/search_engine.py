from abc import  ABC, abstractmethod
import pandas as pd
from typing import Tuple
import yaml
from urlfinding.common import UrlFinderConfig


class SearchEngine(ABC):
    _registry = {}

    def __init__(self, settings: UrlFinderConfig):
        self.settings = settings
        self._blacklist = []
        self.output_columns = ['date', 'seqno', 'query', 'title', 'snippet', 'url_se', 'pagemap']

    @abstractmethod
    def _process_query(self) -> Tuple[pd.DataFrame, str]:
        raise NotImplementedError       
    
    @abstractmethod
    def _load_search_item(self, search_item):
        raise NotImplementedError  

    def search(self, search_item) -> Tuple[pd.DataFrame, str]:
        self._load_search_item(search_item)
        return self._process_query()
    
    def excluded_sites(self) -> str:
        return ' '.join(['-site:' + url for url in self._blacklist])
    
    def available_engines(self):
        return self._registry

    @classmethod
    def register(cls, name: str):
        """Decorator function to add subclasses to the registry with a given name

        Args:
            name (str): Name by which the subclass will be referenced in the dictionary.
        """
        def wrapper(subclass):
            cls._registry[name.lower()] = subclass
            return subclass
        return wrapper
    
    @classmethod
    def from_config(cls, settings: UrlFinderConfig) -> "SearchEngine":
        engine = settings.search_engine or 'google'
        try:
            return cls._registry[engine](settings)
        except KeyError:
            raise ValueError(f"Unknown search engine: {engine}")
    
