import yaml

from urlfinding.search import Search
from urlfinding.search_engine import SearchEngine
from urlfinding.url_classifier import UrlClassifier
from urlfinding.extract import Extract

from urlfinding.common import UrlFindingDefaults, MappingsConfig, UrlFinderConfig
from pathlib import Path

from typing import List

class UrlFinder:
    
    def __init__(self, 
                 url_finder_config: UrlFinderConfig, 
                 mappings: MappingsConfig, 
                 population_path: str = None,
                 working_directory: str = None,
                 classifier_path: str = None,
                 url_blacklist: List[str] | None = None):

        self.config = url_finder_config
        self.search_engine = SearchEngine.from_config(self.config)

        self.url_blacklist = url_blacklist
        
        self.mappings = mappings      
        self.population_path = Path(population_path or UrlFindingDefaults.POPULATION)
        self.working_directory = Path(working_directory or UrlFindingDefaults.CWD)

        self.searcher = Search(self.search_engine, self.mappings, self.population_path, 
                               self.url_blacklist, working_directory=self.working_directory)

        self.classifier_path = classifier_path        
        self.url_classifier = UrlClassifier(self.classifier_path, self.mappings, self.population_path)
        self.extractor = Extract(self.mappings, self.population_path, self.working_directory, 
                               self.url_blacklist)
    
    @classmethod
    def from_paths(cls, 
                 url_finder_config_path: str = None, 
                 mappings_path: str = None, 
                 population_path: str = None,
                 working_directory: str = None,
                 classifier_path: str = None,
                 url_blacklist_path: str = None):
        config = UrlFindingDefaults.get_search_engine_config(url_finder_config_path)
        mappings = UrlFindingDefaults.get_mappings_config(Path(mappings_path or UrlFindingDefaults.MAPPINGS))
        url_blacklist = UrlFindingDefaults.read_linesplit_list_csv(url_blacklist_path)        
        return cls(config, mappings, population_path, 
                   working_directory, classifier_path, url_blacklist)
    
    def search(self, nrows: int):
        return self.searcher.run(self.config, nrows, self.population_path)
        

    def extract():
        pass
