import yaml
import os
from pydantic.dataclasses import dataclass
from pydantic import Field
from pydantic import TypeAdapter

from typing import Dict, List

@dataclass
class MappingsConfig:
    mapping: Dict[str, str]
    computed: Dict[str, List[str]]
    search: Dict[str, List]
    features: List[str]
    train: Dict

@dataclass
class UrlFinderConfig:
    search_engine: str
    key: str
    searchengineid: str
    geolocation: str
    language: str
    ddg_language: str = Field(alias='DDGlanguage')
    ddg_user_agent: str = Field(alias='DDGuser-agent')

class UrlFinder:
    cwd = os.getcwd()
    MAPPINGS = f'{cwd}/config/mappings.yml'
    POPULATION = f'{cwd}/data/companies.csv'
    
    def __init__(self, url_finder_config_path: str = None, mappings_path: str = None, population_path: str = None):
        with open(url_finder_config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        self.config = TypeAdapter(UrlFinderConfig).validate_python(raw_config)
        self.mappings_path = mappings_path or UrlFinder.MAPPINGS
        self.population_path = population_path or UrlFinder.POPULATION

    @staticmethod
    def get_mappings_config(mappings_path: str) -> MappingsConfig:
        with open(mappings_path, 'r') as f:
            config = yaml.safe_load(f)
        mappings_config = MappingsConfig(
            mapping = {val:key for key, val in config['input']['columns'].items() if (val is not None) and (not isinstance(val, list))},
            computed = {key:val for key, val in config['input']['columns'].items() if isinstance(val, list)},
            search = config['search'],
            features = config['features'],
            train = config['train']
        )
        return mappings_config
