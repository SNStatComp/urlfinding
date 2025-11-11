# common.py
import os
import yaml
from pydantic.dataclasses import dataclass
from pydantic import Field, BaseModel

from typing import Dict, List, Optional
from bidict import bidict
from pathlib import Path

@dataclass
class MappingsConfig:
    mapping: Dict[str, str]
    computed: Dict[str, List[str]]
    search: Dict[str, List]
    features: List[str]
    train: Dict

    def get_bidict_mapping(self) -> bidict:
        return bidict(self.mapping)

class UrlFinderConfig(BaseModel):
    search_engine: str
    geolocation: str
    language: str
    ddg_language: Optional[str] = Field(default=None, alias='DDGlanguage')
    ddg_user_agent: Optional[str] = Field(default=None, alias='DDGuser-agent')    
    key: Optional[str] = None
    searchengineid: Optional[str] = None

    class Config:
        validate_by_name = True  # allows using field names if aliases aren't used

class UrlFindingDefaults:
    CWD = os.getcwd()
    MAPPINGS = Path(CWD) / 'config/mappings.yml'
    POPULATION = Path(CWD) / 'data/companies.csv'

    @staticmethod
    def get_mappings_config(mappings_path: str | Path) -> MappingsConfig:
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
    
    @staticmethod
    def get_search_engine_config(config_path: str | Path) -> UrlFinderConfig:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return UrlFinderConfig(**config)

    @staticmethod
    def read_linesplit_list_csv(file_path: str | Path) -> List[str]:
        if file_path:
            with open(file_path, 'r') as f:
                return f.read().splitlines()
        return []
    