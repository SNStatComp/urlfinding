# common.py
import os
import yaml
from pydantic import BaseModel
from typing import Dict, List
from bidict import bidict


class MappingsConfig(BaseModel):
    mapping: Dict[str, str]
    computed: Dict[str, List[str]]
    search: Dict[str, List]
    features: List[str]
    train: Dict

    def get_bidict_mapping(self) -> bidict:
        return bidict(self.mapping)


class UrlFindingDefaults:
    CWD = os.getcwd()
    MAPPINGS = f'{CWD}/config/mappings.yml'
    POPULATION = f'{CWD}/data/companies.csv'

    @staticmethod
    def get_mappings_config(mappings_path: str) -> MappingsConfig:
        with open(mappings_path, 'r') as f:
            config = yaml.safe_load(f)
        mappings_config = MappingsConfig(
            mapping={val: key for key, val in config['input']['columns'].items() if
                     (val is not None) and (not isinstance(val, list))},
            computed={key: val for key, val in config['input']['columns'].items() if isinstance(val, list)},
            search=config['search'],
            features=config['features'],
            train=config['train']
        )
        return mappings_config
