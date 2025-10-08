import yaml
from pydantic import Field, BaseModel

from typing import Optional

from urlfinding.search import Search
from urlfinding.url_classifier import UrlClassifier
from urlfinding.extract import Extract

from urlfinding.common import UrlFindingDefaults


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


class UrlFinder:

    def __init__(self,
                 url_finder_config_path: str = None,
                 mappings_path: str = None,
                 population_path: str = None,
                 working_directory: str = None,
                 classifier_path: str = None,
                 url_blacklist_path: str = None):
        with open(url_finder_config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        self.config = UrlFinderConfig(**raw_config)
        self.mappings_path = mappings_path or UrlFindingDefaults.MAPPINGS
        self.population_path = population_path or UrlFindingDefaults.POPULATION
        self.working_directory = working_directory or UrlFindingDefaults.CWD

        self.search = Search(self.population_path, mappings_path, working_directory=UrlFindingDefaults.CWD)

        self.classifier_path = classifier_path
        self.url_blacklist_path = url_blacklist_path
        self.url_classifier = UrlClassifier(self.classifier_path, self.mappings_path, self.population_path)
        self.extract = Extract(self.mappings_path, self.population_path, self.working_directory,
                               url_blacklist_path)
