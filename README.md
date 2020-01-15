# urlfinding
Generic software for finding websites of enterprises using a search engine and machine learning.

This repo is still work in progress.

## Introduction
This repository contains the software that was used for research on finding enterprise websites.
For a detailed description of the methodology implemented we refer to the
discussion paper on this subject from Statistics Netherlands:

[Searching for business websites](https://www.cbs.nl/en-gb/background/2020/01/searching-for-business-websites) by Arnout van Delden, Dick Windmeijer and Olav ten Bosch

In short the software operates as follows:
* training a model for finding websites using google search for predicting in the train and test phase
* applying the trained model to a dataset with unknown URLs using google search in the predict phase

This process model is shown in the figure below:
![process model](docs/urlfinding_process_model.png)

It is possible to skip the train and test phase and use the pre-trained model that is provided in this repository.

## Google Search IDs

This software uses the *Google custom search JSON API*
which offers 100 search queries per day for free. Use the paid version if you need more.

To get started configure a custom search engine and get your API key from [here](https://developers.google.com/custom-search/v1/overview).
Make sure to enable the search engine feature 'Search whole internet'.
Then add the API key and the search engine ID to the `config.yml` file.

## Install urlfinding

Assuming a Python Anaconda distribution, use the following commands to install urlfinding:
```bash
git clone https://github.com/SNStatComp/urlfinding.git # or download and unzip this repository
cd urlfinding
python setup.py install
```
## Quick start: finding websites of NSIs

The examples folder contains a Python notebook [examples/nsis.ipynb](examples/nsis.ipynb) showing how to search for websites of National Statistical Offices (NSIs)
using the pre-trained model provided in this repo and how to train your own model.

## API

Include the `urlfinding` module as follows:
```
import urlfinding as uf
```
Then you have the following functions:

### Scrape

`uf.scrape.start(base_file, googleconfig, blacklist, nrows)`

This function startes a Google search.

- `base_file`: A .csv file with a list of enterprises for which you want to find the webaddress. If you want to use the pretrained ML model provided (data/model.pkl_) the file must at least include the following columns: _id, tradename, legalname, address, postalcode and locality. The column names can be specified in a mapping file (see config/mappings.yml for an example)

- `googleconfig`: This file contains your credentials for using the Google custom search engine API

- `blacklist`: A file containing urls you want to exclude from your search

- `nrows`: Number of enterprises you want to search for. Google provides 100 queries per day for free. In this example for every enterprise 6 queries are performed, thus for 10 enterprises 6 * 10 = 60 queries. Every query returns at most 10 search results.

This function creates a file (<YYYYMMDD_>_searchResult.csv_) in the _data folder containing the search results, where YYYYMMDD is the current date.


### Process

`uf.process.start(date, data_files, blacklist)`

This function created a feature file to be used for training your Machine Learning model or predicting using your an already trained model.


- `date`: Used for adding a 'timestamp' to the name of the created feature file

- `data_files`: list of files containing the search results

- `blacklist`: see above

This function creates the feature file <YYYYMMDD_>_features___agg.csv in the data folder


### Predict

`uf.predict.start(feature_file, model_file, base_file)`

This function predicts urls using a previously trained ML model.

- `feature_file`: file containing the features

- `model_file`: Pickle file containing the ML model (created with our package)

- `base_file`: See base_file at `uf.scrape.start()`

This function creates the file <base__file>_url.csv in the data folder containing the predicted urls. This file contains all data from the base file with 3 columns added:

- `host`: the predicted url

- `eqPred`: An indicator showing whether the predicted url is the right one

- `pTrue`: An indicator showing the confidence of the prediction, a number between 0 and 1 where 0: almost certain not the url and 1: almost certain the right url. eqPred is derived from pTrue: if pTrue>0.5 then eqPred=True else eqPred=False

### Train

