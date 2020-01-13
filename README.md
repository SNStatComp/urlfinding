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

This software uses [Google custom search JSON API](https://developers.google.com/custom-search/v1/overview)
which offers 100 search queries per day for free. Use the paid version if you need more.

To get started configure a custom search engine and get your API key from the link above.
Make sure to enable the search engine feature 'Search whole internet'.
Then add the API key and the search engine ID to the `config.yml` file of the software.

## Install urlfinding

We assume an Python Anaconda distribution

Use the following commands to install urlfinding.
```bash
git clone https://github.com/SNStatComp/urlfinding.git # or download and unzip this repository
cd urlfinding
python setup.py install
```
## Quick start: finding websites of NSIs

The examples folder contains several quickstart Python notebooks.
Notebook `nsis_pretrained.pynb` contains an example to search for websites of National Statistical Offices (NSIs) using the pre-trained model provided in this repo.

Notebook `nsis_train.pynb` contains an example to search for websites of National Statistical Offices NSIs) including the training phase.

## API

Include the `urlfinding` module as follows
```
import urlfinding as uf
```
Then you have the following functions:

### Scrape

### Process

### Predict

### Train

