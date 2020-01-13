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

## Get your Google Search IDs

This software uses [Google custom search JSON API](https://developers.google.com/custom-search/v1/overview).
They offer 100 search queries per day for free, use the paid version for more.

To get started configure a custom search engine and get your API key, both from the link above.
In the configuration of the search engine make sure you turn on the button 'Search whole internet' or whatever it is called in your language.
Then configure the API key and the search engine ID in the config of the config.yml file of the software.

## Install from source

Use the following commands to install urlfinding.
```bash
git clone https://github.com/SNStatComp/urlfinding.git # or download and unzip this repository
cd urlfinding
python setup.py install
```
## Quick start: finding websites of NSIs

In this example we search for websites of Nationa Statistical Offices (NSIs) using a pre-trained model provided in this repo.

<TODO> 

## Software modules

### The Google Search module

### Using the pre-trained model

### Training using your own dataset

