# urlfinding
Generic software for finding websites of enterprises using a search engine and machine learning.

This repo is still work in progress, keep an eye on its progress.

## Introduction
This repository contains the software that was used for research on finding enterprise websites.
For a detailed description of the methodology implemented we refer to the
discussion paper on this subject from Statistics Netherlands:

[Searching for business websites](https://www.cbs.nl/en-gb/background/2020/01/searching-for-business-websites), by Arnout van Delden, Dick Windmeijer and Olav ten Bosch

In short the software operates as follows:
* training a model for finding websites using google search for predicting in the train and test phase
* applying the trained model to a dataset with unknown URLs using google search in the predict phase

This process model is shown in the figure below:
![process model](docs/urlfinding_process_model.png)

It is possible to skip the train and test phase and use the pre-trained model that is provided in this repository.

## Quick start: finding websites of NSIs

In this example we search for websites of Nationa Statistical Offices (NSIs) using a pre-trained model provided in this repo.

<TODO> 

## Software modules

### The Google Search module

### Using the pre-trained model

### Training using your own dataset

