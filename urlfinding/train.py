import pandas as pd
import numpy as np
import sklearn.metrics as met
import re
import matplotlib.pyplot as plt 
import pycm
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, PrecisionRecallCurve
from yellowbrick.style.palettes import PALETTES, SEQUENCES
from yaml import load, FullLoader
import seaborn as sns

sns.set()
cwd = os.getcwd()
def createTrainTest(data, pop, id, feat, target):
    bidX = pop.loc[:, [id]]
    #bidy = pop.loc[:, [id]]
    bidX_train, bidX_test, _, _ = train_test_split(bidX, bidX, test_size=0.3, random_state=0)

    train = pd.merge(data, bidX_train, on=id)
    X_train = train.loc[:, feat]
    y_train = train.loc[:, target]
    
    test = pd.merge(data, bidX_test, on=id)
    X_test = test.loc[:, feat]
    y_test = test.loc[:, target]
    return X_train, y_train, X_test, y_test, train, test

def initModel(mla, date, case):
    with open(f'{cwd}/config/{date}hyperparam_{case}.yml', 'r') as f:
        params = load(f, Loader=FullLoader)
    if mla == 'nb':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif mla == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=0)
        model.set_params(**params[mla])
    elif mla == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=0)
        model.set_params(**params[mla])
    elif mla == 'svm':
        from sklearn.svm import SVC
        model = SVC(random_state=0, probability=True)
        model.set_params(**params[mla])
    return model

def createModel(date, X_train, y_train, mla, case, saveModel=True):
    model = initModel(mla, date, case)
    model.fit(X_train, y_train)
    result = {'model': model, 'model_features': X_train.columns.tolist()}
    if saveModel:
        joblib.dump(result,f'./data/{date}{mla}_{case}.pkl')
    return result

def visualizeResults(date, X_train, y_train, X_test, y_test, model):
    plt.rcParams['figure.figsize'] = (10,6)
    plt.rcParams['font.size'] = 14

    visualizer = ClassificationReport(model, classes=[False,True], support='count', cmap='Blues')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}class_rep.png')
    cm = visualizer.scores_

    visualizer = ConfusionMatrix(model, classes=[False,True], cmap='Blues')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}conf_matrix.png')

    visualizer = ROCAUC(model, classes=[False,True])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}ROCAUC.png')

    visualizer = PrecisionRecallCurve(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}prec_recall.png')
    return cm

def getFrequentUrls(df, threshold):
    counts = df['host'].value_counts()
    frac = counts > threshold
    return frac[frac].index.tolist()

def getFrequentUrls2(df, threshold):
    g = df.groupby('host')['Id'].nunique()
    return g[g > threshold].index

def read_datafiles(filelist, popFile):
    pop = pd.read_csv(popFile, sep=';')
    pop = pop
    data = pd.DataFrame()
    for dataFile in filelist:
        data = data.append(pd.read_csv(dataFile, sep=';').fillna(2))
    urlsToRemove = getFrequentUrls(data, 60)
    data = data[~data['host'].isin(urlsToRemove)]

    moreUrlsToRemove = getFrequentUrls2(data, 1)
    data = data[~data['host'].isin(moreUrlsToRemove)]
    
    data['seqno'] = data.sort_values(by=['Id', 'queryType', 'seqno'], axis=0).groupby(['Id', 'queryType']).cumcount()/10 + 0.1
    return data.reset_index(), pop

def get_feat_target(cols, case):
    if case =='all':
        feat = [x for x in cols if re.match("^eq.+", x)]
        #feat += ['seqno', 'queryType', 'zscore']
        feat.remove('eqUrl')
    elif case == 'no_mean':
        feat = [x for x in cols if (re.match("^eq.+(min|max)", x))]
    elif case == 'max':
        feat = [x for x in cols if (re.match("^eq.+max", x))]
    elif case == 'top':
        feat = ['eqSnippetAddress_max', 'eqSnippetLocality_max', 'eqSnippetPostalcode_max',
                'eqTitleLegalName_max', 'eqTitleLegalName_mean', 'eqTitleLegalName_min',
                'eqTitleTradeName_max', 'eqTitleTradeName_mean']
    elif case == "no_mean_top12":
        feat = ['eqSnippetPostalcode_max', 'eqTitleLegalName_max', 'eqTitleTradeName_max',
                'eqTitleLegalName_min', 'eqTitleTradeName_min', 'eqSnippetPostalcode_min',
                'eqSnippetLocality_max', 'eqTitleAddress_max', 'eqSnippetAddress_min',
                'eqTitleAddress_min']
    elif case == 'no_mean_top8':
        feat = ['eqSnippetAddress_max', 'eqSnippetAddress_min', 'eqSnippetLocality_max',
                'eqSnippetPostalcode_max', 'eqTitleLegalName_max', 'eqTitleTradeName_max']
    elif case == 'no_score':
        feat = [x for x in cols if (re.match("^eq.+(min|max)", x))]
    if case != 'no_score':
        feat += ['seq_score_perc', 'zscore']
    target = 'eqUrl'
    return feat, target

def start(date, dataFile, popFile, mla, case):
    pop = pd.read_csv(popFile, sep=';')
    data = pd.read_csv(dataFile, sep=';')
    feat, target = get_feat_target(data.columns.tolist(), case)
    X_train, y_train, X_test, y_test, train, test = createTrainTest(data, pop, 'Id', feat, target)
    model = createModel(date, X_train, y_train, mla, case, saveModel=True)
    cm = visualizeResults(date, X_train, y_train, X_test, y_test, model['model'])
    cmdf = pd.DataFrame(data=cm)
    return model, train, test, X_train, y_train, X_test, y_test, cmdf

# mla = 'svm'
# case = 'no_mean'
# date = '20190726'
# featurefile = './data/20190517features_agg.csv'
# population = './data/companies.csv'
# model, train, test, X_train, y_train, X_test, y_test, cm = start(date, featurefile, population, mla, case)