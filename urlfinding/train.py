import re
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, \
    ROCAUC, PrecisionRecallCurve
from yaml import load, FullLoader
from urlfinding.common import get_config

sns.set()
cwd = os.getcwd()
MAPPINGS = f'{cwd}/config/mappings.yml'
POPULATION = f'{cwd}/data/companies.csv'

def createTrainTest(data, pop, feat):
    id = 'Id'
    target = 'eqUrl'
    bidX = pop.loc[:, [id]]
    bidX_train, bidX_test, _, _ = train_test_split(bidX, bidX, test_size=0.3, random_state=0)

    train = pd.merge(data, bidX_train, on=id)
    X_train = train.loc[:, feat]
    y_train = train.loc[:, target]

    test = pd.merge(data, bidX_test, on=id)
    X_test = test.loc[:, feat]
    y_test = test.loc[:, target]
    return X_train, y_train, X_test, y_test

def initModel(mla, hyperparam, date):
    if mla == 'nb':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif mla == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=0)
        model.set_params(**hyperparam)
    elif mla == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=0)
        model.set_params(**hyperparam)
    elif mla == 'svm':
        from sklearn.svm import SVC
        model = SVC(random_state=0, probability=True)
        model.set_params(**hyperparam)
    return model

def createModel(date, X_train, y_train, mla, hyperparam, save_model=True):
    model = initModel(mla, hyperparam, date)
    model.fit(X_train, y_train)
    result = {'model': model, 'model_features': X_train.columns.tolist()}
    if save_model:
        joblib.dump(result, f'./data/{date}{mla}.pkl')
    return result

def visualize_results(date, X_train, y_train, X_test, y_test, model):
    if not os.path.exists(f'{cwd}/figures'):
        os.makedirs(f'{cwd}/figures')
    plt.rcParams['figure.figsize'] = (10,6)
    plt.rcParams['font.size'] = 14

    visualizer = ClassificationReport(model, classes=[False, True], support='count', cmap='Blues')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}class_rep.png')
    cm = visualizer.scores_

    visualizer = ConfusionMatrix(model, classes=[False, True], cmap='Blues')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    fig = visualizer.ax.get_figure()
    fig.savefig(f'{cwd}/figures/{date}conf_matrix.png')

    visualizer = ROCAUC(model, classes=[False, True])
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

# def get_feat_target(cols, case):
#     if case == 'all':
#         feat = [x for x in cols if re.match("^eq.+", x)]
#         #feat += ['seqno', 'queryType', 'zscore']
#         feat.remove('eqUrl')
#     elif case == 'no_mean':
#         feat = [x for x in cols if re.match("^eq.+(min|max)", x)]
#     elif case == 'max':
#         feat = [x for x in cols if re.match("^eq.+max", x)]
#     elif case == 'top':
#         feat = ['eqSnippetAddress_max', 'eqSnippetLocality_max', 'eqSnippetPostalcode_max',
#                 'eqTitleLegalName_max', 'eqTitleLegalName_mean', 'eqTitleLegalName_min',
#                 'eqTitleTradeName_max', 'eqTitleTradeName_mean']
#     elif case == "no_mean_top12":
#         feat = ['eqSnippetPostalcode_max', 'eqTitleLegalName_max', 'eqTitleTradeName_max',
#                 'eqTitleLegalName_min', 'eqTitleTradeName_min', 'eqSnippetPostalcode_min',
#                 'eqSnippetLocality_max', 'eqTitleAddress_max', 'eqSnippetAddress_min',
#                 'eqTitleAddress_min']
#     elif case == 'no_mean_top8':
#         feat = ['eqSnippetAddress_max', 'eqSnippetAddress_min', 'eqSnippetLocality_max',
#                 'eqSnippetPostalcode_max', 'eqTitleLegalName_max', 'eqTitleTradeName_max']
#     elif case == 'no_score':
#         feat = [x for x in cols if re.match("^eq.+(min|max)", x)]
#     if case != 'no_score':
#         feat += ['seq_score_perc', 'zscore']
#     target = 'eqUrl'
#     return feat, target

def train(date, dataFile, save_model=True, visualize_scores=False, config=MAPPINGS):
    '''
    This function trains a classifier, saves the trained model and shows some performance figures.
    Before using this function the classifier to train and the features and hyperparameters to use for this classifier
    has to by defined in the mapping file 
    Parameters:
    - date: Used for adding a 'timestamp' to the name of the created model file
    - data_file: The file containing the training data
    - save_model: If True, saves the model (default: True)
    - visualize_scores: If True, shows and saves figures containing performance measures (classification report, confusionmatrix,
    precision recall curve and ROCAUC curve). The figures are saved in the folder 'figures'. (default: False)
    '''
    pop = pd.read_csv(POPULATION, sep=';')
    data = pd.read_csv(dataFile, sep=';')
    _, _, _, _, train_param = get_config(config)
    mla = train_param['classifier']
    feat, hyperparam = train_param['feature_selection'], train_param['hyperparam'][mla]
    X_train, y_train, X_test, y_test = createTrainTest(data, pop, feat)
    model = createModel(date, X_train, y_train, mla, hyperparam, save_model=save_model)
    if visualize_scores:
        visualize_results(date, X_train, y_train, X_test, y_test, model['model'])