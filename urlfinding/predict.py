import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
from yaml import load, FullLoader
import os

cwd = os.getcwd()
MAPPINGS = f'{cwd}/config/mappings.yml'
POPULATION = './data/companies.csv'

def predict(model, data, pop):
    try:
        X = data[model['model_features']]
    except:
        raise Exception('columns of dataset must match features in model.')
        
    y_pred = model['model'].predict(X)
    y_prob = model['model'].predict_proba(X)
    res = pd.concat(
        [data,
         pd.DataFrame(y_pred,columns=['eqPred']),
         pd.DataFrame(y_prob[:,1], columns=['pTrue'])],
        axis=1)
    pred = res.iloc[res.groupby(['Id'])['pTrue'].idxmax()]
    result = pd.merge(pop,
                      pred[['Id', 'host', 'eqPred', 'pTrue']],
                      how='right', on=['Id'])
    return result, res

def start(feature_file, model_file, base_file):
    with open(MAPPINGS,'r') as f:
        config = load(f, Loader=FullLoader)
    recid_base = config['columns']['Id']

    model = joblib.load(model_file)
    data = pd.read_csv(feature_file, sep=';')
    pop = pd.read_csv(POPULATION, sep=';')
    result, _ = predict(model, data, pop)

    inp = pd.read_csv(base_file, sep=';')
    inp_url = inp.merge(result[['Id', 'host', 'eqPred', 'pTrue']], left_on=recid_base, right_on='Id', how='left')
    inp_url.drop(['Id'], axis=1, inplace=True)
    base = os.path.splitext(base_file)[0]
    inp_url.to_csv(f'{base}_url.csv', sep=';')
    print(f'Predictions saved in {base}_url.csv')

