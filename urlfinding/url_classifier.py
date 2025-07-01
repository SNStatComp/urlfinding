from sklearn.base import BaseEstimator
from urlfinding.common import UrlFindingDefaults
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import joblib

from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, \
    ROCAUC, PrecisionRecallCurve
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from pathlib import Path

from typing import Tuple, Dict

sns.set_theme()

class UrlClassifier():    

    def __init__(self, model_path:str, 
                 mappings_path: str = UrlFindingDefaults.MAPPINGS,
                 population_path: str = UrlFindingDefaults.POPULATION):
        self.id_column = 'Id'
        self.target_column = 'eqUrl'
        self.model_path = model_path
        self.population_path = population_path
        self.mappings_config = UrlFindingDefaults.get_mappings_config(mappings_path)

    @staticmethod
    def init_model(classifier: str, hyperparam: dict = None):
        hyperparam = hyperparam or {}

        classifiers = {
            'nb': lambda: GaussianNB(),
            'forest': lambda: RandomForestClassifier(random_state=0, **hyperparam),
            'tree': lambda: DecisionTreeClassifier(random_state=0, **hyperparam),
            'svm': lambda: SVC(random_state=0, probability=True, **hyperparam),
        }

        try:
            return classifiers[classifier]()
        except KeyError:
            raise ValueError(f"Unknown classifier: {classifier}")

    def create_model(self, date, x_train, y_train, classifier, hyperparam, save_model=True) -> BaseEstimator:
        model = self.init_model(classifier, hyperparam)
        model.fit(x_train, y_train)
        result = {'model': model, 'model_features': x_train.columns.tolist()}
        if save_model:
            output_path = Path("./data")
            output_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, output_path / f"{date}{classifier}.pkl")
        return model

    def train(self, date, data_file, save_model=True, visualize_scores=False):
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
        population = pd.read_csv(self.population_path, sep=';')
        data = pd.read_csv(data_file, sep=';')
        train_param = self.mappings_config.train

        classifier = train_param['classifier']
        features, hyperparam = train_param['feature_selection'], train_param['hyperparam'][classifier]
        x_train, y_train, x_test, y_test = self.create_train_test(data, population, features)
        model = self.create_model(date, x_train, y_train, classifier, hyperparam, save_model=save_model)
        if visualize_scores:
            self.visualize_results(date, x_train, y_train, x_test, y_test, model['model'])
    
    def create_train_test(self, data: pd.DataFrame, population: pd.DataFrame, features: list):
        # Select ID column and split
        ids = population[[self.id_column]]
        ids_train, ids_test = train_test_split(ids, test_size=0.3, random_state=0)

        # Merge to create train/test sets
        train = pd.merge(data, ids_train, on=self.id_column)
        test = pd.merge(data, ids_test, on=self.id_column)

        x_train = train[features].copy()
        y_train = train[self.target_column].copy()

        x_test = test[features].copy()
        y_test = test[self.target_column].copy()

        return x_train, y_train, x_test, y_test

    @staticmethod
    def visualize_results(date, x_train, y_train, x_test, y_test, model, working_directory=UrlFindingDefaults.CWD):
        figures_dir = Path(working_directory) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams['figure.figsize'] = (10,6)
        plt.rcParams['font.size'] = 14

        visualizers = [
            (ClassificationReport(model, classes=[False, True], support='count', cmap='Blues'),
             figures_dir / f"{date}class_rep.png"),
            (ConfusionMatrix(model, classes=[False, True], cmap='Blues'),
             figures_dir / f"{date}conf_matrix.png"),
            (ROCAUC(model, classes=[False, True]),
             figures_dir / f"{date}ROCAUC.png"),
            (PrecisionRecallCurve(model),
             figures_dir / f"{date}prec_recall.png"),
        ]

        cm = None
        for visualizer, figure_path in visualizers:
            visualizer.fit(x_train, y_train)
            visualizer.score(x_test, y_test)
            visualizer.show()
            fig = visualizer.ax.get_figure()
            fig.savefig(figure_path)
            # Retrieve from ClassificationReport
            if cm is None and hasattr(visualizer, 'scores_'):
                cm = visualizer.scores_        
        return cm
    
    @staticmethod
    def get_prediction(
        model: Dict[str, any], 
        data: pd.DataFrame, 
        population: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_summary_
        Predict using a trained model and match predictions to population.

        Args:
            model (Dict[str, any]): _description_
            data (pd.DataFrame): _description_
            population (pd.DataFrame): _description_

        Raises:
            Exception: _description_

        Returns:
        - result: DataFrame with predicted 'host', 'eqPred', 'pTrue' merged with population on 'Id'
        - res: Full DataFrame with all rows and predictions
        """
        try:
            features = data[model['model_features']]
        except KeyError:
            raise ValueError("Dataset columns must match features in model.")
        
        clf: BaseEstimator = model['model']

        if not hasattr(clf, "predict_proba"):
            raise TypeError(f"Model of type {type(clf).__name__} does not support `predict_proba`.")

        y_pred = clf.predict(features)
        y_prob = clf.predict_proba(features)[:, 1]

        res = pd.concat([
            data.reset_index(drop=True),
            pd.DataFrame(y_pred, columns=['eqPred']),
            pd.DataFrame(y_prob, columns=['pTrue'])
        ], axis=1)

        # Select highest-confidence prediction per Id
        pred = res.loc[res.groupby('Id')['pTrue'].idxmax()]

        result = pd.merge(population,
                        pred[['Id', 'host', 'eqPred', 'pTrue']],
                        how='right', on='Id')
        return result, res

    def predict(self, features_path: str, model_path: str, input_urls_file: str) -> None:
        """
        Predicts URLs using a previously trained ML model.

        Parameters:
        - features_path: Path to CSV file containing input features.
        - model_file: Path to a .pkl file containing a trained model (created with this package).
        - input_urls_file: Path to the input CSV with original URLs (used for later enrichment).

        Output:
        - A file named <input_urls_file>_url.csv in the same folder,
        containing the original data + columns:
            - 'host': predicted URL
            - 'eqPred': True if pTrue > 0.5
            - 'pTrue': confidence score from 0 to 1
        """

        sep=";"
        coupling_id = self.mappings_config.input['columns']['Id']

        # Load model and data
        model = joblib.load(model_path)
        features_df = pd.read_csv(features_path, sep=sep)
        population_df = pd.read_csv(self.population_path, sep=sep)

        # Predict
        result_df, _ = self.get_prediction(model, features_df, population_df)

        # Load base URLs and merge predictions
        urls_df = pd.read_csv(input_urls_file, sep=';')
        merged_df = urls_df.merge(
            result_df[['Id', 'host', 'eqPred', 'pTrue']], 
            left_on=coupling_id, 
            right_on='Id', 
            how='left').drop(columns=['Id'])
        
        # Save result
        output_path = Path(input_urls_file).with_name(f"{Path(input_urls_file).stem}_url.csv")
        merged_df.to_csv(output_path, sep=sep, index=False)
        print(f"Predictions saved in {output_path}")
