from dtreeviz.trees import dtreeviz
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt

from analysis.base import AnalysisModel
from column_names import *

class RandomForest(AnalysisModel):
    CONFIG = {
        "target": SEVERITY,
        "should_categorize_age": True,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "should_normalize": True,
        "drop_diseases": False,
        "drop_symptoms": True,
        "drop": [DATE],
    }
      
    def __init__(self, train, test, train_labels, test_labels, classes, filename=""):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.model = RandomForestClassifier(bootstrap=True, 
                                            class_weight=None, 
                                            criterion='gini',
                                            max_depth=10000, 
                                            max_features='auto', 
                                            min_impurity_decrease=0.0, 
                                            min_impurity_split=None,
                                            min_samples_leaf=0.001,  #0.00001, #
                                            #min_samples_split=2,
                                            min_weight_fraction_leaf=0.0, 
                                            n_estimators=10,  #100
                                            n_jobs=-1,
                                            oob_score=True, 
                                            random_state=None, 
                                            verbose=0,
                                            warm_start=False)
    
    def visualize_model(self, filename=None): 
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc")
        viz = dtreeviz(self.model.estimators_[-1], 
                       self.test, 
                       self.test_labels,
                       target_name="target",
                       feature_names=self.test.columns,
                       class_names=self.classes)
        viz.save(filename + ".svg")
    
    def make_shap_values(self, show=True):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train)
        plt.rcParams.update({'figure.figsize': (100, 60)})
        plt.close()
        shap.summary_plot(shap_values, self.train, show=show)
        plt.savefig(self.make_filename("shap_values_bar"), format='png')
        plt.close()
        shap.summary_plot(shap_values[-1], self.train, plot_type="dot", show=show)
        plt.savefig(self.make_filename("shap_values_dot"), format='png')
        return shap_values
    
def get_best_params(data):
    forest = RandomForest(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
    forest.train_model()
    forest.test_model()
    param_grid = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [10, 50],
        'max_leaf_nodes': [2, 5, 10],
        'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
        'min_samples_leaf': [0.5, 0.1, 0.001, 0.005, 0.0001],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001],
        'criterion': ['gini', 'entropy'],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.5],
        "class_weight": ['balanced', None],
    }
    CV_rfc = GridSearchCV(estimator=forest.model, param_grid=param_grid, cv= 5)
    CV_rfc.fit(data.df, data.target)
    print(CV_rfc.best_params_)    
    
    # Result:
    # max_features: auto
    # max_depth: 10000
    # max_leaf_nodes: 5  / 10 with binary severity
    # min_samples_leaf: 0.5   / 1 with binary severity
    # min_samples_split: 2   / 5 with binary severity
    # min_impurity_decrease: 0.0
    # criterion: gini
    # min_weight_fraction_leaf: 0.0
    # class_weight: None