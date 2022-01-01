from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import shap
import numpy as np

from analysis.base import AnalysisModel
from column_names import *
from printer import Printer

class LightGradientBoosting(AnalysisModel):
    CONFIG = {
        "target": SEVERITY,
        "should_binary_severe": False,
        "should_categorize_age": True,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "should_normalize": True,
        "drop_diseases": False,
        "drop_symptoms": False,
        "drop": [DATE]
    }
    
    PARAMS = {
        "n_estimators": [100,],
        "learning_rate": [0.06, 0.1, 0.4],
        "max_depth": [5],
        "subsample": [None, 0.1],
        "random_state": [0]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename=""):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.model = LGBMClassifier(n_estimators=100, 
                                    learning_rate=0.1, 
                                    max_depth=5, 
                                    random_state=0)
        self.type = "LightGradientBoosting"
        
    def visualize_model(self, filename=None):
        Printer.print("Can't visualize Light Gradient Boosting yet")
        return self
    
    def make_shap_values_table(self):
        if (self.binary_only and self.get_classes_count() > 2):
            Printer.print("SHAP only supported for binary classification for " + self.type)
            return
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train)
        expected_values = explainer.expected_value
        avg_shap_values_severe = np.mean(shap_values[1], axis=0)
        avg_shap_values_mild = np.mean(shap_values[0], axis=0)
        
        # Filter shap_values for only rows where expected_value is not 0
        avg_severe_shap_values_severe = np.mean(avg_shap_values_severe[expected_values[1] != 0], axis=0)
        
        for col, severe, mild in zip(self.get_beautified_column_names(), avg_severe_shap_values_severe, avg_shap_values_mild):
            severe_str = "{:.1f}".format(10000*((severe/50).round(8))) + "%"
            Printer.print(severe_str)