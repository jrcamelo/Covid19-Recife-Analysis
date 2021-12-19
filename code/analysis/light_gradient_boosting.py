from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import shap

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