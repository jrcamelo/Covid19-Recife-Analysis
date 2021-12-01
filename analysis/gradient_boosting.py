from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

from analysis.base import AnalysisModel
from column_names import *

import shap
class GradientBoosting(AnalysisModel):
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
    
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename=""):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.model = GradientBoostingClassifier(n_estimators=100, 
                                                learning_rate=0.1, 
                                                max_depth=5, random_state=0)
        
    def visualize_model(self, filename=None):
        print("Can't visualize Gradient Boosting yet")
        return self
      