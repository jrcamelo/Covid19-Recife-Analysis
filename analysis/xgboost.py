from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

from analysis.base import AnalysisModel
from column_names import *
from printer import Printer


class XGBoost(AnalysisModel):
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
        self.model = XGBClassifier(n_estimators=100,
                                   use_label_encoder=False,
                                  #  min_child_weight=1,
                                  #  subsample=0.8,
                                  #  colsample_bytree=0.8,
                                  #  gamma=0,
                                  #  reg_alpha=0,
                                  #  reg_lambda=1,
                                  #  learning_rate=0.1,
                                  #  max_depth=3,
                                   n_jobs=-1)
        self.type = "XGBoost"
        
    def visualize_model(self, filename=None):
        Printer.print("Can't visualize XGBoost yet")
        return self
