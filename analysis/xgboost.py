from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from analysis.base import AnalysisModel
from column_names import *

import shap

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
        
    def visualize_model(self, filename=None):
        print("Can't visualize XGBoost yet")
        return self
      
    def make_shap_values(self, show=True):
        explainer = shap.TreeExplainer(self.model)
        plt.rcParams.update({'figure.figsize': (100, 60)})
        shap_values = explainer.shap_values(self.train)
        shap.summary_plot(shap_values, self.train, show=show)
        plt.savefig(self.make_filename("shap_values_bar") + ".png", format='png')
        shap.summary_plot(shap_values[-1], self.train, plot_type="dot", show=show)
        plt.savefig(self.make_filename("shap_values_dot") + ".png", format='png')
        return shap_values