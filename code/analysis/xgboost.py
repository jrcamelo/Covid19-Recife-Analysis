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
    
    PARAMS = {
        "n_estimators": [100, 200],
        "min_child_weight": [1, 3, 5, 7, 9, 10],
        "subsample": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "gamma": [0, 0.1, 0.4, 0.7, 1.0],
        "reg_alpha": [0, 0.1, 0.4, 0.7, 1.0],
        "reg_lambda": [0, 0.1, 0.4, 0.7, 1.0],
        "learning_rate": [0.01, 0.1, 1.0],
        "max_depth": [1, 4, 7, 10],
        "random_state": [0]
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
    
    def make_shap_values(self, show=True, approximate=False):
        if (self.binary_only and self.get_classes_count() > 2):
            Printer.print("SHAP only supported for binary classification for " + self.type)
            return
        explainer = shap.TreeExplainer(self.model, approximate=approximate)
        shap_values = explainer.shap_values(self.train)
        Printer.print("SHAP values BAR for " + self.type + " " + self.filename)
        shap.summary_plot(shap_values, self.train, plot_type="bar", show=show, feature_names=self.get_beautified_column_names(), class_names=self.get_beautified_classes())
        plt.savefig(self.make_filename("SHAP-Bar") + ".png", format='png', dpi=1000, bbox_inches='tight')
        plt.clf()
        classes_count = self.get_classes_count()
        for i in range(classes_count):
            if (classes_count == 2 and i == 1):
                continue
            try:
                Printer.print("SHAP values PLOT " + str(i) + " for " + self.type + " " + self.filename)
                shap.summary_plot(shap_values, self.train, plot_type="dot", show=show, feature_names=self.get_beautified_column_names(), class_names=self.get_beautified_classes())
                plt.savefig(self.make_filename("SHAP-Plot-" + str(i)) + ".png", format='png', dpi=1000, bbox_inches='tight')
                plt.clf()
            except AssertionError as e:
                Printer.print("\n\nERROR at SHAP values PLOT " + str(i) + " for " + self.type + " " + self.filename + "\n\n")
                print(e)
                                
        return shap_values
