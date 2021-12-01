from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import graphviz
import shap

from column_names import *
from analysis.base import AnalysisModel

class DecisionTree(AnalysisModel):
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
        
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename=""):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.model = DecisionTreeClassifier(max_depth=10000,
                                            class_weight=None,
                                            criterion='gini',
                                            max_features=None,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=0.0001,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            random_state=None,
                                            splitter='best')
        
    def visualize_model(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc-tree")
        export_graphviz(self.model, 
                        out_file=filename,
                        feature_names=self.test.columns,
                        class_names=self.classes, 
                        filled=True, 
                        rounded=True,
                        proportion=True,
                        node_ids=True,
                        rotate=False,
                        label='all')
        graph = graphviz.Source.from_file(filename)
        graph.render(filename + ".svg", view=False, format='svg')
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