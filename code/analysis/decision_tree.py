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
        self.model = DecisionTreeClassifier(max_depth=5000,
                                            class_weight=None,
                                            criterion='gini',
                                            max_features='auto', 
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=0.0001,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            random_state=None,
                                            splitter='best')
        self.type = "DecisionTree"
        
    def visualize_model(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc-tree")
        export_graphviz(self.model, 
                        out_file=filename,
                        feature_names=self.get_beautified_column_names(),
                        class_names=self.get_beautified_classes(), 
                        filled=True, 
                        rounded=True,
                        proportion=True,
                        node_ids=True,
                        rotate=False,
                        label='all')
        graph = graphviz.Source.from_file(filename)
        graph.render(filename + ".svg", view=False, format='svg')
        return self
