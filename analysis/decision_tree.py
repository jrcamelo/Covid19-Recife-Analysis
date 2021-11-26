import shap
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from column_names import SEVERITY, AGE, DATE

class DecisionTree:
    CONFIG = {
        "target": SEVERITY,
        "should_categorize_age": True,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "should_normalize": True,
        "drop_diseases": False,
        "drop_symptoms": True,
        "drop": [DATE]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes, max_depth=None):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.classes = classes
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=self.max_depth,
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
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        
    def test_model(self):
        self.predictions = self.model.predict(self.test)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        self.precision = metrics.precision_score(self.test_labels, self.predictions, average='weighted')
        self.recall = metrics.recall_score(self.test_labels, self.predictions, average='weighted')
        self.f1 = metrics.f1_score(self.test_labels, self.predictions, average='weighted')
        
    def visualize_model(self, filename):
        print(self.test_labels.unique())
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
        graph.render(filename, view=True, format='svg')
        
    def make_shap_values(self, filename="decision_tree"):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train)
        shap.summary_plot(shap_values, self.train)
        plt.savefig(filename + "-shap-summary-bar.png", format="png")  
        shap.summary_plot(shap_values[-1], self.train, plot_type="dot")
        return shap_values

def run_decision_tree(data):
    tree = DecisionTree(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
    tree.train_model()
    tree.test_model()
    print(tree.accuracy)
    print(tree.confusion_matrix)
    print(tree.classification_report)
    tree.visualize_model("results/decision_tree_" + str(tree.accuracy))
    return tree

def run_multiple_decision_trees(data):
    trees = []
    acc = []
    for i in range(100):
        data.split_data()
        tree = DecisionTree(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
        tree.train_model()
        tree.test_model()
        trees.append(tree)        
        acc.append(tree.accuracy)
        print(tree.accuracy)
    print("Average Accuracy: " + str(sum(acc)/len(acc)))
    print("Max Accuracy: " + str(max(acc)))
    print("Min Accuracy: " + str(min(acc)))
    print("")
    
    best_tree = max(trees, key=lambda x: x.accuracy)
    print(best_tree.accuracy)
    print(best_tree.confusion_matrix)
    print(best_tree.classification_report)
    
    best_tree.visualize_model("best_of_100_decision_tree_" + str(best_tree.accuracy))
    return best_tree
        
        