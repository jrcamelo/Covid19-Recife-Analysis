import pandas as pd
import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from dtreeviz.trees import dtreeviz


from column_names import SEVERITY, AGE, DATE

class RandomForest:
    CONFIG = {
        "target": SEVERITY,
        "should_categorize_age": True,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "should_categorize_vaccination": True,
        "drop_diseases": False,
        "drop_symptoms": False,
        "drop": [DATE]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.classes = classes
        self.model = RandomForestClassifier(bootstrap=True, 
                                            class_weight=None, 
                                            criterion='gini',
                                            max_depth=None, 
                                            max_features='auto', 
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, 
                                            min_impurity_split=None,
                                            min_samples_leaf=0.00001, #
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0, 
                                            n_estimators=100,  #
                                            n_jobs=4,
                                            oob_score=False, 
                                            random_state=None, 
                                            verbose=0,
                                            warm_start=False)
    
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        
    def test_model(self):
        self.model.score(self.test, self.test_labels)
        self.predictions = self.model.predict(self.test)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        
    def visualize_model(self, filename):
        feature_imp = pd.Series(self.model.feature_importances_, index=self.test.columns)
        matplotlib.rc('figure', figsize=(30, 10))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.savefig(filename + "-feature-importance.png", format="png")
        
        viz = dtreeviz(self.model.estimators_[-1], 
                       self.test, 
                       self.test_labels,
                       target_name="target",
                       feature_names=self.test.columns,
                       class_names=self.classes,
                       
                       )
        viz.save(filename + ".svg")
        #viz.open()

def run_random_forest(data):
    forest = RandomForest(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
    forest.train_model()
    forest.test_model()
    print(forest.accuracy)
    print(forest.confusion_matrix)
    print(forest.classification_report)
    forest.visualize_model("results/random_forest_" + str(forest.accuracy))

def run_multiple_random_forests(data):
    trees = []
    acc = []
    for i in range(100):
        data.split_data()
        tree = RandomForest(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
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
    
    best_tree.visualize_model("best_of_100_random_forest_" + str(best_tree.accuracy))