import pandas as pd
import graphviz
from IPython.display import SVG,display
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn import metrics
from dtreeviz.trees import dtreeviz
import pdb

from column_names import SEVERITY, AGE, DATE, VACCINATION_PERCENTAGE

class KNN:
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
    
    def __init__(self, train, test, train_labels, test_labels, classes, n_neighbors):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.classes = classes
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights='distance',
                                          algorithm='auto',
                                          p=2,
                                          metric_params=None,
                                          n_jobs=1)    
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        
    def test_model(self):
        self.model.score(self.test, self.test_labels)
        self.predictions = self.model.predict(self.test)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        self.precision = metrics.precision_score(self.test_labels, self.predictions, average='weighted')
        self.recall = metrics.recall_score(self.test_labels, self.predictions, average='weighted')
        self.f1 = metrics.f1_score(self.test_labels, self.predictions, average='weighted')
        
    def visualize_model(self, filename):
        dot_data = dtreeviz(self.model, self.classes, self.train.columns,
                            feature_names=self.train.columns,
                            class_names=self.classes)
        graph = graphviz.Source(dot_data)
        graph.render(filename)
        display(SVG(graph.pipe(format='svg')))
        

def run_knn(data, visualize=True):
    classificator = KNN(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
    classificator.train_model()
    classificator.test_model()
    if (visualize):
        print(classificator.accuracy)
        print(classificator.confusion_matrix)
        print(classificator.classification_report)
        classificator.visualize_model("results/knn_" + str(classificator.accuracy))
    return classificator

def run_multiple_knn(data):
    knns = []
    acc = []
    for i in range(100):
        data.split_data()
        knn = run_knn(data, False)
        knns.append(knn)
        acc.append(knn.accuracy)
    print("Average Accuracy: " + str(sum(acc)/len(acc)))
    print("Max Accuracy: " + str(max(acc)))
    print("Min Accuracy: " + str(min(acc)))
    print("")
    
    best = max(knns, key=lambda x: x.accuracy)
    print(best.accuracy)
    print(best.confusion_matrix)
    print(best.classification_report)
    
    best.visualize_model("best_of_100_knns_" + str(best.accuracy))
    return best

def get_best_knn_params(data):
    knn = run_knn(data, False)
    knn.train_model()
    knn.test_model()
    param_grid = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    }
    knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=1, cv=3, n_jobs=-1)
    knn_grid.fit(data.train, data.train_labels)
    print(knn_grid.best_params_)
    print(knn_grid.best_score_)
    print(knn_grid.best_estimator_)
    return knn_grid
    
        
