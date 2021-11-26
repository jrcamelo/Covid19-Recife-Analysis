import pandas as pd
import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from dtreeviz.trees import dtreeviz

from column_names import *

class AnalysisModel:
    CONFIG = {
        "target": SEVERITY,
        "drop": [DATE]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes=None):
        self.classes = classes
        if classes == None:
            self.classes = train_labels.unique()
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.model = None
        self.type = "model"
        self.current_time = pd.datetime.now().strftime("%m-%d-%H-%M-%S")
        
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
        if filename == None:
            filename = self.make_filename(self.accuracy + "acc")
        dot_data = dtreeviz(self.model, self.classes, self.train.columns,
                            feature_names=self.train.columns,
                            class_names=self.classes)
        graph = graphviz.Source(dot_data)
        graph.render(filename)
        display(SVG(graph.pipe(format='svg')))
      
    def make_filename(self, filename):
        return "results/" + self.current_time + " " + self.type + " - " + filename
        
        
    @staticmethod
    def run_classificator(clazz, data, visualize=True):
        classificator = clazz(data.train, data.test, data.train_labels, data.test_labels)
        classificator.train_model()
        classificator.test_model()
        print(classificator.type)
        print(classificator.accuracy)
        print(classificator.confusion_matrix)
        print(classificator.classification_report)
        print("-")
        if (visualize):
            classificator.visualize_model()
        return classificator
     
    @staticmethod 
    def run_multiple_classificators(clazz, data, n=20):
        classificators = []
        acc = []
        for _ in range(n):
            data.split_data()
            classificator = AnalysisModel.run_classificator(clazz, data, False)
            classificators.append(classificator)
            acc.append(classificator.accuracy)
        
        print("Average Accuracy: " + str(sum(acc)/len(acc)))
        print("Max Accuracy: " + str(max(acc)))
        print("Min Accuracy: " + str(min(acc)))
        print("")
        
        best = max(classificators, key=lambda x: x.accuracy)
        print(best.accuracy)
        print(best.confusion_matrix)
        print(best.classification_report)
        
        best.visualize_model(best.make_filename("best_of_" + str(n)))
        return best

