import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from pylab import rcParams
from IPython.display import SVG,display
from sklearn import metrics
from dtreeviz.trees import dtreeviz

from column_names import *

class AnalysisModel:
    CONFIG = {
        "target": SEVERITY,
        "drop": [DATE]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename=""):
        if classes is None:
            classes = ["LEVE", "GRAVE", "OBITO"]      
        self.classes = classes
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.model = None
        self.type = "model"
        self.filename = filename
        self.current_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        return self
        
    def test_model(self):
        self.run_model_score()
        self.predictions = self.model.predict(self.test)
        print(self.predictions)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        self.precision = metrics.precision_score(self.test_labels, self.predictions, average='weighted')
        self.recall = metrics.recall_score(self.test_labels, self.predictions, average='weighted')
        self.f1 = metrics.f1_score(self.test_labels, self.predictions, average='weighted')
        return self

    def run_model_score(self):
        self.model.score(self.test, self.test_labels)
    
    def print_results(self, verbose=False):
        print(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + " - " + self.type)
        print("Accuracy: " + str(self.accuracy))
        print("Confusion Matrix:" + str(self.confusion_matrix))
        print("Classification Report: " + str(self.classification_report))
        if (verbose):
            print("Precision: " + self.precision)
            print("Recall: " + self.recall)
            print("F1: " + self.f1)
        print("")
        return self
    
    def plot_roc_curve(self):
        if len(self.test_labels.unique()) > 2:
            print("ROC curve not supported for more than 2 classes")
            return
        self.fpr, self.tpr, _ = metrics.roc_curve(self.test_labels, self.predictions)
        self.roc_auc = metrics.roc_auc_score(self.test_labels, self.predictions)
        plt.figure()
        lw = 2
        plt.plot(self.fpr, self.tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        return self
        
    def visualize_model(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc")
        dot_data = dtreeviz(self.get_visualized_model(), 
                            self.classes, 
                            self.train.columns,
                            feature_names=self.train.columns,
                            class_names=self.classes)
        graph = graphviz.Source(dot_data)
        graph.render(filename)
        display(SVG(graph.pipe(format='svg')))
        return self
    
    def visualize_feature_importance(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc_Feature-Importance")
        feature_imp = pd.Series(self.model.feature_importances_, index=self.test.columns)
        matplotlib.rc('figure', figsize=(30, 10))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.savefig(filename, format="png")
        
    def get_visualized_model(self):
        return self.model        
      
    def make_filename(self, filename):
        return "results/" + self.current_time + " " + self.type + " - " + filename + "-" + self.filename
              
        
    @staticmethod
    def run_classificator(clazz, data, visualize=True):
        classificator = clazz(data.train, data.test, data.train_labels, data.test_labels)
        classificator.train_model()
        classificator.test_model()
        classificator.print_results()
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
        
        best.visualize_model(best.make_filename("best_of_" + str(n) + "-" + best.accuracy + "acc"))
        return best

