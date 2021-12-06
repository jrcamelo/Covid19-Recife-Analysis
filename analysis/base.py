import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import shap
from pylab import rcParams
from IPython.display import SVG,display
from sklearn import metrics
from dtreeviz.trees import dtreeviz

from column_names import *
from printer import Printer

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
        self.current_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        self.binary_only = False
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        return self
        
    def test_model(self):
        self.run_model_score()
        self.predictions = self.model.predict(self.test)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        self.precision = metrics.precision_score(self.test_labels, self.predictions, average='weighted')
        self.precision_macro = metrics.precision_score(self.test_labels, self.predictions, average='macro')
        self.recall = metrics.recall_score(self.test_labels, self.predictions, average='weighted')
        self.recall_macro = metrics.recall_score(self.test_labels, self.predictions, average='macro')
        self.f1 = metrics.f1_score(self.test_labels, self.predictions, average='weighted')
        self.f1_macro = metrics.f1_score(self.test_labels, self.predictions, average='macro')
        return self

    def run_model_score(self):
        self.model.score(self.test, self.test_labels)
    
    def print_results(self, verbose=False):
        Printer.print(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + " - " + self.type)
        Printer.print("Accuracy: " + str(self.accuracy))
        Printer.print("Confusion Matrix:" + str(self.confusion_matrix))
        Printer.print("Classification Report: " + str(self.classification_report))
        if (verbose):
            Printer.print("Precision: " + self.precision)
            Printer.print("Recall: " + self.recall)
            Printer.print("F1: " + self.f1)
        Printer.print("")
        return self
    
    def plot_roc_curve(self):
        if len(self.test_labels.unique()) > 2:
            Printer.print("ROC curve not supported for more than 2 classes")
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
        plt.savefig(self.make_filename("roc-curve.png"), format='png')
        return self
        
    def visualize_model(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(self.accuracy) + "acc")
        dot_data = self.make_dtreeviz()
        graph = graphviz.Source(dot_data)
        graph.render(str(filename))
        display(SVG(graph.pipe(format='svg')))
        return self
    
    def make_dtreeviz(self):
        return dtreeviz(self.model, 
                        self.classes, 
                        self.get_beautified_column_names(),
                        class_names=self.get_beautified_classes())
    
    def visualize_feature_importance(self, filename=None):
        if filename == None:
            filename = self.make_filename(str(round(self.accuracy)) + "acc_Feature-Importance")
        feature_imp = pd.Series(self.model.feature_importances_, index=self.get_beautified_column_names())
        matplotlib.rc('figure', figsize=(30, 10))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.savefig(filename, format="png")
        plt.clf()
        
          
    def make_shap_values(self, show=True):
        if (self.binary_only and self.get_classes_count() > 2):
            Printer.print("SHAP only supported for binary classification for " + self.type)
            return
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train)
        Printer.print("SHAP values BAR for " + self.type + " " + self.filename)
        shap.summary_plot(shap_values, self.train, plot_type="bar", show=show, feature_names=self.get_beautified_column_names(), class_names=self.get_beautified_classes())
        plt.savefig(self.make_filename("SHAP-Bar") + ".png", format='png', dpi=1000, bbox_inches='tight')
        plt.clf()
        classes_count = self.get_classes_count()
        for i in range(classes_count):
            if (classes_count == 2 and i == 0):
                continue
            try:
                Printer.print("SHAP values PLOT " + str(i) + " for " + self.type + " " + self.filename)
                shap.summary_plot(shap_values[i], self.train, plot_type="dot", show=show, feature_names=self.get_beautified_column_names(), class_names=self.get_beautified_classes())
                #shap.save_html(file, self.make_filename("SHAP-Plot-" + str(i)) + ".html")
                plt.savefig(self.make_filename("SHAP-Plot-" + str(i)) + ".png", format='png', dpi=1000, bbox_inches='tight')
                plt.clf()
            except AssertionError:
                Printer.print("\n\nERROR at SHAP values PLOT " + str(i) + " for " + self.type + " " + self.filename + "\n\n")
                                
        return shap_values
      
    def make_filename(self, filename):
        return "results/" + self.type + " " + filename + "-" + self.filename
    
    def get_classes_count(self):
        return len(self.train_labels.value_counts())
    
    def get_beautified_column_names(self):
        return [BETTER_COLUMN_NAMES.get(i, i) for i in self.train.columns]
    
    def get_beautified_classes(self):
        classes = ["LEVE", "GRAVE"]
        if (self.get_classes_count() > 2):
            classes.append("ÓBITO")
        return classes
    
    def get_beautified_train_labels(self):
        return [self.get_beautified_classes()[i] for i in self.train_labels]    
    def get_beautified_test_labels(self):
        return [self.get_beautified_classes()[i] for i in self.test_labels]
              
    @staticmethod
    def run_classificator(clazz, data, name="", visualize=True, should_print=True):
        classificator = clazz(data.train, data.test, data.train_labels, data.test_labels, None, filename=name)
        classificator.train_model()
        classificator.test_model()
        if (should_print):
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
        
        Printer.print("Average Accuracy: " + str(sum(acc)/len(acc)))
        Printer.print("Max Accuracy: " + str(max(acc)))
        Printer.print("Min Accuracy: " + str(min(acc)))
        Printer.print("")
        
        best = max(classificators, key=lambda x: x.accuracy)
        Printer.print(best.accuracy)
        Printer.print(best.confusion_matrix)
        Printer.print(best.classification_report)
        
        best.visualize_model(best.make_filename("best_of_" + str(n) + "-" + best.accuracy + "acc"))
        return best

