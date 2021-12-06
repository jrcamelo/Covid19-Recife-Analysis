from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from analysis.base import AnalysisModel
from column_names import *
from printer import Printer

class KNN(AnalysisModel):
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
    
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename="", k=7):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k,
                                          weights='distance',
                                          algorithm='auto',
                                          p=2,
                                          metric_params=None,
                                          n_jobs=-1)
        self.type = "KNN"
        
    def visualize_model(self, filename=None):
        if filename is None:
            filename = self.make_filename(str(self.accuracy) + "acc")
        Printer.print("Can't plot KNN yet")
        # plot_decision_regions(X=self.train.values,
        #                       y=self.train_labels.values,
        #                       clf=self.model,
        #                       legend=2,
        #                       colors=['green', 'yellow', 'red'],
        #                       markers=['m', 's', 'd'])
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.title('K-Nearest Neighbors')
        # plt.show()
        # plt.savefig(filename, format='png')
        return self
        
        
    