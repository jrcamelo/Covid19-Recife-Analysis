import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from column_names import SEVERITY, AGE_GROUP, DATE

class DecisionTree:
    CONFIG = {
        "target": SEVERITY,
        "should_categorize_age": True,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "drop_diseases": False,
        "drop_symptoms": False,
        "drop": [AGE_GROUP, DATE]
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes, max_depth=None):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.classes = classes
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)
        
    def train_model(self):
        self.model.fit(self.train, self.train_labels)
        
    def test_model(self):
        self.predictions = self.model.predict(self.test)
        self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        
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
        