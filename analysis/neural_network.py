from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

from analysis.base import AnalysisModel
from column_names import *

class NeuralNetwork(AnalysisModel):
    CONFIG = {
        "target": SEVERITY,
        "should_binary_severe": True,
        "should_categorize_age": False,
        "should_categorize_gender": True,
        "should_categorize_severity": True,
        "should_categorize_booleans": True,
        "should_normalize": True,
        "drop_diseases": False,
        "drop_symptoms": False,
        "drop": [DATE]
    }
    
    DEFAULT_MODEL_ARGS = {
        "input_layer_output_dim": 12,
        "2nd_layer_output_dim": 12,
        "output_layer_output_dim": 1,
        "init": "uniform",
        "activation": "relu",
        "output_activation": "sigmoid",
        "optimizer": "adam",
        "optimizer_loss": "binary_crossentropy",
        "optimizer_metrics": ["accuracy", "mse"],
        
    }
    
    def __init__(self, train, test, train_labels, test_labels, classes=None, filename="", model_config=None):
        super().__init__(train, test, train_labels, test_labels, classes, filename)
        self.cfg = model_config if model_config is not None else NeuralNetworkConfig(self.train, self.train_labels)
        self.model = Sequential()
        self.add_input_layer()
        self.add_second_layer()
        self.add_output_layer()
        self.compile_model()

    def add_input_layer(self):
        dense = Dense(self.cfg.input_layer_output_dim,
                      kernel_initializer=self.cfg.kernel_initializer,
                      activation=self.cfg.activation,
                      input_dim=self.cfg.input_columns)
        self.model.add(dense)
            
    def add_second_layer(self):
        dense = Dense(self.cfg.second_layer_output_dim,
                      kernel_initializer=self.cfg.kernel_initializer,
                      activation=self.cfg.activation)
        self.model.add(dense)
        
    def add_output_layer(self):
        dense = Dense(self.cfg.output_layer_output_dim,
                      kernel_initializer=self.cfg.kernel_initializer,
                      activation=self.cfg.output_activation)
        self.model.add(dense)
        
    def compile_model(self):
        self.model.compile(optimizer=self.cfg.optimizer,
                           loss=self.cfg.optimizer_loss,
                           metrics=self.cfg.optimizer_metrics)
        
    def train_model(self):
        self.model.fit(self.train, 
                       self.train_labels,
                       batch_size=self.cfg.batch_size,
                       epochs=self.cfg.epochs,
                       verbose=1)
        
    def test_model(self):
        self.run_model_score()
        self.predictions = self.model.predict(self.test)
        self.predictions = self.predictions > 0.5
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
        print("Accuracy: " + str(self.accuracy))
        print("Confusion Matrix:" + str(self.confusion_matrix))
        print("Classification Report: " + str(self.classification_report))
        if (verbose):
            print("Precision: " + self.precision)
            print("Recall: " + self.recall)
            print("F1: " + self.f1)
        print("")
        return self
        
    def run_model_score(self):
        self.model.evaluate(self.test, 
                            self.test_labels, 
                            verbose=0)



@dataclass
class NeuralNetworkConfig:
    """Class for configuring the Neural Network model"""
    
    input_layer_output_dim: int = 64
    second_layer_output_dim: int = 64
    output_layer_output_dim: int = 1
    kernel_initializer = "uniform"
    activation = "relu"
    output_activation = "sigmoid"
    optimizer = "adam"
    optimizer_metrics = ["accuracy", "mse", "mae"]
    batch_size = 32
    epochs = 5 # 100
    
    # Dynamic values
    input_columns = 1
    optimizer_loss = "binary_crossentropy"
    
    def __init__(self, data, target) -> None:
        self.input_columns = len(data.columns)
        print(target.value_counts())
        if (len(target.value_counts()) > 2):
            self.optimizer_loss = "categorical_crossentropy"
            self.output_layer_output_dim = len(target.value_counts()) - 1
