from dataclasses import dataclass
from keras.models import Sequential, model_from_json
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense
from sklearn import metrics

from analysis.base import AnalysisModel
from column_names import *
from printer import Printer

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
        self.type = "NeuralNetwork"
        # self.create_model()
        self.model = KerasClassifier(build_fn=self.create_model, epochs=self.cfg.epochs, batch_size=self.cfg.batch_size, verbose=0)

    def create_model(self):
        self.model = Sequential()
        self.add_input_layer()
        self.add_second_layer()
        self.compile_model()
        return self.model
    
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
    
    def run_model_score(self):
        self.model.evaluate(self.test, 
                            self.test_labels, 
                            verbose=0)
        # kfold = KFold(n_splits=10, shuffle=True)
        # self.results = cross_val_score(self.model, self.test, self.test_labels, cv=kfold, scoring="accuracy")
        
    def test_model(self):
        self.run_model_score()
        self.predictions = self.model.predict(self.test)
        if (self.cfg.optimizer_loss == "binary_crossentropy"):
            self.predictions = self.predictions > 0.5
        Printer.print(self.predictions)
        self.save()
        
        # print("Baseline: %.2f%% (%.2f%%)" % (self.results.mean()*100, self.results.std()*100))
        try:
            self.classification_report = metrics.classification_report(self.test_labels, self.predictions)
        except ValueError:
            self.classification_report = "No data"
            print("Could not generate classification report")
        try:
            self.confusion_matrix = metrics.confusion_matrix(self.test_labels, self.predictions)
        except ValueError:
            self.confusion_matrix = "No data"
            print("Could not generate confusion matrix")
        try:
            self.accuracy = metrics.accuracy_score(self.test_labels, self.predictions)
        except ValueError:
            self.accuracy = 0
            print("Could not generate accuracy score")
        try:
            self.precision = metrics.precision_score(self.test_labels, self.predictions)
        except ValueError:
            self.precision = 0
            print("Could not generate precision score")
        try:
            self.recall = metrics.recall_score(self.test_labels, self.predictions)
        except ValueError:
            self.recall = 0
            print("Could not generate recall score")
        try:
            self.f1 = metrics.f1_score(self.test_labels, self.predictions)
        except ValueError:
            self.f1 = 0
            print("Could not generate f1 score")            
        return self
    
    def print_results(self, verbose=False):
        Printer.print("Accuracy: " + str(self.accuracy))
        Printer.print("Confusion Matrix:" + str(self.confusion_matrix))
        Printer.print("Classification Report: " + str(self.classification_report))
        if (verbose):
            Printer.print("Precision: " + self.precision)
            Printer.print("Recall: " + self.recall)
            Printer.print("F1: " + self.f1)
        Printer.print("")
        return self
    
    def grid_search(self):
        # Create hyperparameter space
        epoch_values = [10, 25, 50, 100, 150, 200]
        batches = [10, 20, 30, 40, 50, 100, 1000]
        optimizers = ['rmsprop', 'adam', 'SGD']
        neurons = [16, 32, 64, 128, 256]
        lr_values = [0.001, 0.01, 0.1, 0.2, 0.3]

        # Create hyperparameter options
        hyperparameters = dict(batch_size=batches)

        grid = GridSearchCV(self.model, hyperparameters, cv=5)
        grid_result = grid.fit(self.train, self.train_labels)
        Printer.print("Best Parameters: " + str(grid_result.best_params_))
        Printer.print("Best Accuracy: " + str(grid_result.best_score_))
        return self
    
    def visualize_model(self, filename=None):
        return self
        

        
    def save(self):
        json = self.model.to_json()
        with open("NN-" + self.filename + ".json", "w") as json_file:
            json_file.write(json)
        self.model.save_weights("NN-" + self.filename + ".h5")
    
    def load(self, filename = None):
        if not filename:
            filename = "NN-" + self.filename
        json_file = open(filename + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(filename + ".h5")

        

@dataclass
class NeuralNetworkConfig:
    """Class for configuring the Neural Network model"""
    
    input_layer_output_dim: int = 64
    second_layer_output_dim: int = 64
    output_layer_output_dim: int = 3
    kernel_initializer = 'glorot_uniform'
    activation = 'relu'
    output_activation = 'softmax'
    optimizer = 'adam'
    optimizer_metrics = ["accuracy", "mse", "mae"]
    batch_size = 32
    epochs = 200 # 100
    
    # Dynamic values
    input_columns = 1
    optimizer_loss = "binary_crossentropy"
    
    def __init__(self, data, target) -> None:
        self.input_columns = len(data.columns)
        Printer.print(target.value_counts())
        self.output_layer_output_dim: int = 2
        if (len(target.value_counts()) > 2):
            self.output_layer_output_dim: int = 3
            self.optimizer_loss = "sparse_categorical_crossentropy"
            self.output_layer_output_dim = len(target.value_counts()) - 1
