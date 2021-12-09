from numpy import testing
import pandas as pd
import numpy as np
import pdb
import time
import datetime
from sklearn.model_selection import GridSearchCV 


from data import Dataset
from analysis.base import AnalysisModel
from analysis.decision_tree import DecisionTree
from analysis.random_forest import RandomForest
from analysis.knn import KNN
from analysis.neural_network import NeuralNetwork, NeuralNetworkConfig
from analysis.xgboost import XGBoost
from analysis.gradient_boosting import GradientBoosting
from analysis.light_gradient_boosting import LightGradientBoosting
from column_names import *

from printer import Printer

Printer.print(time.perf_counter())
Printer.print(datetime.datetime.now().strftime("%m-%d-%H:%M:%S"))
data = Dataset(NeuralNetwork.CONFIG,
                should_update_data=False,
                should_binary_severe=False,
                target=SEVERITY,
                # undersample_amount=1,
                # oversample_amount=1,
                should_normalize=True,
                should_categorize_age=True,
                should_categorize_gender=True,
                should_categorize_severity=True,
                should_categorize_booleans=True,
                drop_diseases=False,
                drop_symptoms=False,
                drop=[DATE, DISEASE_OTHER])
Printer.print(time.perf_counter())


# XGBoost.run_classificator(XGBoost, data, "XGBoost", True).make_shap_values(True)
# GradientBoosting.run_classificator(GradientBoosting, data, "GradientBoosting", True).make_shap_values(True)
# AnalysisModel.run_classificator(LightGradientBoosting, data, "LightGradientBoosting", True).make_shap_values(True)



#data = Dataset(should_update_data=True)
#Printer.print(data.df.head())
#Printer.print(data.df.info())
#Printer.print(data.df.describe())
#Printer.print(data.count_uniques())
#Printer.print(data.df.infer_objects())

#Printer.print(data.df.dtypes)
#data.write_to_csv('test.csv')

#data.count_uniques()
#Printer.print(data.df.describe())
#Printer.print(data.df.head())  
#Printer.print(data.df.dtypes)

# run_multiple_decision_trees(data)
# run_multiple_random_forests(data)

#tree = run_random_forest(data)



# get_best_params(data)


# forest = RandomForest(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
# forest.train_model()
# forest.test_model()
# Printer.print(forest.accuracy)
# Printer.print(forest.confusion_matrix)
# Printer.print(forest.classification_report)

# #forest.visualize_model("new_params_no_severe2")
# forest.make_shap_values(0)

# run_multiple_random_forests(data)
# run_random_forest(data)


# param_grid = {
#     "criterion": ['gini', 'entropy'],
#     "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     "class_weight": ['balanced', None],
    
# }
# CV_rfc = GridSearchCV(estimator=forest.model, param_grid=param_grid, cv= 5)
# CV_rfc.fit(data.df, data.target)
# Printer.print(CV_rfc.best_params_)
# Printer.print(CV_rfc.best_estimator_)
# forest.make_shap_values("random_forest_" + str(forest.accuracy), data.df)

#run_knn(data)
#get_best_knn_params(data)





# tree.visualize_model("testrandomf")

# Printer.print(data.count_uniques())

Printer.print(time.perf_counter())

            # model = NeuralNetwork(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"])
            # Printer.print(time.perf_counter())
            # model.train_model()
            # Printer.print(time.perf_counter())
            # model.test_model()
            # Printer.print(time.perf_counter())
            # Printer.print(model.accuracy)
            # Printer.print(model.confusion_matrix)
            # Printer.print(model.classification_report)


# def knn(n):
#     model = KNN(data.train, data.test, data.train_labels, data.test_labels, ["LEVE", "GRAVE", "OBITO", "ASSINTOMATICO"], n)
#     model.train_model()
#     model.test_model()
#     Printer.print(model.accuracy)
#     Printer.print(model.confusion_matrix)
#     Printer.print(model.classification_report)
    
# knn(7)

# run_decision_tree(data)

#   data.plot_densities().plot_correlation()
# AnalysisModel.run_classificator(KNN, data).plot_densities()
# AnalysisModel.run_classificator(DecisionTree, data).make_shap_values()
# AnalysisModel.run_classificator(RandomForest, data).make_shap_values()








def test_with_filter(models, name, filter_column, filter_value, binary=False, drop_symptoms=False):
    data = Dataset(DecisionTree.CONFIG,
                should_update_data=False,
                should_binary_severe=binary,
                target=SEVERITY,
                should_normalize=True,
                should_categorize_age=True,
                should_categorize_gender=True,
                should_categorize_severity=True,
                should_categorize_booleans=True,
                drop_diseases=False,
                drop_symptoms=drop_symptoms,
                drop=[DATE, DISEASE_OTHER],
                filter_column=filter_column,
                filter_column_value=filter_value,
                filename=name)
        
    for model in models:
        Printer.print("\n\n" + str(model.__name__) + " " + str(filter_column) + " with " + str(filter_value) + " --- " + str(name))
        AnalysisModel.run_classificator(model, data, name, False).make_shap_values(False)

full = [RandomForest, DecisionTree, XGBoost, LightGradientBoosting, ]
binary = [RandomForest, DecisionTree, XGBoost, LightGradientBoosting, GradientBoosting]

# test_with_filter(full, "normal", None, None, binary=False, drop_symptoms=False)
# test_with_filter(full, "alguma-v", VACCINATION_ABOVE_0, True, binary=False, drop_symptoms=False)
# test_with_filter(full, "nenhuma-v", VACCINATION_ABOVE_0, False, binary=False, drop_symptoms=False)
# test_with_filter(full, "v-acima-30", VACCINATION_ABOVE_30, True, binary=False, drop_symptoms=False)
# test_with_filter(full, "v-acima-45", VACCINATION_ABOVE_45, True, binary=False, drop_symptoms=False)

# # test_with_filter(full, "sem-sintomas-normal", None, None, binary=False, drop_symptoms=True)
# # test_with_filter(full, "sem-sintomas-alguma-v", VACCINATION_ABOVE_0, True, binary=False, drop_symptoms=True)
# # test_with_filter(full, "sem-sintomas-nenhuma-v", VACCINATION_ABOVE_0, False, binary=False, drop_symptoms=True)
# # test_with_filter(full, "sem-sintomas-v-acima-30", VACCINATION_ABOVE_30, True, binary=False, drop_symptoms=True)
# # test_with_filter(full, "sem-sintomas-v-acima-45", VACCINATION_ABOVE_45, True, binary=False, drop_symptoms=True)

# test_with_filter(binary, "binary-normal", None, None, binary=True, drop_symptoms=False)
# test_with_filter(binary, "binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=False)
# test_with_filter(binary, "binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=False)
# test_with_filter(binary, "binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=False)
# test_with_filter(binary, "binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=False)

# test_with_filter(binary, "sem-sintomas-binary-normal", None, None, binary=True, drop_symptoms=True)
# test_with_filter(binary, "sem-sintomas-binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=True)
# test_with_filter(binary, "sem-sintomas-binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=True)
# test_with_filter(binary, "sem-sintomas-binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=True)
# test_with_filter(binary, "sem-sintomas-binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=True)





def test_metrics(models, name, filter_column, filter_value, binary=False, drop_symptoms=False):
    filter_column_str = filter_column
    filter_value_str = filter_value
    if (filter_column is None):
        filter_column_str = "NONE"
    if (filter_value is None):
        filter_value_str = "NONE"

    
    metrics = {}
    for model in models:
        key = str(model.__name__) + " " + str(filter_column_str) + " with " + str(filter_value_str) + " --- " + str(name)
    
        if (key not in metrics):
            metrics[key] = []
        for i in range(0, 10):
            data = Dataset(DecisionTree.CONFIG,
                            should_update_data=False,
                            should_binary_severe=binary,
                            target=SEVERITY,
                            should_normalize=True,
                            should_categorize_age=True,
                            should_categorize_gender=True,
                            should_categorize_severity=True,
                            should_categorize_booleans=True,
                            drop_diseases=False,
                            drop_symptoms=drop_symptoms,
                            drop=[DATE, DISEASE_OTHER],
                            filter_column=filter_column,
                            filter_column_value=filter_value,
                            filename=name)
            metrics[key].append(AnalysisModel.run_classificator(model, data, name, False, False))
            
    for key in metrics:
        try:
            Printer.print(key)
            Printer.print("\n\tAccuracy avg: " + str(np.mean([metric.accuracy for metric in metrics[key]])))
            Printer.print("\tAccuracy best: " + str(np.max([metric.accuracy for metric in metrics[key]])))
            Printer.print("\tAccuracy worst: " + str(np.min([metric.accuracy for metric in metrics[key]])))
            Printer.print("\tAccuracy std: " + str(np.std([metric.accuracy for metric in metrics[key]])))
            Printer.print("\n\tPrecision avg: " + str(np.mean([metric.precision for metric in metrics[key]])))
            Printer.print("\tPrecision best: " + str(np.max([metric.precision for metric in metrics[key]])))
            Printer.print("\tPrecision worst: " + str(np.min([metric.precision for metric in metrics[key]])))
            Printer.print("\tPrecision std: " + str(np.std([metric.precision for metric in metrics[key]])))
            Printer.print("\tPrecision Macro avg: " + str(np.mean([metric.precision_macro for metric in metrics[key]])))
            Printer.print("\tPrecision Macro best: " + str(np.max([metric.precision_macro for metric in metrics[key]])))
            Printer.print("\tPrecision Macro worst: " + str(np.min([metric.precision_macro for metric in metrics[key]])))
            Printer.print("\tPrecision Macro std: " + str(np.std([metric.precision_macro for metric in metrics[key]])))
            Printer.print("\n\tRecall avg: " + str(np.mean([metric.recall for metric in metrics[key]])))
            Printer.print("\tRecall best: " + str(np.max([metric.recall for metric in metrics[key]])))
            Printer.print("\tRecall worst: " + str(np.min([metric.recall for metric in metrics[key]])))
            Printer.print("\tRecall std: " + str(np.std([metric.recall for metric in metrics[key]])))
            Printer.print("\tRecall Macro avg: " + str(np.mean([metric.recall_macro for metric in metrics[key]])))
            Printer.print("\tRecall Macro best: " + str(np.max([metric.recall_macro for metric in metrics[key]])))
            Printer.print("\tRecall Macro worst: " + str(np.min([metric.recall_macro for metric in metrics[key]])))
            Printer.print("\tRecall Macro std: " + str(np.std([metric.recall_macro for metric in metrics[key]])))
            Printer.print("\n\tF1 avg: " + str(np.mean([metric.f1 for metric in metrics[key]])))
            Printer.print("\tF1 best: " + str(np.max([metric.f1 for metric in metrics[key]])))
            Printer.print("\tF1 worst: " + str(np.min([metric.f1 for metric in metrics[key]])))
            Printer.print("\tF1 std: " + str(np.std([metric.f1 for metric in metrics[key]])))
            Printer.print("\tF1 Macro avg: " + str(np.mean([metric.f1_macro for metric in metrics[key]])))
            Printer.print("\tF1 Macro best: " + str(np.max([metric.f1_macro for metric in metrics[key]])))
            Printer.print("\tF1 Macro worst: " + str(np.min([metric.f1_macro for metric in metrics[key]])))
            Printer.print("\tF1 Macro std: " + str(np.std([metric.f1_macro for metric in metrics[key]])))
            Printer.print("\n\tROC avg: " + str(np.mean([metric.roc_auc for metric in metrics[key]])))
            Printer.print("\tROC best: " + str(np.max([metric.roc_auc for metric in metrics[key]])))
            Printer.print("\tROC worst: " + str(np.min([metric.roc_auc for metric in metrics[key]])))
            Printer.print("\tROC std: " + str(np.std([metric.roc_auc for metric in metrics[key]])))
            
            best_f1 = np.argmax([metric.f1_macro for metric in metrics[key]])
            Printer.print("\n\tBest F1 macro: " + str(metrics[key][best_f1].f1_macro))
            Printer.print("\tAccuracy: " + str(metrics[key][best_f1].accuracy))
            Printer.print("\tPrecision: " + str(metrics[key][best_f1].precision))
            Printer.print("\tPrecision macro: " + str(metrics[key][best_f1].precision_macro))
            Printer.print("\tRecall: " + str(metrics[key][best_f1].recall))
            Printer.print("\tRecall macro: " + str(metrics[key][best_f1].recall_macro))
            Printer.print("\tF1: " + str(metrics[key][best_f1].f1))
            Printer.print("\tF1 macro: " + str(metrics[key][best_f1].f1_macro))
            Printer.print("\tROC: " + str(metrics[key][best_f1].roc_auc))
            Printer.print(metrics[key][best_f1].confusion_matrix)
            Printer.print(metrics[key][best_f1].classification_report)
            
        except Exception as e:
            Printer.print(e)
            Printer.print("\n\n")
        try:
            # best_i = np.argmax([metric.precision_macro for metric in metrics[key]])
            # metrics[key][best_i].visualize_model()
            # metrics[key][best_i].plot_roc_curve()
            pass
        except Exception as e:
            Printer.print(e)
            Printer.print("\n\n")
        Printer.print("\n\n\n")
    Printer.print("\n\n\n")


    


test_metrics(full, "normal", None, None, binary=False, drop_symptoms=False)
test_metrics(full, "alguma-v", VACCINATION_ABOVE_0, True, binary=False, drop_symptoms=False)
test_metrics(full, "nenhuma-v", VACCINATION_ABOVE_0, False, binary=False, drop_symptoms=False)
test_metrics(full, "v-acima-30", VACCINATION_ABOVE_30, True, binary=False, drop_symptoms=False)
test_metrics(full, "v-acima-45", VACCINATION_ABOVE_45, True, binary=False, drop_symptoms=False)

# test_metrics(full, "sem-sintomas-normal", None, None, binary=False, drop_symptoms=True)
# test_metrics(full, "sem-sintomas-alguma-v", VACCINATION_ABOVE_0, True, binary=False, drop_symptoms=True)
# test_metrics(full, "sem-sintomas-nenhuma-v", VACCINATION_ABOVE_0, False, binary=False, drop_symptoms=True)
# test_metrics(full, "sem-sintomas-v-acima-30", VACCINATION_ABOVE_30, True, binary=False, drop_symptoms=True)
# test_metrics(full, "sem-sintomas-v-acima-45", VACCINATION_ABOVE_45, True, binary=False, drop_symptoms=True)

test_metrics(binary, "binary-normal", None, None, binary=True, drop_symptoms=False)
test_metrics(binary, "binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=False)
test_metrics(binary, "binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=False)
test_metrics(binary, "binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=False)
test_metrics(binary, "binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=False)

test_metrics(binary, "sem-sintomas-binary-normal", None, None, binary=True, drop_symptoms=True)
test_metrics(binary, "sem-sintomas-binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=True)
test_metrics(binary, "sem-sintomas-binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=True)
test_metrics(binary, "sem-sintomas-binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=True)
test_metrics(binary, "sem-sintomas-binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=True)





def test_neural_with_filter(name, filter_column, filter_value, binary=False, drop_symptoms=False):
    data = Dataset(NeuralNetwork.CONFIG,
                should_update_data=False,
                should_binary_severe=binary,
                target=SEVERITY,
                should_normalize=True,
                should_categorize_age=True,
                should_categorize_gender=True,
                should_categorize_severity=True,
                should_categorize_booleans=True,
                drop_diseases=False,
                drop_symptoms=drop_symptoms,
                drop=[DATE],
                filter_column=filter_column,
                filter_column_value=filter_value,
                filename=name)
    if (filter_column is None):
        filter_column = "NONE"
    if (filter_value is None):
        filter_value = "NONE"
        
    Printer.print("\n\nNeural Network " + str(filter_column) + " with " + str(filter_value) + " --- " + str(name))
    AnalysisModel.run_classificator(NeuralNetwork, data, name, False)
    
# test_neural_with_filter("normal", None, None, binary=False, drop_symptoms=False)
# test_neural_with_filter("sem-sintomas-normal", None, None, binary=False, drop_symptoms=True)
# test_neural_with_filter("binary-normal", None, None, binary=True, drop_symptoms=False)
# test_neural_with_filter("sem-sintomas-binary-normal", None, None, binary=True, drop_symptoms=True)


def plot_dataset_graphs(name, filter_column, filter_value, binary=False, drop_symptoms=False):
       
        data = Dataset(NeuralNetwork.CONFIG,
                should_update_data=False,
                should_binary_severe=binary,
                target=SEVERITY,
                should_normalize=True,
                should_categorize_age=False,
                should_categorize_gender=True,
                should_categorize_severity=True,
                should_categorize_booleans=True,
                drop_diseases=True,#
                drop_symptoms=True, # 
                drop=[DATE],
                filter_column=filter_column,
                filter_column_value=filter_value,
                filename=name)
        # data.plot_correlation()
        # data.plot_densities()
        data.plot_pairplot()
        # Printer.print("\n\n" + str(filter_column) + " with " + str(filter_value) + " --- " + str(name))
        # data.print_severity_distribution()
        # Printer.print("\n\n")
        # data.print_column_densities_according_to_severity()
        # Printer.print("\n\n\n")
        
# plot_dataset_graphs("normal", None, None, binary=False, drop_symptoms=False)
# plot_dataset_graphs("alguma-v", VACCINATION_ABOVE_0, True, binary=False, drop_symptoms=False)
# plot_dataset_graphs("nenhuma-v", VACCINATION_ABOVE_0, False, binary=False, drop_symptoms=False)
# plot_dataset_graphs("v-acima-30", VACCINATION_ABOVE_30, True, binary=False, drop_symptoms=False)
# plot_dataset_graphs("v-acima-45", VACCINATION_ABOVE_45, True, binary=False, drop_symptoms=False)

# plot_dataset_graphs("binary-normal", None, None, binary=True, drop_symptoms=False)
# plot_dataset_graphs("binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=False)
# plot_dataset_graphs("binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=False)
# plot_dataset_graphs("binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=False)
# plot_dataset_graphs("binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=False)

# plot_dataset_graphs("sem-sintomas-binary-normal", None, None, binary=True, drop_symptoms=True)
# plot_dataset_graphs("sem-sintomas-binary-alguma-v", VACCINATION_ABOVE_0, True, binary=True, drop_symptoms=True)
# plot_dataset_graphs("sem-sintomas-binary-nenhuma-v", VACCINATION_ABOVE_0, False, binary=True, drop_symptoms=True)
# plot_dataset_graphs("sem-sintomas-binary-v-acima-30", VACCINATION_ABOVE_30, True, binary=True, drop_symptoms=True)
# plot_dataset_graphs("sem-sintomas-binary-v-acima-45", VACCINATION_ABOVE_45, True, binary=True, drop_symptoms=True)        

Printer.print("----------------------------------------------------\n\n\n\n")