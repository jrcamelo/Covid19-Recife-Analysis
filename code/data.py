import pandas as pd
import time
from column_names import AGE, SEVERITY, VACCINATION_PERCENTAGE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

from dataset.base import DatasetBase
from dataset.cases import TRAIN_TO_TEST_RATIO
from dataset.cases_mild import DatasetMild
from dataset.cases_severe import DatasetSevere
from dataset.vaccination import DatasetVaccination
from dataset.categorizer import categorize_age, categorize_booleans, categorize_gender, categorize_severity, make_binary_mild_severe, make_binary_death
from dataset.normalizer import vaccination_thermometer_normalization, normalize_min_max, one_hot_encoding

from printer import Printer

DATA_PATH = "./dataset/data/fulldata.csv"
TRAIN_TO_TEST_RATIO = 0.8
class Dataset(DatasetBase):
    def __init__(self,
                 config=None,
                 should_update_data=True,
                 should_binary_severe=False,
                 should_death_only=False,
                 target=None,
                 oversample_amount=None,
                 undersample_amount=None,
                 should_categorize_gender=None, 
                 should_categorize_age=None, 
                 should_categorize_severity=None,
                 should_categorize_booleans=None,
                 should_normalize=None,
                 drop=None,
                 drop_symptoms=None,
                 drop_diseases=None,
                 filter_column=None,
                 filter_column_value=None,
                 filename=""):
        super().__init__()
        self.should_update_data = should_update_data
        self.should_binary_severe = should_binary_severe
        self.should_categorize_gender = should_categorize_gender
        self.should_categorize_age = should_categorize_age
        self.should_categorize_severity = should_categorize_severity
        self.should_death_only = should_death_only
        self.should_categorize_booleans = should_categorize_booleans
        self.should_normalize = should_normalize
        self.drop_symptoms = drop_symptoms
        self.drop_diseases = drop_diseases
        self.filename = filename
        
        if (config != None):
            if (target == None):
                target = config.get("target", SEVERITY)
            if (undersample_amount == None):
                undersample_amount = config.get("undersample_amount", 0)
            if (oversample_amount == None):
                oversample_amount = config.get("oversample_amount", 0)
            if (self.should_categorize_gender == None):
                self.should_categorize_gender = config.get("should_categorize_gender", True)
            if (self.should_categorize_age == None):
                self.should_categorize_age = config.get("should_categorize_age", True)
            if (self.should_categorize_severity == None):
                self.should_categorize_severity = config.get("should_categorize_severity", True)
            if (self.should_categorize_booleans == None):
                self.should_categorize_booleans = config.get("should_categorize_booleans", True)
            if (self.should_normalize == None):
                self.should_normalize = config.get("should_normalize", True)
            if (drop_symptoms == None):
                drop_symptoms = config.get("drop_symptoms", True)
            if (drop_diseases == None):
                drop_diseases = config.get("drop_diseases", False)
            if (drop == None):
                drop = config.get("drop", [AGE])
        self.target_column = target
                
        self.load(should_update_data)
       
        # Drops requested columns
        self.do_drop(drop, drop_symptoms, drop_diseases)
        
        self.categorize(self.should_categorize_gender, self.should_categorize_age, self.should_categorize_severity, self.should_categorize_booleans)
        if (self.should_normalize):
            self.normalize_columns(should_categorize_booleans)
        if (self.should_binary_severe):
            make_binary_mild_severe(self.df)
        elif (self.should_death_only):
            make_binary_death(self.df)
        
        # Remove data according to filters
        self.filter_data(filter_column, filter_column_value)
        
        # Split into train and test
        self.split_data()
        
        if (undersample_amount > 0):
            self.undersample(undersample_amount)
        elif (oversample_amount > 0):
            self.oversample(oversample_amount)
        
        self.df_with_target = self.df.copy()
        self.df.drop(self.target_column, inplace=True, axis=1)
        self.train.drop(self.target_column, inplace=True, axis=1)
        self.test.drop(self.target_column, inplace=True, axis=1)
        
    def categorize(self, gender, age, severity, booleans):
        if (gender):
            categorize_gender(self.df)
        if (age):
            categorize_age(self.df)
        if (severity):
            categorize_severity(self.df)
        if (booleans):
            categorize_booleans(self.df)
            
    def normalize_columns(self, should_categorize_booleans):
        Printer.print("Normalizing columns")
        for column in self.df.columns:
            if (column == VACCINATION_PERCENTAGE):
                vaccination_thermometer_normalization(self.df, should_categorize_booleans)                
            elif (column != self.target_column and column != SEVERITY):
                normalize_min_max(self.df, column)
                
    def undersample(self, amount):
        Printer.print("Undersampling: " + str(amount))
        count_not_mild = self.train[self.train[SEVERITY] != 0].shape[0]
        rus = RandomUnderSampler(sampling_strategy={0: amount * count_not_mild})
        self.train, self.train_labels = rus.fit_resample(self.train, self.train_labels)
        count_not_mild = self.test[self.test[SEVERITY] != 0].shape[0]
        rus = RandomUnderSampler(sampling_strategy={0: amount * count_not_mild})
        self.test, self.test_labels = rus.fit_resample(self.test, self.test_labels)
    
    def oversample(self, amount):
        Printer.print("Oversampling: " + str(amount))
        Printer.print(self.train_labels.value_counts())
        oversample = SMOTE()
        self.train, self.train_labels = oversample.fit_resample(self.train, self.train_labels)
        self.test, self.test_labels = oversample.fit_resample(self.test, self.test_labels)
            
    def set_target(self, target):
        self.target_column = target
        self.target = self.df[self.target_column]
        # self.df.drop(self.target_column, inplace=True, axis=1)
        self.df.target_name = target
            
    def do_drop(self, drop, drop_symptoms, drop_diseases):
        if (drop == None):
            drop = []
        if (drop_symptoms):
            drop += [col for col in self.df.columns if col.startswith("sintoma_")]
        if (drop_diseases):
            drop += [col for col in self.df.columns if col.startswith("doenca_")]
        Printer.print("Dropping: " + ','.join(drop))
        self.df.drop(drop, inplace=True, axis=1)
        
    def filter_data(self, column, value):
        # Keep only data where the column has the specified value
        if (column != None and value != None):
            self.df = self.df[self.df[column] == value]        
        
    def split_data(self):
        # Split data into train and test while stratifying by severity
        self.train, self.test, self.train_labels, self.test_labels = train_test_split(
            self.df,
            self.df[SEVERITY],
            test_size=1-TRAIN_TO_TEST_RATIO,
            train_size=TRAIN_TO_TEST_RATIO,
            stratify=self.df[SEVERITY])


    def load(self, should_update_data):
        if (not should_update_data):
            try:
                self.df = pd.read_csv(DATA_PATH, sep=";")
                # Printer.print(DATA_PATH + " loaded")
            except FileNotFoundError as e:
                Printer.print("No pre-processed data found")
                self.df = pd.DataFrame()   
        if (self.df.empty):
            Printer.print("Loading from cases and vaccination data...")
            self.new_data()
        
    def new_data(self):
        start = time.perf_counter()
        self.df = pd.DataFrame()
        Printer.print(str(time.perf_counter()) + ": Reading mild cases...")
        self.mild = DatasetMild()
        Printer.print(str(time.perf_counter()) + ": Reading severe cases...")
        self.severe = DatasetSevere()
        self.df = pd.concat([self.mild.df, self.severe.df])
        Printer.print(str(time.perf_counter()) + ": Reading vaccination data...")
        self.vaccination = DatasetVaccination()
        Printer.print(str(time.perf_counter()) + ": Appending vaccination percentage...")
        self.df = self.vaccination.append_vaccination_percentage(self.df)
        Printer.print(str(time.perf_counter()) + ": Done with " + str(time.perf_counter() - start))
        self.df.to_csv("./dataset/data/fulldata.csv", sep=";", index=False)
        
    def print_percentages(self):
        for column in self.df_with_target.columns:
            self.print_column_percentage_count(column)            
    
    def print_column_percentage_count(self, column):
        Printer.print(column)
        for value in self.df_with_target[column].unique():
            count = self.df_with_target[self.df_with_target[column] == value].shape[0]
            percentage = count / self.df_with_target.shape[0]
            Printer.print(str(value) + ": " + str(count) + " (" + str(percentage) + ")")
        Printer.print("")
        
    def plot_correlation(self, filename=""):
        rcParams['figure.figsize'] = 22, 22
        fig = plt.figure()
        sns.heatmap(self.df_with_target.corr(), annot=True, fmt=".1f")
        # plt.show()
        fig.savefig(self.filename + filename + 'correlation.png')
        return self
        
    def plot_densities(self, filename=""):
        outcome_0 = self.df[self.df_with_target[SEVERITY] == 0]
        outcome_1 = self.df[self.df_with_target[SEVERITY] == 1]
        outcome_2 = self.df[self.df_with_target[SEVERITY] == 2]
        count = len(self.df.columns)
        rcParams['figure.figsize'] = 15, count * 2
        fig, axs = plt.subplots(count, 1)
        fig.suptitle('Features densities for different severities')
        plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95,
                            wspace = 0.2, hspace = 0.9)
        for column_name in self.train.columns:
            try:             
                ax = axs[self.train.columns.get_loc(column_name) - 1]
                ax.set_xlabel('Valores de ' + column_name)
                ax.set_title(column_name)
                if (outcome_0[column_name].std() != 0):
                    outcome_0[column_name].plot(kind='density', 
                                                ax=ax, 
                                                subplots=True, 
                                                sharex=False, 
                                                color="green", 
                                                legend=True,
                                                label='Casos LEVES')
                if (outcome_1[column_name].std() != 0):
                    outcome_1[column_name].plot(kind='density', 
                                                ax=ax, 
                                                subplots=True, 
                                                sharex=False, 
                                                color="red", 
                                                legend=True,
                                                label='Casos GRAVES')
                if (outcome_2[column_name].std() != 0):
                    outcome_2[column_name].plot(kind='density', 
                                                ax=ax, 
                                                subplots=True, 
                                                sharex=False, 
                                                color="black", 
                                                legend=True,
                                                label='ÓBITOS')
            except Exception as e:
                Printer.print(e)

            ax.grid('on')
        # plt.show()
        fig.savefig(self.filename + filename + 'column-densities.png')
        return self
    
    def plot_pairplot(self, filename=""):
        rcParams['figure.figsize'] = 15, 15
        fig = plt.figure()
        sns.pairplot(self.df_with_target, hue=SEVERITY)
        # plt.show()
        fig.savefig(self.filename + filename + 'pairplot.png')
        return self
    
    def print_severity_distribution(self):
        # Print counts and percentages of each severity
        Printer.print("MILD: " + str(len(self.df_with_target[SEVERITY] == 0)) + " (" + str(len(self.df_with_target[SEVERITY] == 0) / len(self.df_with_target) * 100) + "%)")
        Printer.print("SEVERE: " + str(len(self.df_with_target[SEVERITY] == 1)) + " (" + str(len(self.df_with_target[SEVERITY] == 1) / len(self.df_with_target) * 100) + "%)")
        Printer.print("DEATH: " + str(len(self.df_with_target[SEVERITY] == 2)) + " (" + str(len(self.df_with_target[SEVERITY] == 2) / len(self.df_with_target) * 100) + "%)")
        return self
        
    def print_column_densities_according_to_severity(self):
        for column_name in self.train.columns:
            count = len(self.train[column_name].unique())
            if (count > 2):
                continue
            # Count how many rows have label as 0, 1 and 2
            count_0 = len(self.df_with_target[self.df_with_target[SEVERITY] == 0][column_name])
            count_1 = len(self.df_with_target[self.df_with_target[SEVERITY] == 1][column_name])
            count_2 = len(self.df_with_target[self.df_with_target[SEVERITY] == 2][column_name])
            count_column_0 = len(self.df_with_target.loc[(self.df_with_target[SEVERITY] == 0) & (self.df_with_target[column_name] == 1)])
            count_column_1 = len(self.df_with_target.loc[(self.df_with_target[SEVERITY] == 1) & (self.df_with_target[column_name] == 1)])
            if (count_2 > 0):
                count_column_2 = len(self.df_with_target.loc[(self.df_with_target[SEVERITY] == 2) & (self.df_with_target[column_name] == 1)])
            
            percentage_column_0 = count_column_0 / count_0
            percentage_column_1 = count_column_1 / count_1
            percentage_column_2 = -1
            if (count_2 > 0):
                percentage_column_2 = count_column_2 / count_2
                
            # Print counts and percentages as table
            Printer.print("\n" + column_name + ":")
            Printer.print("LEVES: " + str(count_column_0) + " / " + str(count_0) + " (" + str(percentage_column_0) + ")")
            Printer.print("GRAVES: " + str(count_column_1) + " / " + str(count_1) + " (" + str(percentage_column_1) + ")")
            if (count_2 > 0):
                Printer.print("ÓBITOS: " + str(count_column_2) + " / " + str(count_2) + " (" + str(percentage_column_2) + ")")
            
        return self
        