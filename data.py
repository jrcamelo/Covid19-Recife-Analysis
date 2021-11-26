import pandas as pd
import time
from column_names import AGE, SEVERITY, VACCINATION_PERCENTAGE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from dataset.base import DatasetBase
from dataset.cases import TRAIN_TO_TEST_RATIO
from dataset.cases_mild import DatasetMild
from dataset.cases_severe import DatasetSevere
from dataset.vaccination import DatasetVaccination
from dataset.categorizer import categorize_age, categorize_booleans, categorize_gender, categorize_severity, make_binary_mild_severe
from dataset.normalizer import vaccination_thermometer_normalization, normalize_min_max

DATA_PATH = "./dataset/data/fulldata.csv"
TRAIN_TO_TEST_RATIO = 0.8
class Dataset(DatasetBase):
    def __init__(self,
                 config=None,
                 should_update_data=False,
                 should_binary_severe=False,
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
                 drop_diseases=None):
        super().__init__()
        if (config != None):
            if (target == None):
                target = config.get("target", SEVERITY)
            if (undersample_amount == None):
                undersample_amount = config.get("undersample_amount", 0)
            if (oversample_amount == None):
                oversample_amount = config.get("oversample_amount", 0)
            if (should_categorize_gender == None):
                should_categorize_gender = config.get("should_categorize_gender", True)
            if (should_categorize_age == None):
                should_categorize_age = config.get("should_categorize_age", True)
            if (should_categorize_severity == None):
                should_categorize_severity = config.get("should_categorize_severity", True)
            if (should_categorize_booleans == None):
                should_categorize_booleans = config.get("should_categorize_booleans", True)
            if (should_normalize == None):
                should_normalize = config.get("should_normalize", True)
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
        
        self.categorize(should_categorize_gender, should_categorize_age, should_categorize_severity, should_categorize_booleans)
        if (should_normalize):
            self.normalize_columns(should_categorize_booleans)
        if (should_binary_severe):
            make_binary_mild_severe(self.df)
        
        # Split into train and test
        self.split_data()
        
        if (undersample_amount > 0):
            self.undersample(undersample_amount)
        elif (oversample_amount > 0):
            self.oversample(oversample_amount)
        
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
        print("Normalizing columns")
        for column in self.df.columns:
            if (column == VACCINATION_PERCENTAGE):
                vaccination_thermometer_normalization(self.df, should_categorize_booleans)                
            elif (column != self.target_column and column != SEVERITY):
                normalize_min_max(self.df, column)
                
                            
    def undersample(self, amount):
        print("Undersampling: " + str(amount))
        count_not_mild = self.train[self.train[SEVERITY] != 0].shape[0]
        rus = RandomUnderSampler(sampling_strategy={0: amount * count_not_mild})
        self.train, self.train_labels = rus.fit_resample(self.train, self.train_labels)
        print(self.train_labels.value_counts())
    
    def oversample(self, amount):
        print("Oversampling: " + str(amount))
        count_mild = self.train[self.train[SEVERITY] == 0].shape[0]
        oversample_amount = amount * count_mild
        uniques = self.train_labels.unique()
        oversample_hash = {x: oversample_amount for x in uniques}
        ros = RandomOverSampler(sampling_strategy=oversample_hash)
        self.train, self.train_labels = ros.fit_resample(self.train, self.train_labels)
        print(self.train_labels.value_counts())
            
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
        print("Dropping: " + ','.join(drop))
        self.df.drop(drop, inplace=True, axis=1)
        
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
                print(DATA_PATH + " loaded")
            except FileNotFoundError as e:
                print("No pre-processed data found")
                self.df = pd.DataFrame()   
        if (self.df.empty):
            print("Loading from cases and vaccination data...")
            self.new_data()
        
    def new_data(self):
        start = time.perf_counter()
        self.df = pd.DataFrame()
        print(str(time.perf_counter()) + ": Reading mild cases...")
        self.mild = DatasetMild()
        print(str(time.perf_counter()) + ": Reading severe cases...")
        self.severe = DatasetSevere()
        self.df = pd.concat([self.mild.df, self.severe.df])
        print(str(time.perf_counter()) + ": Reading vaccination data...")
        self.vaccination = DatasetVaccination()
        print(str(time.perf_counter()) + ": Appending vaccination percentage...")
        self.df = self.vaccination.append_vaccination_percentage(self.df)
        print(str(time.perf_counter()) + ": Done with " + str(time.perf_counter() - start))
        self.df.to_csv("./dataset/data/fulldata.csv", sep=";", index=False)
        