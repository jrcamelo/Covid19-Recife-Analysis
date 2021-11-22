import pandas as pd
import time
from column_names import AGE, SEVERITY
from sklearn.model_selection import train_test_split

from dataset.base import DatasetBase
from dataset.cases import TRAIN_TO_TEST_RATIO
from dataset.cases_mild import DatasetMild
from dataset.cases_severe import DatasetSevere
from dataset.vaccination import DatasetVaccination
from dataset.categorizer import categorize_age, categorize_booleans, categorize_gender, categorize_severity, categorize_vaccination

DATA_PATH = "./dataset/data/fulldata.csv"
TRAIN_TO_TEST_RATIO = 0.8
class Dataset(DatasetBase):
    def __init__(self,
                 config=None,
                 should_update_data=False, 
                 target=None,
                 should_categorize_gender=None, 
                 should_categorize_age=None, 
                 should_categorize_severity=None,
                 should_categorize_booleans=None,
                 should_categorize_vaccination=None,
                 drop=None,
                 drop_symptoms=None,
                 drop_diseases=None):
        super().__init__()
        if (config != None):
            if (target == None):
                target = config.get("target", SEVERITY)
            if (should_categorize_gender == None):
                should_categorize_gender = config.get("should_categorize_gender", True)
            if (should_categorize_age == None):
                should_categorize_age = config.get("should_categorize_age", True)
            if (should_categorize_severity == None):
                should_categorize_severity = config.get("should_categorize_severity", True)
            if (should_categorize_booleans == None):
                should_categorize_booleans = config.get("should_categorize_booleans", True)
            if (should_categorize_vaccination == None):
                should_categorize_vaccination = config.get("should_categorize_vaccination", True)
            if (drop_symptoms == None):
                drop_symptoms = config.get("drop_symptoms", True)
            if (drop_diseases == None):
                drop_diseases = config.get("drop_diseases", False)
            if (drop == None):
                drop = config.get("drop", [AGE])
            
                
        self.load(should_update_data)
        self.categorize(should_categorize_gender, should_categorize_age, should_categorize_severity, should_categorize_booleans, should_categorize_vaccination)
        self.set_target(target)
        self.do_drop(drop, drop_symptoms, drop_diseases)
        self.split_data()        
        
    def categorize(self, gender, age, severity, booleans, vaccination):
        if (gender):
            categorize_gender(self.df)
        if (age):
            categorize_age(self.df)
        if (severity):
            categorize_severity(self.df)
        if (booleans):
            categorize_booleans(self.df)
        if (vaccination):
            categorize_vaccination(self.df)
            
    def set_target(self, target):
        self.target_column = target
        self.target = self.df[self.target_column]
        self.df.drop(self.target_column, inplace=True, axis=1)
            
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
            self.target,
            train_size=TRAIN_TO_TEST_RATIO,
            stratify=self.target)        


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
        