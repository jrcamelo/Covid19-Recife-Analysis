import pandas as pd
import time

from dataset.base import DatasetBase
from dataset.cases_mild import DatasetMild
from dataset.cases_severe import DatasetSevere
from dataset.vaccination import DatasetVaccination

DATA_PATH = "./dataset/data/fulldata.csv"
class Dataset(DatasetBase):    
    def __init__(self, should_update_data=False):
        super().__init__()
        if (not should_update_data):
            self.load_data()
        if (self.df.empty):
            print("Loading from cases and vaccination data...")
            self.new_data()
        
    def load_data(self):
        try:
            self.df = pd.read_csv(DATA_PATH, sep=";")
            print(DATA_PATH + " loaded")
        except FileNotFoundError as e:
            print("No pre-processed data found")
            self.df = pd.DataFrame()      
        
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
        