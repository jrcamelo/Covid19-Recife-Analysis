import pandas as pd

from dataset.base import DatasetBase
from dataset.cases_mild import DatasetMild
from dataset.cases_severe import DatasetSevere
from dataset.vaccination import DatasetVaccination

class Dataset(DatasetBase):    
    def __init__(self):
        self.df = pd.DataFrame()
        self.mild = DatasetMild()
        self.severe = DatasetSevere()
        self.df = pd.concat([self.mild.df, self.severe.df])
        # self.vaccination = DatasetVaccination()