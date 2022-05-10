# TODO Remove
import time
import pandas as pd

from dataset.base import DatasetBase
from column_names import RAW_DATE, RAW_FIRST_DOSE, RAW_SECOND_DOSE, RAW_UNIQUE_DOSE, RAW_REINFORCEMENT_DOSE, RAW_TOTAL, DATE, VACCINATION_PERCENTAGE

DATA_FOLDER = './dataset/data/'

VACCINATION_FILENAME = "pcr-vacinometro-05-05-2022.csv"

RAW_DATE = 'data_de_vacinacao'
RAW_FIRST_DOSE = 'dose_1'
RAW_SECOND_DOSE = 'dose_2'
RAW_UNIQUE_DOSE = 'dose_unica'
RAW_REINFORCEMENT_DOSE = 'dose_de_reforco'
RAW_TOTAL = 'total'

DATE = 'data_notificacao'
VACCINATION_PERCENTAGE = 'populacao_vacinada'
class DatasetVaccination(DatasetBase):
    
    RECIFE_POPULATION = 1661017 # Estimate in 2021
    
    def __init__(self):
        super().__init__()
        self.filename = VACCINATION_FILENAME
        self.delimiter = ","
        self.read_csv(DATA_FOLDER + self.filename)
        self.csv_to_df()
        self.remove_extra_columns()
        self.sort_by_date()
        self.count_on_day = {}
        self.percentage_on_day = {}
        self.count()
        
    def append_vaccination_percentage(self, cases):
        cases[VACCINATION_PERCENTAGE] = 0.0
        for date, percentage in self.percentage_on_day.items():
            cases.loc[cases[DATE] == date, VACCINATION_PERCENTAGE] = percentage
        return cases
        
    def remove_extra_columns(self):
        self.df.dropna(axis=1, how='all', inplace=True)
        self.df.drop([RAW_FIRST_DOSE, RAW_REINFORCEMENT_DOSE, RAW_TOTAL], axis=1, inplace=True)
        
    def sort_by_date(self):
        self.df.sort_values(by=RAW_DATE, inplace=True)
    
    def count(self):
        cumulative_count = 0
        for index, row in self.df.iterrows():
            day_count = row[RAW_SECOND_DOSE] + row[RAW_UNIQUE_DOSE]
            cumulative_count += day_count
            # Get the string date on DATE and format it to YYYY-MM-DD
            date = row[RAW_DATE]
            date = date.split('/')
            date = '20' + date[2] + '-' + date[1] + '-' + date[0]            
            self.count_on_day[date] = cumulative_count
        for key, value in self.count_on_day.items():
            self.percentage_on_day[key] = value / self.RECIFE_POPULATION
        
        
