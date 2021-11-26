import numpy as np

from column_names import *
from dataset.categorizer import categorize_booleans

def vaccination_thermometer_normalization(df, should_categorize_booleans):
    df[VACCINATION_ABOVE_0] = False
    df[VACCINATION_ABOVE_15] = False
    df[VACCINATION_ABOVE_30] = False
    df[VACCINATION_ABOVE_45] = False
    df[VACCINATION_ABOVE_60] = False
    df[VACCINATION_ABOVE_75] = False
    df[VACCINATION_ABOVE_90] = False
    
    vaccination_15_percent = 0.15
    vaccination_30_percent = 0.30
    vaccination_45_percent = 0.45
    vaccination_60_percent = 0.60
    vaccination_75_percent = 0.75
    vaccination_90_percent = 0.90
    df[VACCINATION_ABOVE_90] = df[VACCINATION_PERCENTAGE] > vaccination_90_percent
    df[VACCINATION_ABOVE_75] = df[VACCINATION_PERCENTAGE] > vaccination_75_percent
    df[VACCINATION_ABOVE_60] = df[VACCINATION_PERCENTAGE] > vaccination_60_percent
    df[VACCINATION_ABOVE_45] = df[VACCINATION_PERCENTAGE] > vaccination_45_percent
    df[VACCINATION_ABOVE_30] = df[VACCINATION_PERCENTAGE] > vaccination_30_percent
    df[VACCINATION_ABOVE_15] = df[VACCINATION_PERCENTAGE] > vaccination_15_percent
    df[VACCINATION_ABOVE_0] = df[VACCINATION_PERCENTAGE] > 0
    
    df.drop(VACCINATION_PERCENTAGE, inplace=True, axis=1)    
    if (should_categorize_booleans):
        categorize_booleans(df)

def normalize_min_max(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())