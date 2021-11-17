import pandas as pd

class DiseaseNormalizer:
    def __init__(self, original):
        self.original = original
        self.df = pd.DataFrame()
    
    def get_normalized_columns(self):
        return self.df