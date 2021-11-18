import pandas as pd

class DatasetBase:
    CHUNK_SIZE = 50000
  
    def __init__(self):
        self.csv = pd.DataFrame()
        self.df = pd.DataFrame()
        self.run_on_objects = []
        self.delimiter = ";"
            
    def read_csv(self, file):
        for chunk in pd.read_csv(file, chunksize=self.CHUNK_SIZE, delimiter=self.delimiter, low_memory=False):
            self.csv = pd.concat([self.csv, chunk])
    
    def csv_to_df(self):
        self.df = self.csv.copy()
    
    def rename_columns(self, old, new):      
        if (old == []):
            print("ERROR: No old column names defined")
            return
        if (len(old) != len(new)):
            print("ERROR: Old and new column names don't match in length")
            return
        column_rename = dict(zip(old, new))
        self.df.rename(columns=column_rename, inplace=True)        
    
    def create_blank_columns_if_not_exist(self, columns, blank_data):
        for column in columns:
            if (column not in self.df.columns):
                blank = blank_data.get(column, None)
                self.df[column] = blank
                
    def delete_columns(self, columns):
        for column in columns:
            if (column in self.df.columns):
                del self.df[column]
                
    def iterate_columns(self):
        for column in self.df.columns:
            if (self.df[column].dtype == 'object'):
                for method in self.run_on_objects:
                    method(column)
    
    def normalize_strings(self, column):
        self.df[column] = self.df[column].str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
         
    def replace_value_in_column(self, column, values, new_value):
        self.df.loc[self.df[column].isin(values), column] = new_value
    
    def count_uniques(self):
        for column in self.df.columns:
            print(self.df[column].value_counts())
            print("")
        
    def write_to_csv(self, file):
        self.df.to_csv(file, index=False, sep=";", encoding="utf-8")