from column_names import AGE, GENDER, SEVERITY

def categorize_booleans(df):
  for column in df.columns:
    if df[column].dtype == bool:
      df.loc[df[column] == True, column] = 1
      df.loc[df[column] == False, column] = 0
  
    
def categorize_gender(df):        
    df.loc[df[GENDER] == "MASCULINO", GENDER] = 0
    df.loc[df[GENDER] == "FEMININO", GENDER] = 1
    df.loc[df[GENDER] == "INDEFINIDO", GENDER] = 2
    df[GENDER] = df[GENDER].astype(int)
    return df

def categorize_age(df):
    df.loc[df[AGE] == "0-9", AGE] = 0
    df.loc[df[AGE] == "10-19", AGE] = 1
    df.loc[df[AGE] == "20-29", AGE] = 2
    df.loc[df[AGE] == "30-39", AGE] = 3
    df.loc[df[AGE] == "40-49", AGE] = 4
    df.loc[df[AGE] == "50-59", AGE] = 5
    df.loc[df[AGE] == "60-69", AGE] = 6
    df.loc[df[AGE] == "70-79", AGE] = 7
    df.loc[df[AGE] == "80+", AGE] = 8
    df[AGE] = df[AGE].astype(int)
    return df
    
def categorize_severity(df):
    df.loc[df[SEVERITY] == "ASSINTOMATICO", SEVERITY] = 5
    df.loc[df[SEVERITY] == "LEVE", SEVERITY] = 0
    df.loc[df[SEVERITY] == "GRAVE", SEVERITY] = 1
    df.loc[df[SEVERITY] == "INTERNADO", SEVERITY] = 2
    df.loc[df[SEVERITY] == "OBITO", SEVERITY] = 3
    df[SEVERITY] = df[SEVERITY].astype(int)
    return df