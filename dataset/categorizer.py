
COL_AGE = "grupo_idade"
COL_GENDER = "sexo"
COL_SEVERITY = "severidade"
COL_DATE = "data_notificacao"

def categorize_booleans(df):
  for column in df.columns:
    if df[column].dtype == bool:
      df.loc[df[column] == True, column] = 1
      df.loc[df[column] == False, column] = 0
  
    
def categorize_gender(df):        
    df.loc[df[COL_GENDER] == "MASCULINO", COL_GENDER] = 0
    df.loc[df[COL_GENDER] == "FEMININO", COL_GENDER] = 1
    df.loc[df[COL_GENDER] == "INDEFINIDO", COL_GENDER] = 2
    df[COL_GENDER] = df[COL_GENDER].astype(int)
    return df

def categorize_age(df):
    df.loc[df[COL_AGE] == "0-9", COL_AGE] = 0
    df.loc[df[COL_AGE] == "10-19", COL_AGE] = 1
    df.loc[df[COL_AGE] == "20-29", COL_AGE] = 2
    df.loc[df[COL_AGE] == "30-39", COL_AGE] = 3
    df.loc[df[COL_AGE] == "40-49", COL_AGE] = 4
    df.loc[df[COL_AGE] == "50-59", COL_AGE] = 5
    df.loc[df[COL_AGE] == "60-69", COL_AGE] = 6
    df.loc[df[COL_AGE] == "70-79", COL_AGE] = 7
    df.loc[df[COL_AGE] == "80+", COL_AGE] = 8
    df[COL_AGE] = df[COL_AGE].astype(int)
    return df
    
def categorize_severity(df):
    df.loc[df[COL_SEVERITY] == "ASSINTOMATICO", COL_SEVERITY] = 5
    df.loc[df[COL_SEVERITY] == "LEVE", COL_SEVERITY] = 0
    df.loc[df[COL_SEVERITY] == "GRAVE", COL_SEVERITY] = 1
    df.loc[df[COL_SEVERITY] == "INTERNADO", COL_SEVERITY] = 2
    df.loc[df[COL_SEVERITY] == "OBITO", COL_SEVERITY] = 3
    df[COL_SEVERITY] = df[COL_SEVERITY].astype(int)
    return df