from column_names import AGE, GENDER, SEVERITY, VACCINATION_PERCENTAGE

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
    df.loc[df[AGE] < 10, AGE] = 0
    df.loc[(df[AGE] >= 10) & (df[AGE] < 20), AGE] = 1
    df.loc[(df[AGE] >= 20) & (df[AGE] < 30), AGE] = 2
    df.loc[(df[AGE] >= 30) & (df[AGE] < 40), AGE] = 3
    df.loc[(df[AGE] >= 40) & (df[AGE] < 50), AGE] = 4
    df.loc[(df[AGE] >= 50) & (df[AGE] < 60), AGE] = 5
    df.loc[(df[AGE] >= 60) & (df[AGE] < 70), AGE] = 6
    df.loc[(df[AGE] >= 70) & (df[AGE] < 80), AGE] = 7
    df.loc[df[AGE] >= 80, AGE] = 8
    df[AGE] = df[AGE].astype(int)
    return df
  
def categorize_vaccination(df):
    df.loc[df[VACCINATION_PERCENTAGE] == 0, VACCINATION_PERCENTAGE] = 0
    df.loc[(df[VACCINATION_PERCENTAGE] > 0) & (df[VACCINATION_PERCENTAGE] < 0.1), VACCINATION_PERCENTAGE] = 1
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.1) & (df[VACCINATION_PERCENTAGE] < 0.2), VACCINATION_PERCENTAGE] = 2
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.2) & (df[VACCINATION_PERCENTAGE] < 0.3), VACCINATION_PERCENTAGE] = 3
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.3) & (df[VACCINATION_PERCENTAGE] < 0.4), VACCINATION_PERCENTAGE] = 4
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.4) & (df[VACCINATION_PERCENTAGE] < 0.5), VACCINATION_PERCENTAGE] = 5
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.5) & (df[VACCINATION_PERCENTAGE] < 0.6), VACCINATION_PERCENTAGE] = 6
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.6) & (df[VACCINATION_PERCENTAGE] < 0.7), VACCINATION_PERCENTAGE] = 7
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.7) & (df[VACCINATION_PERCENTAGE] < 0.8), VACCINATION_PERCENTAGE] = 8
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.8) & (df[VACCINATION_PERCENTAGE] < 0.9), VACCINATION_PERCENTAGE] = 9
    df.loc[(df[VACCINATION_PERCENTAGE] >= 0.9) & (df[VACCINATION_PERCENTAGE] < 0.99999), VACCINATION_PERCENTAGE] = 10
    df[VACCINATION_PERCENTAGE] = df[VACCINATION_PERCENTAGE].astype(int)
    return df

def categorize_severity(df):
    df.loc[df[SEVERITY] == "ASSINTOMATICO", SEVERITY] = 5
    df.loc[df[SEVERITY] == "LEVE", SEVERITY] = 0
    df.loc[df[SEVERITY] == "GRAVE", SEVERITY] = 1
    df.loc[df[SEVERITY] == "INTERNADO", SEVERITY] = 2
    df.loc[df[SEVERITY] == "OBITO", SEVERITY] = 3
    df[SEVERITY] = df[SEVERITY].astype(int)
    return df