from column_names import AGE, GENDER, SEVERITY, VACCINATION_PERCENTAGE

def categorize_booleans(df):
    for column in df.columns:
        if df[column].dtype == bool:
            df.loc[df[column] == True, column] = 1
            df.loc[df[column] == False, column] = 0
            df[column] = df[column].astype(int)
      
    
def categorize_gender(df):        
    df.loc[df[GENDER] != "MASCULINO", GENDER] = 1
    df.loc[df[GENDER] == "MASCULINO", GENDER] = 0
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

def categorize_severity(df):
    df.loc[df[SEVERITY] == "LEVE", SEVERITY] = 0
    df.loc[df[SEVERITY] == "GRAVE", SEVERITY] = 1
    df.loc[df[SEVERITY] == "OBITO", SEVERITY] = 2
    # df.loc[df[SEVERITY] == "INTERNADO", SEVERITY] = 3
    # df.loc[df[SEVERITY] == "ASSINTOMATICO", SEVERITY] = 5
    df[SEVERITY] = df[SEVERITY].astype(int)
    return df

def make_binary_mild_severe(df):
    df.loc[df[SEVERITY] != 0, SEVERITY] = 1
    df[SEVERITY] = df[SEVERITY].astype(int)
