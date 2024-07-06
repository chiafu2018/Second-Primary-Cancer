import pandas as pd

df = pd.read_csv('Taiwan.csv')

# Make SEER data have the same column name

column_name_mapping = {
    # 'old_name': 'new_name'
    'SYMOGN':'Laterality',
    'TMRSZ':'Tumorsz',
    'SSF1':'SepNodule',
    'SSF2':'PleuInva',
    'SSF4':'PleuEffu',
    'SSF6':'EGFR',
    'SSF7':'ALK',
    'RT':'Radiation',
    'ST':'Chemotherapy',
    'OP':'Surgery', 
    'AJCCstage': 'AJCC'
}


df.rename(columns=column_name_mapping, inplace=True)

df.to_csv('Taiwan_en.csv', index=False)
