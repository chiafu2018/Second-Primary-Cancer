import pandas as pd

s = set()
df = pd.read_csv('SEER_en.csv')

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                    'Chemotherapy', 'Surgery']

df = df[global_feature]
encode = pd.get_dummies(df, drop_first=False, columns=global_feature)
s.update(encode.columns)

df = pd.read_csv('Taiwan_en.csv')

df = df[global_feature]
new = pd.get_dummies(df, drop_first=False, columns=global_feature)
s.update(new.columns)

print(f"[{s}]")
print(len(s))