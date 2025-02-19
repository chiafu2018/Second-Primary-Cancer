import pandas as pd

# Load CSV file into DataFrame
df = pd.read_csv('SEER_en.csv')

# Select the column and then get unique values
column_name = ["Histologic Type ICD-O-3", 'Target']

for col in column_name:
    unique_values = df[col].unique()
    print(f"{col}: {unique_values}\n")
