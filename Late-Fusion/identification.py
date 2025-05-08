import pandas as pd

# Load datasets
df_taiwan = pd.read_csv('Data_folder/Taiwan_en.csv')
df_usa = pd.read_csv('Data_folder/SEER_en.csv')

# Define features
features = ['PTHLTYPE_Histology', 'Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
            'Chemotherapy', 'Surgery', 'PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC', 'Income', 'Area', 'Race']


feature_similarity_score = {}

similarity_threshold = 0.6

global_features = []
taiwan_features = []
usa_features = []


# Calculate feature similarity score
for feature in features:
    similarity = 0
    
    if feature in df_taiwan.columns and feature in df_usa.columns:
        taiwan_unique = set(df_taiwan[feature].unique())  # Drop NaNs before getting unique values
        usa_unique = set(df_usa[feature].unique())

        union = taiwan_unique.union(usa_unique)
        intersection = taiwan_unique.intersection(usa_unique)
        similarity = len(intersection) / len(union)

        if similarity >= similarity_threshold: 
            global_features.append(feature)
        else: 
            taiwan_features.append(feature)
            usa_features.append(feature) 

    elif feature in df_taiwan.columns: 
        taiwan_features.append(feature)
    elif feature in df_usa.columns:
        usa_features.append(feature)

    print(f"{feature} : {similarity}")


print(f"Global Features: {global_features}")
print(f"Taiwanese Unique Features: {taiwan_features}")
print(f"SEER Unique Features: {usa_features}")


# One hot encoding global feature group 
global_features_one_hot = ['Laterality', 'Gender', 'SepNodule', 'PleuInva', 'PTHLTYPE_Histology']

df_taiwan_global_encoded = pd.get_dummies(df_taiwan[global_features_one_hot], drop_first=False, columns=global_features_one_hot)
df_usa_global_encoded = pd.get_dummies(df_usa[global_features_one_hot], drop_first=False, columns=global_features_one_hot)

global_features_en = set(df_taiwan_global_encoded).union(set(df_usa_global_encoded))

print(global_features_en)
print(len(global_features_en))

