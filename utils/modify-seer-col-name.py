import pandas as pd

df = pd.read_csv('SEER.csv')

# Let SEER data have the same encoding as Taiwan data

def encode_age_range(age_range):
    if '-' in age_range:
        lower, upper = age_range.split('-')
        lower = int(lower)
        upper = int(upper.split()[0])  # Remove 'years' and convert to int
    elif '+' in age_range:
        lower = int(age_range.split('+')[0])
        upper = float('inf')
    if lower >= 20 and upper <= 30:
        return 2
    elif lower >= 30 and upper <= 40:
        return 3
    elif lower >= 40 and upper <= 50:
        return 4
    elif lower >= 50 and upper <= 60:
        return 5
    elif lower >= 60 and upper <= 70:
        return 6
    elif lower >= 70 and upper <= 80:
        return 7
    elif lower >= 81 and upper <= 90:
        return 8
    else:
        return 9
   
def encode_SSF2_range(element):
    if element == 'Not documented; No resection of primary; Not assessed or unknown if assessed':
        return 9
    elif element == 'Blank(s)':
        return 9
    elif element == 'PL1 or PL2; Invasion of visceral pleura present, NOS':
        return 2
    elif element == 'Tumor extends to pleura, NOS; not stated if visceral or parietal':
        return 2
    elif element == 'PL3; Tumor invades into or through the parietal pleura OR chest wall':
        return 2
    elif element == 'PL0; No evidence; Tumor does not completely traverse the elastic layer of pleura':
        return 1
    else:
        return None

def encode_ajcc_stage(element):
    if element == 'IA' or element == 'IB':
        return 1
    elif element == 'IIA' or element == 'IIB':
        return 2
    elif element == 'IIIA' or element == 'IIIB':
        return 3
    elif element == 'IV':
        return 4
    else:
        return 9

def encode_tumorsz(element):
    if int(element) < 49:
        return 1
    elif 50 < int(element) < 99:
        return 2
    elif 100 < int(element) < 149:
        return 3
    elif 150 < int(element) < 998:
        return 4
    else:
        return 9

def encode_LYMND(element):
    if int(element) == 0:
        return 1
    elif 1 <=  int(element) <= 2:
        return 2
    elif 3 <= int(element) <= 6:
        return 3
    elif 7 <= int(element) <= 15:
        return 4
    elif 16 <= int(element) <= 95:
        return 5
    else:
        return 9


def encode_income(element):
    if element == '< $35,000' or element == '$35,000 - $39,999':
        return 0
    elif element == '$40,000 - $44,999':
        return 1
    elif element == '$45,000 - $49,999':
        return 2
    elif element == '$50,000 - $54,999':
        return 3
    elif element == '$55,000 - $59,999':
        return 4
    elif element == '$60,000 - $64,999':
        return 5
    elif element == '$65,000 - $69,999':
        return 6
    elif element == '$70,000 - $74,999':
        return 7
    elif element == '$75,000+':
        return 8
    else:
        return 9

def encode_rural_urban(element):
    if element == 'Counties in metropolitan areas ge 1 million pop':
        return 1
    elif element == 'Counties in metropolitan areas of 250,000 to 1 million pop':
        return 2
    elif element == 'Nonmetropolitan counties adjacent to a metropolitan area':
        return 3
    elif element == 'Nonmetropolitan counties not adjacent to a metropolitan area':
        return 4
    elif element == 'Counties in metropolitan areas of lt 250 thousand pop':
        return 5
    else:
        return 9

def encode_race(element):
    if element == 'White':
        return 0
    elif element == 'Black':
        return 1
    elif element == 'Other (American Indian/AK Native, Asian/Pacific Islander)':
        return 2
    else:
        return 9

def encode_race_origin(element):
    if element == 'Non-Hispanic White':
        return 0
    elif element == 'Non-Hispanic Black':
        return 1
    elif element == 'Hispanic (All Races)':
        return 2
    elif element == 'Non-Hispanic Asian or Pacific Islander':
        return 3
    elif element == 'Non-Hispanic American Indian/Alaska Native':
        return 4
    else:
        return 9    
    


def encode_radiation(element):
    if element ==' None/Unknown' or element == 'Radiation, NOS  method or source not specified' \
        or element == 'Recommended, unknown if administered' or element == 'Refused (1988+)':
        return 1 
    else:
        return 2



# if string in 'Laterality' column contains 'Right', then replace it with 1, 
# else if contains 'Left', then replace it with 2, 
# else if contains 'Paired', then replace it with 3,
# else replace it with 9
df['Laterality'] = df['Laterality'].apply(lambda x: 1 if 'Right' in x else 2 if 'Left' in x else 3 if 'Paired' in x else 9)


df['Age recode with <1 year olds'] = df['Age recode with <1 year olds'].apply(encode_age_range)

# if 'Sex' column is Male, then replace it with 1, else 2
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'Male' else 2)

# if 'Separate Tumor Nodules Ipsilateral Lung Recode (2010+)' column contains 'None', then replace it with 1, 
# else if contains 'Seprate nodules', then replace it with 2, else replace it with 9 
df['Separate Tumor Nodules Ipsilateral Lung Recode (2010+)'] = df['Separate Tumor Nodules Ipsilateral Lung Recode (2010+)'].apply(lambda x: 1 if 'None' in x else 2 if 'Seprate nodules' in x else 9)

# if 'Visceral and Parietal Pleural Invasion Recode (2010+)' column contains 'None', then replace it with 1,
# Too much blanks in this column
df['Visceral and Parietal Pleural Invasion Recode (2010+)'] = df['Visceral and Parietal Pleural Invasion Recode (2010+)'].apply(encode_SSF2_range)

# if 'Tumor Size Summary (2016+)' column is 'Blank(s)', then replace it with 9
# if between 1-49, then replace it with 1
# else if between 50-99, then replace it with 2
# else if between 100-149, then replace it with 3
# else if greater or equal to 150, then replace it with 4
df['CS tumor size (2004-2015)'] = df['CS tumor size (2004-2015)'].apply(encode_tumorsz)

# if 'Regional nodes positive' column is 0, then replace it with 1
# else if is 1-2, then replace it with 2
# else if is 3-6, then replace it with 3
# else if is 7-15, then replace it with 4
# else if is greater or equal to 16, then replace it with 5, else replace it with 9
df['Regional nodes positive (1988+)'] = df['Regional nodes positive (1988+)'].apply(encode_LYMND)

# Check if 'None/Unknown' exists in the 'Radiation recode' column
df['Radiation recode'] = df['Radiation recode'].apply(encode_radiation)

df['Chemotherapy recode (yes, no/unk)'] = df['Chemotherapy recode (yes, no/unk)'].apply(lambda x: 1 if x == 'No/Unknown' else 2)

df['Derived AJCC Stage Group, 6th ed (2004-2015)'] = df['Derived AJCC Stage Group, 6th ed (2004-2015)'].apply(encode_ajcc_stage)

df['RX Summ--Surg Prim Site (1998+)'] = df['RX Summ--Surg Prim Site (1998+)'].apply(lambda x: 1 if int(x) == 0 else 2)

# if 'Sequence number' column is '1st of 2 or more primaries', then replace it with 1, else 0
df['Target'] = 0
df.loc[df['Sequence number'] == '1st of 2 or more primaries', 'Target'] = 1


df['Median household income inflation adj to 2021'] = df['Median household income inflation adj to 2021'].apply(encode_income)
df['Rural-Urban Continuum Code'] = df['Rural-Urban Continuum Code'].apply(encode_rural_urban)
df['Race recode (White, Black, Other)'] = df['Race recode (White, Black, Other)'].apply(encode_race)
df['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'] = df['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'].apply(encode_race_origin)


# Make SEER data have the same column name

column_name_mapping = {
    # 'old_name': 'new_name'
    'Laterality':'Laterality',
    'Histologic Type ICD-O-3':'PTHLTYPE',
    'Age recode with <1 year olds':'Age',
    'Sex':'Gender',
    'Separate Tumor Nodules Ipsilateral Lung Recode (2010+)':'SepNodule',
    'Visceral and Parietal Pleural Invasion Recode (2010+)':'PleuInva',
    'CS tumor size (2004-2015)':'Tumorsz',
    'Regional nodes positive (1988+)':'LYMND',
    'Radiation recode':'Radiation',
    'Chemotherapy recode (yes, no/unk)':'Chemotherapy',
    'RX Summ--Surg Prim Site (1998+)':'Surgery', 
    'Derived AJCC Stage Group, 6th ed (2004-2015)': 'AJCC',
    'Median household income inflation adj to 2021':'Income',
    'Rural-Urban Continuum Code':'Area',
    'Race recode (White, Black, Other)':'Race', 
    'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)':'Origin'
}

#Drop the data before 2010 out 
df.rename(columns=column_name_mapping, inplace=True)
df = df[df['Year of diagnosis'] >= 2010]

df.to_csv('SEER_en.csv', index=False)
