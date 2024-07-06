import pandas as pd
from tableone import TableOne

data = pd.read_csv('SEER_en.csv')

print(data.columns)

'''
squamous cell carcinoma (8051-2, 8070-6, 8078, 8083-4, 8090, 8094, 8120, 8123)
small cell carcinoma (8002, 8041-5)
adenocarcinoma (8015, 8050, 8140-1, 8143-5, 8147, 8190, 8201, 8211, 8250-5, 8260, 8290, 8310, 8320, 8323, 8333, 8401, 8440, 8470-1, 8480-1, 8490, 8503, 8507, 8550, 8570-2, 8574, 8576)
large cell carcinoma (8012-4, 8021, 8034, 8082
other specified carcinoma (8003-4, 8022, 8030-3, 8035, 8200, 8240-1, 8243-6, 8249, 8430, 8525, 8560, 8562, 8575)
carcinoma not otherwise specified [NOS] (8010-1, 8020, 8230)
non-small cell carcinoma (8046) 
malignant neoplasm NOS (8000-1)
'''

mapping = {
    'squamous cell carcinoma': ['8051', '8052', '8070', '8071', '8072', '8073', '8074', '8075', '8076', '8078', '8083', '8084', '8090', '8094', '8120', '8123'],
    'small cell carcinoma': ['8002', '8041', '8042', '8043', '8044', '8045'],
    'adenocarcinoma': ['8015', '8050', '8140', '8141', '8143', '8144', '8145', '8147', '8190', '8201', '8211', '8250', '8251', '8252', '8253', '8254', '8255', '8260', '8290', '8310', '8320', '8323', '8333', '8401', '8440', '8470', '8471', '8480', '8481', '8490', '8503', '8507', '8550', '8570', '8571', '8572', '8574', '8576'],
    'large cell carcinoma': ['8012', '8013', '8014', '8021', '8034', '8082'],
    'other specified carcinoma': ['8003', '8004', '8022', '8030', '8031', '8032', '8033', '8035', '8200', '8240', '8241', '8243', '8244', '8245', '8246', '8249', '8430', '8525', '8560', '8562', '8575'],
    'carcinoma not otherwise specified [NOS]': ['8010', '8011', '8020', '8230'],
    'non-small cell carcinoma': ['8046'],
    'malignant neoplasm NOS': ['8000', '8001']
}

# Flatten the mapping for quick lookup
flat_mapping = {}
for group, codes in mapping.items():
    for code in codes:
        flat_mapping[code] = group

# Function to encode PTHLTYPE
def encode_PTHLTYPE(pthltype):
    return flat_mapping.get(pthltype, 'unknown')

# Apply encoding
data['Histologic Type ICD-O-3'] = data['Histologic Type ICD-O-3'].astype(str)  # Ensure PTHLTYPE is string type
data['Histologic Type ICD-O-3_encoded'] = data['Histologic Type ICD-O-3'].apply(encode_PTHLTYPE)
data['Target']=data['Target'].astype(int)

# Show result
print(data[['Histologic Type ICD-O-3', 'Histologic Type ICD-O-3_encoded']])


myTable = TableOne(data, columns=['Histologic Type ICD-O-3_encoded'], groupby='Target', pval=True)
myTable.to_csv('PTHL_tableone.csv')