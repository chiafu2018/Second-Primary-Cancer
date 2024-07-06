import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, roc_curve
import utils

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Already remove the "LOC" that centralized learning doesn't need this
columns = ["TMRPRMYST", "SYMOGN", "Gender", 
           "SSF1", "SSF2", "SSF4", "SSF6", "SSF7", "Age", "DIFF", "TMRSZ", 
           "LYMND", "AJCCstage", "MAGN", "RT", "ST", "BMI_label", "CIG", "BN",
            "ALC", "OP", "Target"]

df = pd.read_csv('Data_folder/Data_preprocessed.csv')
df = df[columns]

trainset, testset = utils.data_preprocess(df, size = 0.35)
x_test, y_test = testset.drop(columns=['Target']),testset['Target']
print(f"Dataset for secondary training (data number, feature number): {x_test.shape}")


# The optimal cut off would be where tpr is high and fpr is low
model_cen = load_model("Model_cen")
cen_prob = model_cen.predict(x_test)
fpr, tpr, threshold_cen = roc_curve(y_test, cen_prob)
optimal_index = np.argmax(tpr-fpr)
y_cen = [1 if (prob >= threshold_cen[optimal_index]) else 0 for prob in cen_prob]


model_fed = load_model("Model_fed")
fed_prob = model_fed.predict(x_test)
fpr, tpr, threshold_fed = roc_curve(y_test, fed_prob)
optimal_index = np.argmax(tpr-fpr)
y_fed = [1 if (prob >= threshold_fed[optimal_index]) else 0 for prob in fed_prob]



if(np.sum(y_cen)>0 and np.sum(y_fed)>0):
    f1_local = f1_score(y_test, y_cen)
    f1_global = f1_score(y_test, y_fed)
    f1 = {
        'f1 global': np.array(f1_global).astype(float),
        'f1 local': np.array(f1_local).astype(float)
    }

    f1_csv = pd.DataFrame(f1, index=[0])
    print(f1_csv)
    f1_csv.to_csv("init.csv",index=False)

    result = {
        'global model predict yes prob': fed_prob.reshape(-1,),
        'global model predict no prob': np.array([1 - prob for prob in fed_prob.reshape(-1,)]),    
        'local model predict yes prob': cen_prob.reshape(-1,),
        'local model predict no prob': np.array([1 - prob for prob in cen_prob.reshape(-1,)]),
        'Outcome': y_test
    }

    middle = pd.DataFrame(result)
    middle.to_csv("middle.csv",index=False)

else:
    print("Your model just predict all zeros. Please train both your models again or enlarge your testset.")



