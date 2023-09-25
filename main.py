import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from GPEE import GPFE
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('/Users/choeseong-geun/DT_tree/heart_failure_clinical_records_dataset_clean.csv')
df = df[[col for col in df.columns if col != 'Decision'] + ['Decision']]
unit = {'age' : None,
        'anaemia' : None,
        'creatinine_phosphokinase' : 'mcg/L',
        'diabetes' : None,
        'ejection_fraction' : 'percentage',
        'high_blood_pressure' : None,
        'platelets' : 'kiloplatelets/mL',
        'serum_creatinine' : 'mg/dL',
        'serum_sodium' : 'mEq/L',
        'sex' : None,
        'smoking' : None,
        'time' : None
        }


ML_model = {'ML': 'classification', 'model': DecisionTreeClassifier(random_state=0, max_depth=3),'classification_evaluate': accuracy_score,'regression_evaluate':None}

GP_config = {'population_size': 10, 'chromosome_size': 15, 'max_generation': 99999}

GPFE(data=df,
     test_split_portion=0.3,
     validation_split_portion=0.3, # validation_split_portion을 None으로 넣었을 경우 training accuracy를 구함
     ML_model=ML_model,
     GP_config=GP_config,
     unit=unit
     )