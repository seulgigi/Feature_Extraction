import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from FeatureExtraction import GPFE

#df = pd.read_csv('dataset/A.csv', encoding='cp949')
df = pd.read_csv('dataset/CNC_total_small.csv', encoding='cp949')
# unit = { 'featuer_name' : 'unit' }
unit = {'species_Adelie' : None,
        'species_Chinstrap' : None,
        'species_Gentoo' : None,
        'island_Torgersen' : None,
        'island_Dream' : None,
        'island_Biscoe' : None,
        'culmen_length_mm' : 'mm',
        'culmen_depth_mm' : 'mm',
        'flipper_length_mm' : 'mm',
        'body_mass_g' : 'g'
        }
#ML_model = {'ML': 'regression', 'model': DecisionTreeRegressor(random_state=0, max_depth=5)}
ML_model = {'ML': 'classification', 'model': DecisionTreeClassifier(random_state=0, max_depth=2), 'evaluate': accuracy_score}
GP_config = {'population_size': 10, 'chromosome_size': 15, 'max_generation': 999999}

GPFE(data=df,
     split_portion=0.1,
     ML_model=ML_model,
     GP_config=GP_config,
     unit=unit
     )
