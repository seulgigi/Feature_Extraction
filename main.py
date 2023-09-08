import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from FeatureExtraction import GPFE

df = pd.read_csv('/Users/choeseong-geun/DT_tree/penguins.csv', encoding='cp949')
df=pd.get_dummies(df)
df=df[['culmen_length_mm','culmen_depth_mm','flipper_length_mm',
        'body_mass_g','species_Adelie','species_Chinstrap','species_Gentoo',
        'island_Biscoe','island_Dream','island_Torgersen','Decision']]

unit = ['mm','mm','mm','g','BLANK','BLANK','BLANK','BLANK','BLANK','BLANK'] * 1

ML_model = {'ML': 'classification', 'model': DecisionTreeClassifier(random_state=0,max_depth=3)}

GP_config = {'population_size': 15, 'chromosome_size': 15, 'max_generation':500}

GPFE(data=df,
     split_portion=0.1,
     ML_model=ML_model,
     GP_config=GP_config,
     unit=unit
     )
