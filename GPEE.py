from sklearn.model_selection import train_test_split
from FeatureExtraction import feature_extraction
import copy
def GPFE(data, test_split_portion: float, validation_split_portion, ML_model, GP_config: dict, unit: dict):

    data1=copy.deepcopy(data)
    train_unit = []
    for feature in unit.keys():
        if unit[feature] == None:
            data1 = data1.drop([feature], axis=1)
        else:
            train_unit.append(unit[feature])

    attribute_name = [column for column in data1.columns[:-1]]
    attribute = dict(zip(attribute_name, train_unit))


    if test_split_portion:
        train_1, test = train_test_split(data, stratify=data['Decision'], test_size=test_split_portion, random_state=2)

        if validation_split_portion != 0:
            train, validation = train_test_split(train_1, stratify=train_1['Decision'], test_size=validation_split_portion, random_state=2)
            train, validation, test = train.to_dict(), validation.to_dict(), test.to_dict()

        else:
            train, validation, test = train_1.to_dict(),train_1.to_dict(), test.to_dict() # TODO validation_split_portion None이라면 validation -> training

    else:
        train, validation, test = data.to_dict(), data.to_dict(), data.to_dict()




    data = data.to_dict()

    feature_extraction(GP_config, ML_model, attribute, attribute_name, train, validation, test, data, train_unit, validation_split_portion)