from sklearn.model_selection import train_test_split
from FeatureExtraction import feature_extraction
def GPFE(data, split_portion: float, ML_model, GP_config: dict, unit: list):
    if split_portion:
        train, test = train_test_split(data, stratify=data['Decision'], test_size=split_portion, random_state=2)
        train, validation = train_test_split(data, stratify=data['Decision'], test_size=split_portion, random_state=2)
        train, validation, test = train.to_dict(), validation.to_dict(), test.to_dict()
    else:
        train, validation, test = data.to_dict(), data.to_dict(), data.to_dict()

    attribute_name = data.columns[:-1]
    attribute = dict(zip(attribute_name, unit))
    data = data.to_dict()

    feature_extraction(GP_config, ML_model, attribute, attribute_name, train, validation, test, data, unit)