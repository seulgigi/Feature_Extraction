from sklearn.model_selection import train_test_split
from FeatureExtraction import feature_extraction
def GPFE(data, split_portion: float, ML_model, GP_config: dict, unit: list):
    if split_portion:
        train, test = train_test_split(data, stratify=data['Decision'], test_size=split_portion, random_state=2)
        train, validation = train_test_split(data, stratify=data['Decision'], test_size=split_portion, random_state=2)

        # 나눠진 데이터셋을 csv 파일로 저장
        train.to_csv('./dataset/train.csv', index=False)
        test.to_csv('./dataset/test.csv', index=False)
        validation.to_csv('./dataset/validation.csv', index=False)

        train, validation, test = train.to_dict(), validation.to_dict(), test.to_dict()
    else:
        train, validation, test = data.to_dict(), data.to_dict(), data.to_dict()

    attribute_name = data.columns[:-1]
    attribute = dict(zip(attribute_name, unit))
    data = data.to_dict()

    feature_extraction(GP_config, ML_model, attribute, attribute_name, train, validation, test, data, unit)