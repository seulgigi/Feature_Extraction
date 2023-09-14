from sklearn.model_selection import train_test_split
from FeatureExtraction import feature_extraction
def GPFE(data, split_portion: float, ML_model, GP_config: dict, unit: dict):

    train_unit = []
    for feature in unit.keys():
        if unit[feature] == None:
            data = data.drop([feature], axis=1)
        else:
            train_unit.append(unit[feature])

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

    attribute_name = [column for column in data.columns if len(data[column].unique()) > 2] # unique값이 2개 초과 -> encoding 열 제외
    attribute = dict(zip(attribute_name, train_unit))
    data = data.to_dict()

    feature_extraction(GP_config, ML_model, attribute, attribute_name, train, validation, test, data, train_unit)