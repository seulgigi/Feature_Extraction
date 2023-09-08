from sklearn.model_selection import train_test_split
from sympy import sympify
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def GPFE(data, split_portion: float, ML_model, GP_config: dict, unit: list):
    if split_portion:
        train, test = train_test_split(data, stratify=data['Decision'], test_size=0.3, random_state=2)
        train, test = train.to_dict(), test.to_dict()
    else:
        train, test = data.to_dict(), data.to_dict()


    # . 
    attribute_name = [column for column in data.columns if len(data[column].unique()) > 2] # unique값이 2개 초과 -> encoding 열 제외
    attribute = dict(zip(attribute_name, unit))
    data = data.to_dict()

    def feature_extraction(_init_size, _attribute, _attribute_name, _data, _test, df):
        def check_combine(_chr):
            if _chr[2] == 'attr':
                check_attr1 = _attribute[_chr[0]]
            elif _chr[2] == '+' or _chr[2] == '-' or _chr[2] == 'min' or _chr[2] == 'max':
                check_attr1 = _attribute[_chr[0]]
            else:
                check_attr1 = str(sympify('(' + _attribute[_chr[0]] + ')' + _chr[2] + '(' + _attribute[_chr[1]] + ')'))

            if _chr[5] == 'attr':
                check_attr2 = _attribute[_chr[3]]
            elif _chr[5] == '+' or _chr[5] == '-' or _chr[5] == 'min' or _chr[5] == 'max':
                check_attr2 = _attribute[_chr[3]]
            else:
                check_attr2 = str(sympify('(' + _attribute[_chr[3]] + ')' + _chr[5] + '(' + _attribute[_chr[4]] + ')'))

            if (_chr[-1] == '*') or (_chr[-1] == '/'):
                return True
            elif (_chr[-1] == '+') or (_chr[-1] == '-'):
                if check_attr1 == check_attr2:
                    return True
                else:
                    return False

        def check_feasible(chr, _df):
            check_df = copy.deepcopy(_df)
            extracted_feature = combine(chr)
            equation = calculate(chr)

            for i in check_df['Decision'].keys():
                try:
                    eval(equation)
                except:
                    return False
            return True

        def combine(_chr):
            if _chr[2] == 'attr':
                check_attr1 = str(_chr[0])
            elif _chr[2] == 'min' or _chr[2] == 'max':
                check_attr1= str(sympify(_chr[2] + '(' + _chr[0] + ',' + _chr[1] + ')'))
            else:
                check_attr1 = str(sympify('(' + _chr[0] + ')' + _chr[2] + '(' + _chr[1] + ')'))
            if _chr[5] == 'attr':
                check_attr2 = str(_chr[3])
            elif _chr[5] == 'min' or _chr[5] == 'max':
                check_attr2= str(sympify(_chr[5] + '(' + _chr[3] + ',' + _chr[4] + ')'))
            else:
                check_attr2 = str(sympify('(' + _chr[3] + ')' + _chr[5] + '(' + _chr[4] + ')'))
            return str('(' + check_attr1 + ')' + _chr[-1] + '(' + check_attr2 + ')')

        def train_calculate(_chr):
            if _chr[2] != 'attr' and _chr[2] != 'min' and _chr[2] != 'max':
                check_attr1 = '(train_data["' + _chr[0] + '"][i]' + _chr[2] + 'train_data["' + _chr[1] + '"][i])'
            elif _chr[2] == 'min' or _chr[2] == 'max':
                check_attr1 = _chr[2]+'(train_data["' + _chr[0] + '"][i], train_data["' + _chr[1] + '"][i])'
            else:
                check_attr1 = '(train_data["' + _chr[0] + '"][i])'

            if _chr[5] != 'attr' and _chr[5] != 'min' and _chr[5] != 'max':
                check_attr2 = '(train_data["' + _chr[3] + '"][i]' + _chr[5] + 'train_data["' + _chr[4] + '"][i])'
            elif _chr[5] == 'min' or _chr[5] == 'max':
                check_attr2 = _chr[5]+'(train_data["' + _chr[3] + '"][i], train_data["' + _chr[4] + '"][i])'
            else:
                check_attr2 = '(train_data["' + _chr[3] + '"][i])'

            return str(check_attr1 + _chr[-1] + check_attr2)

        def calculate(_chr):
            if _chr[2] != 'attr' and _chr[2] != 'min' and _chr[2] != 'max':
                check_attr1 = '(_df["' + _chr[0] + '"][i]' + _chr[2] + '_df["' + _chr[1] + '"][i])'
            elif _chr[2] == 'min' or _chr[2] == 'max':
                check_attr1 = _chr[2]+'(_df["' + _chr[0] + '"][i], _df["' + _chr[1] + '"][i])'
            else:
                check_attr1 = '(_df["' + _chr[0] + '"][i])'

            if _chr[5] != 'attr' and _chr[5] != 'min' and _chr[5] != 'max':
                check_attr2 = '(_df["' + _chr[3] + '"][i]' + _chr[5] + '_df["' + _chr[4] + '"][i])'
            elif _chr[5] == 'min' or _chr[5] == 'max':
                check_attr2 = _chr[5]+'(_df["' + _chr[3] + '"][i], _df["' + _chr[4] + '"][i])'
            else:
                check_attr2 = '(_df["' + _chr[3] + '"][i])'

            return str(check_attr1 + _chr[-1] + check_attr2)

        def test_calculate(_chr):
            if _chr[2] != 'attr' and _chr[2] != 'min' and _chr[2] != 'max':
                check_attr1 = '(test_data["' + _chr[0] + '"][i]' + _chr[2] + 'test_data["' + _chr[1] + '"][i])'
            elif _chr[2] == 'min' or _chr[2] == 'max':
                check_attr1 = _chr[2] + '(test_data["' + _chr[0] + '"][i], test_data["' + _chr[1] + '"][i])'
            else:
                check_attr1 = '(test_data["' + _chr[0] + '"][i])'

            if _chr[5] != 'attr' and _chr[5] != 'min' and _chr[5] != 'max':
                check_attr2 = '(test_data["' + _chr[3] + '"][i]' + _chr[5] + 'test_data["' + _chr[4] + '"][i])'
            elif _chr[5] == 'min' or _chr[5] == 'max':
                check_attr2 = _chr[5] + '(test_data["' + _chr[3] + '"][i], test_data["' + _chr[4] + '"][i])'
            else:
                check_attr2 = '(test_data["' + _chr[3] + '"][i])'

            return str(check_attr1 + _chr[-1] + check_attr2)

        def init_population(_init_size, _attribute, _attribute_name, _data, _df, chromosomes):
            used_chromosome, population_list = [], []
            while len(population_list) < _init_size:
                _chromosome_list, operation, operation2 = [], ['+', '-', '*', '/'], ['+', '-', '*', '/', 'attr', 'max', 'min']
                chromosome_in_population = chromosomes
                while len(_chromosome_list) != chromosome_in_population:
                    selectOP1 = np.random.choice(operation, 1, replace=True)
                    selectOP2 = np.random.choice(operation2, 2, replace=True)
                    if (selectOP2[0] == '-') or (selectOP2[0] == '+') or (selectOP2[0] == 'max') or (selectOP2[0] == 'min'):
                        selectATTR1 = np.random.choice(_attribute_name, 1, replace=False)
                        try:
                            selectATTR1 = np.append(selectATTR1, np.random.choice(list(
                                filter(lambda x: (_attribute[x] == _attribute[selectATTR1[0]]) and x != selectATTR1[0],
                                       _attribute_name)), 1)[0])
                        except:
                            selectOP2[0] = 'attr'
                    else:
                        selectATTR1 = np.random.choice(_attribute_name, 2, replace=True)

                    if (selectOP2[1] == '-') or (selectOP2[1] == '+') or (selectOP2[1] == 'max') or (selectOP2[1] == 'min'):
                        selectATTR2 = np.random.choice(_attribute_name, 1, replace=False)
                        try:
                            selectATTR2 = np.append(selectATTR2, np.random.choice(list(
                                filter(lambda x: (_attribute[x] == _attribute[selectATTR2[0]]) and x != selectATTR2[0],
                                       _attribute_name)), 1)[0])
                        except:
                            selectOP2[1] = 'attr'
                    else:
                        selectATTR2 = np.random.choice(_attribute_name, 2, replace=True)

                    if selectOP2[0] == 'attr' and selectOP2[1] != 'attr':
                        chr = np.array(
                            [selectATTR1[0], 'BLANK', selectOP2[0], selectATTR2[0], selectATTR2[1], selectOP2[1],
                             selectOP1[0]], dtype='U60')

                    elif selectOP2[0] != 'attr' and selectOP2[1] == 'attr':
                        chr = np.array(
                            [selectATTR1[0], selectATTR1[1], selectOP2[0], selectATTR2[0], 'BLANK', selectOP2[1],
                             selectOP1[0]], dtype='U60')

                    elif selectOP2[0] == 'attr' and selectOP2[1] == 'attr':
                        chr = np.array(
                            [selectATTR1[0], 'BLANK', selectOP2[0], selectATTR2[0], 'BLANK', selectOP2[1],
                             selectOP1[0]],
                            dtype='U60')

                    else:
                        chr = np.array(
                            [selectATTR1[0], selectATTR1[1], selectOP2[0], selectATTR2[0], selectATTR2[1], selectOP2[1],
                             selectOP1[0]], dtype='U60')

                    if check_combine(chr) and check_feasible(chr, _df) and combine(chr) != '1' and combine(
                            chr) != '2' and combine(chr) != '0.5':
                        _chromosome_list.append(chr)
                    else:
                        continue
                population_list.append(_chromosome_list)
            return population_list

        def evaluate_fitness(_population_list, forest, _attribute, _data, _test):
            for _population in _population_list:
                train_data = copy.deepcopy(_data)
                test_data = copy.deepcopy(_test)
                for _chromosome in _population:
                    extracted_feature = combine(_chromosome)
                    train_equation = train_calculate(_chromosome)
                    test_equation = test_calculate(_chromosome)
                    if not extracted_feature in train_data:
                        train_data[extracted_feature] = {}
                        test_data[extracted_feature] = {}

                        for i in train_data['Decision'].keys():
                            try:
                                train_data[extracted_feature].update({i: eval(train_equation)})
                            except:
                                np.place(_chromosome, _chromosome == '/', '*')
                                train_equation = train_calculate(_chromosome)
                                train_data[extracted_feature].update({i: eval(train_equation)})

                        for i in test_data['Decision'].keys():
                            try:
                                test_data[extracted_feature].update({i: eval(test_equation)})
                            except:
                                np.place(_chromosome, _chromosome == '/', '*')
                                test_equation = test_calculate(_chromosome)
                                test_data[extracted_feature].update({i: eval(test_equation)})

                train_chromosome = np.array(list(map(lambda x: list(train_data[x].values()), train_data)))
                test_chromosome = np.array(list(map(lambda x: list(test_data[x].values()), test_data)))

                if ML_model['ML'] == 'regression':
                    regressor = ML_model['model']
                    regressor.fit(
                        np.concatenate((train_chromosome.T[:, :len(unit)], train_chromosome.T[:, len(unit) + 1:]), axis=1),
                        train_chromosome.T[:, len(unit)])
                    y_pred = regressor.predict(
                        np.concatenate((test_chromosome.T[:, :len(unit)], test_chromosome.T[:, len(unit) + 1:]), axis=1))

                    forest.append([_population, mse(test_chromosome.T[:, len(unit)], y_pred)])
                else:
                    classification = ML_model['model']
                    classification.fit(
                        np.concatenate((train_chromosome.T[:, :len(unit)], train_chromosome.T[:, len(unit) + 1:]), axis=1),
                        train_chromosome.T[:, len(unit)])
                    y_pred = classification.predict(
                        np.concatenate((test_chromosome.T[:, :len(unit)], test_chromosome.T[:, len(unit) + 1:]), axis=1))

                    forest.append([_population, accuracy_score(test_chromosome.T[:, len(unit)], y_pred)])
            return forest

        def crossover_mutate(_population_list, _max_generation, forest, _data, test_data, _attribute, _attribute_name,
                             chromosomes):
            generation, operation = 1, ['+', '-', '*', '/', 'max', 'min']
            chromosome_in_population = chromosomes

            def tournament_selection(_forest):
                part = np.random.choice(range(len(_forest)), int(len(_forest) * 0.25), replace=False)
                parents = sorted(list(map(lambda y: _forest[y], part)), key=lambda x: x[1], reverse=False)
                return parents[0]

            if ML_model['ML'] == 'regression':
                best_parent_chromosome, best_parent_MAE = sorted(forest, key=lambda x: x[1], reverse=False)[0]
            else:
                best_parent_chromosome, best_parent_ACC = sorted(forest, key=lambda x: x[1], reverse=True)[0]
            while generation <= _max_generation:
                parent = sorted(forest, key=lambda x: x[1], reverse=False)[:int(_init_size * 0.1) + 1]
                parent.extend([tournament_selection(forest) for i in range(int(_init_size * 0.9) - 1)])
                new_parent = []
                np.random.shuffle(parent)
                parent_chromosome, parent_performance = list(map(lambda x: x[0], parent)), list(
                    map(lambda x: x[1], parent))

                for i in range(int(len(parent_chromosome) / 2)):
                    child1 = parent_chromosome[i * 2][:int(chromosome_in_population / 2)] + parent_chromosome[
                                                                                                i * 2 + 1][
                                                                                            int(chromosome_in_population / 2):]
                    child2 = parent_chromosome[i * 2 + 1][:int(chromosome_in_population / 2)] + parent_chromosome[
                                                                                                    i * 2][
                                                                                                int(chromosome_in_population / 2):]

                    new_parent.append(child1)
                    new_parent.append(child2)

                for idx, _population in enumerate(new_parent):
                    for i in range(int(len(_population) / 2)):
                        mutate_probability = np.random.rand(1)[0]
                        if mutate_probability > 0.7:
                            def mutate1():
                                _copy1 = copy.deepcopy(new_parent[idx][i * 2])
                                _operation = ['+', '-', '*', '/', 'attr', 'max', 'min']

                                selectOP = str(np.random.choice(_operation, 1, replace=True)[0])
                                if (selectOP == '-') or (selectOP == '+') or (selectOP == 'max') or (selectOP == 'min'):
                                    selectATTR = np.random.choice(_attribute_name, 1, replace=False)
                                    try:
                                        selectATTR = np.append(selectATTR, np.random.choice(list(filter(
                                            lambda x: (_attribute[x] == _attribute[selectATTR[0]]) and x != selectATTR[
                                                0],
                                            _attribute)), 1)[0])
                                    except:
                                        selectOP = 'attr'

                                else:
                                    selectATTR = np.random.choice(_attribute_name, 2, replace=True)

                                if selectOP != 'attr':
                                    new_parent[idx][i * 2][0] = selectATTR[0]
                                    new_parent[idx][i * 2][1] = selectATTR[1]
                                    new_parent[idx][i * 2][2] = selectOP

                                else:
                                    new_parent[idx][i * 2][0] = selectATTR[0]
                                    new_parent[idx][i * 2][1] = 'BLANK'
                                    new_parent[idx][i * 2][2] = selectOP

                                if check_combine(new_parent[idx][i * 2]) and check_feasible(new_parent[idx][i * 2],
                                                                                            _data) and check_feasible(
                                    new_parent[idx][i * 2], test_data) and combine(
                                    new_parent[idx][i * 2]) != '1' and combine(
                                    new_parent[idx][i * 2]) != '2' and combine(
                                    new_parent[idx][i * 2]) != '0.5' and combine(new_parent[idx][i * 2 + 1]) != '0':
                                    pass

                                else:
                                    new_parent[idx][i * 2] = _copy1

                            mutate1()
                        mutate_probability = np.random.rand(1)[0]
                        if mutate_probability > 0.7:
                            def mutate2():
                                _copy2 = copy.deepcopy(new_parent[idx][i * 2 + 1])
                                _operation = ['+', '-', '*', '/', 'attr', 'max', 'min']

                                selectOP = str(np.random.choice(_operation, 1, replace=True)[0])
                                if (selectOP == '-') or (selectOP == '+') or (selectOP =='max') or (selectOP=='min'):
                                    selectATTR = np.random.choice(_attribute_name, 1, replace=False)
                                    try:
                                        selectATTR = np.append(selectATTR, np.random.choice(list(filter(
                                            lambda x: (_attribute[x] == _attribute[selectATTR[0]]) and x != selectATTR[
                                                0],
                                            _attribute)), 1)[0])
                                    except:
                                        selectOP = 'attr'
                                else:
                                    selectATTR = np.random.choice(_attribute_name, 2, replace=True)

                                if selectOP != 'attr':
                                    new_parent[idx][i * 2 + 1][0] = selectATTR[0]
                                    new_parent[idx][i * 2 + 1][1] = selectATTR[1]
                                    new_parent[idx][i * 2 + 1][2] = selectOP

                                else:
                                    new_parent[idx][i * 2 + 1][0] = selectATTR[0]
                                    new_parent[idx][i * 2 + 1][1] = 'BLANK'
                                    new_parent[idx][i * 2 + 1][2] = selectOP

                                if check_combine(new_parent[idx][i * 2 + 1]) and check_feasible(
                                        new_parent[idx][i * 2 + 1],
                                        _data) and check_feasible(
                                    new_parent[idx][i * 2 + 1], test_data) and combine(
                                    new_parent[idx][i * 2 + 1]) != '1' and combine(
                                    new_parent[idx][i * 2 + 1]) != '2' and combine(
                                    new_parent[idx][i * 2 + 1]) != '0.5' and combine(new_parent[idx][i * 2 + 1]) != '0':
                                    pass

                                else:
                                    new_parent[idx][i * 2 + 1] = _copy2

                            mutate2()

                        _copy1 = copy.deepcopy(new_parent[idx][i * 2])
                        _copy2 = copy.deepcopy(new_parent[idx][i * 2 + 1])
                        new_parent[idx][i * 2] = np.concatenate((_copy1[:3], _copy2[3:]))
                        new_parent[idx][i * 2 + 1] = np.concatenate((_copy2[:3], _copy1[3:]))
                        if check_combine(new_parent[idx][i * 2]) and check_feasible(new_parent[idx][i * 2],
                                                                                    _data) and check_feasible(
                            new_parent[idx][i * 2], test_data) and combine(
                            new_parent[idx][i * 2]) != '1' and combine(
                            new_parent[idx][i * 2]) != '2' and combine(
                            new_parent[idx][i * 2]) != '0.5' and combine(new_parent[idx][i * 2]) != '0':
                            pass


                        else:

                            new_parent[idx][i * 2] = _copy1

                            new_parent[idx][i * 2 + 1] = _copy2

                        if check_combine(new_parent[idx][i * 2 + 1]) and check_feasible(new_parent[idx][i * 2 + 1],

                                                                                        _data) and check_feasible(

                            new_parent[idx][i * 2 + 1], test_data) and combine(

                            new_parent[idx][i * 2 + 1]) != '1' and combine(

                            new_parent[idx][i * 2 + 1]) != '2' and combine(

                            new_parent[idx][i * 2 + 1]) != '0.5' and combine(new_parent[idx][i * 2 + 1]) != '0':

                            pass


                        else:

                            new_parent[idx][i * 2] = _copy1

                            new_parent[idx][i * 2 + 1] = _copy2

                new_forest = []
                forest = evaluate_fitness(new_parent, new_forest, attribute, _data, test_data)
                generation += 1
                if ML_model['ML'] == 'regression':
                    best_tree, best_MAE = sorted(forest, key=lambda x: x[1], reverse=False)[0]
                    if best_parent_MAE < best_MAE:
                        best_tree, best_MAE = best_parent_chromosome, best_parent_MAE
                    else:
                        best_parent_chromosome, best_parent_MAE = best_tree, best_MAE
                    print('Generation ' + str(generation) + ' best MAE: ' + str(best_MAE))
                    print([combine(i) for i in best_tree])
                else:
                    best_tree, best_acc = sorted(forest, key=lambda x: x[1], reverse=True)[0]
                    if best_parent_ACC > best_acc:
                        best_tree, best_acc = best_parent_chromosome, best_parent_ACC
                    else:
                        best_parent_chromosome, best_parent_ACC = best_tree, best_acc
                    print('Generation ' + str(generation) + ' best ACC: ' + str(best_acc))
                    print([combine(i) for i in best_tree])

        population_list, forest = init_population(_init_size, attribute, attribute_name, _data, df, GP_config['chromosome_size']), []

        forest = evaluate_fitness(population_list, forest, attribute, _data, _test)
        if ML_model['ML'] == 'regression':
            best_tree, best_MAE = sorted(forest, key=lambda x: x[1], reverse=False)[0]
            print('Generation ' + str(1) + ' best MAE: ' + str(best_MAE))
            print([combine(i) for i in best_tree])
        else:
            best_tree, best_ACC = sorted(forest, key=lambda x: x[1], reverse=True)[0]
            print('Generation ' + str(1) + ' best ACC: ' + str(best_ACC))
            print([combine(i) for i in best_tree])
        crossover_mutate(population_list, GP_config['max_generation'], forest, _data, _test, attribute, attribute_name, GP_config['chromosome_size'])

    feature_extraction(GP_config['population_size'], attribute, attribute_name, train, test, data)