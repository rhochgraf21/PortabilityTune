"""
File: decide.py
This file generates a decision tree that infers the optimal-performing portability tuned parameters from a small benchmark.

Given:
  - a vector of k values, in the format: (k < 10)
    - [(GEMMK, VW, WPT, WGS, M, N, K, runtime), ... k[9]] # the Portability Tuned parameter values, the input size, and the runtime (all for this device)
  - an input size (M, N, K)
  - device clock speed, in MHz? [don't provide directly - may overfit]
  - number of compute units? [don't provide directly - may overfit]  
Predict:
  - the optimal parameter configuration for this device and input size [of the Portability Tuned parameters]

Idea:
- choose k Portability Tuned parameters and a representative input size for each
- train a decision tree using the performance results of the k Portability Tuned parameters on the representative input sizes,
  the device clock speed, the number of compute units, and the input size,
  and predict the optimal of the k Portability Tuned parameters for the given (device, input size) pair
(e.g. use a small benchmark to generalize the Portability Tuned code params for a larger benchmark)
"""

import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import numpy as np
from statistics import geometric_mean
from gplearn.genetic import SymbolicRegressor

import src.db as db
from src.db import Parameters

# use make_subset to prune the subset of the database to only include the relevant data (e.g. only keep the results for the Portability Tuned parameter configurations)
# from the subset:
#   column_names = ['param_config', 'input', 'runtime', ...] # set the column names [e.g. 'param_config1', 'input1', 'runtime1', ..., 'given_input', 'output_config']
#   df = pd.DataFrame(columns=column_names)

#   # create a row containing k parameter configurations, the input size, and the runtime
#   for _ in range(10000):
    #   row = []
    #   device = devices[_ % len(devices)]
    #   for paramc in portability_tuned_params:
    #       entr = select a random entry with device and paramc, if none exist, row += np.nan, np.nan, np.nan
    #       row += entr.param_config, entr.input, entr.runtime
    #   entr2 = select a random entry with device
    #   row += entr2.input, entr2.runtime (select the best runtime), params_for_best_runtime
    #   df.loc[-1] = row


# remove any duplicate rows
# df.drop_duplicates(subset=None, keep='first', inplace=True)
# X_train, X_test, y_train, y_test = train_test_split(new_dataset, y_values, test_size=0.3, random_state=1) # 70% training and 30% test
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def build_tree(subset, parameter_variants):
    """
    Builds a decision tree that infers the optimal-performing portability tuned parameters from a small benchmark.

    :param: perfvector: a vector of k values, in the format: (k < 10)
    """
    # set the column names [e.g. 'param_config1', 'input1', 'runtime1', ..., 'given_input', 'output_config']
    column_names = []
    parameter_names = subset.get_parameter_names()
    print(parameter_names)
    for i in range(len(parameter_variants.str_repr())):
        for parameter_name in parameter_names:
            column_names += [parameter_name + str(i)]
        column_names += 'M'+str(i), 'N'+str(i), 'K'+str(i), 'runtime'+str(i)
    column_names += ['M_input', 'N_input', "K_input"]
    # column_names += ['device_core_clock_speed']
    column_names += ['device']
    column_names += ['output_config']

    df = pd.DataFrame(columns=column_names)
    rows = []

    # create a row containing k parameter configurations, the input size, and the runtime
    for i in range(10000):
        print(i)
        row = []
        device = subset.devices[i % len(subset.devices)]
        for paramc in parameter_variants.str_repr():
            entr = subset.get_random_entry(device=device)
            found = False
            if entr is not None:
                # get the runtime for this parameter configuration
                for p, r in entr.unadjusted_results.items(): # unadjusted_results
                    if p == paramc:
                        found = True
                        pc = paramc.split(',')
                        for val in pc:
                            row += [val]
                        for val in entr.input:
                            row += [val]
                        row += [r]
            if entr is None or not found:
                pc = paramc.split(',')
                for val in pc:
                    row += [val]
                if entr is not None:
                    for val in entr.input:
                        row += [val]
                else:
                    row += [np.nan, np.nan, np.nan]
                row += [np.nan]

        # get result entry from this device
        entr2 = subset.get_random_entry(device=device)
        if entr2 is not None:
            # get the runtime for this parameter configuration
            p, r = entr2.get_best_param(adj=False)
            for val in entr2.input:
                row += [val]
            # row += [entr2.device[1]]
            row += [i % len(subset.devices)]
            for paramc, j in zip(parameter_variants.str_repr(), range(len(parameter_variants.str_repr()))):
                if paramc == p:
                    row += [j+1]
            rows.append(row)

    # remove any duplicate rows
    df = pd.DataFrame(rows, columns=column_names)
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(-1)
    # df = df.dropna()
    print(df)
    # separate the 'output_config' column
    y_values = df['output_config']
    # get all of df except output_config column
    data = df.drop('output_config', axis=1)
    data = data.drop('device', axis=1)

    print(data)

    X_train, X_test, y_train, y_test = train_test_split(data, y_values, test_size=0.7, random_state=1) # 70% training and 30% test

    #X_train = df[(df['M_input'] != 1024) | (df['N_input'] != 1024) | (df['K_input'] != 1024)]
    #X_test = df[(df['M_input'] == 1024) & (df['N_input'] == 1024) & (df['K_input'] == 1024)]
    
    #y_train = X_train['output_config']
    #y_test = X_test['output_config']
    #X_train = X_train.drop('output_config', axis=1)
    #X_test = X_test.drop('output_config', axis=1)

    import warnings
    from sklearn.exceptions import DataConversionWarning

    # take the best of 10 trainings of the decision tree
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning)
        performances = []   
        for _ in range(10):
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            print(y_pred)
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

            # identical training and testing split as the one above
            X_train2, X_test2, y_train2, y_test2 = train_test_split(df, y_values, test_size=0.7, random_state=1) # 70% training and 30% test
            X_test2.columns = df.columns
            #print(df.columns)

            #X_train2 = df[(df['M_input'] != 1024) | (df['N_input'] != 1024) | (df['K_input'] != 1024)]
            #X_test2 = df[(df['M_input'] == 1024) & (df['N_input'] == 1024) & (df['K_input'] == 1024)]
            #y_train2 = X_train2['output_config']
            #y_test2 = X_test2['output_config']

            result_dct = dict()
            for rw1, rw2 in zip(X_test.iterrows(), X_test2.iterrows()):
                
                # num = clf.predict(rw)
                # get the predict config from the row
                num = clf.predict([rw1[1]])
                device = subset.devices[rw2[1]['device']]
                input = (rw2[1]['M_input'], rw2[1]['N_input'], rw2[1]['K_input'])
                sub2 = subset.make_subset(subset.kernel, [device], [input], subset.arguments)
                #print(parameter_variants.str_repr()[num[0]-1], device, input)
                #print(parameter_variants.parameters[num[0]-1])
                kvc = sub2.evaluate(Parameters([parameter_variants.parameters[num[0]-1]]))        
                for k, v in kvc[0].items():
                    result_dct[k] = v
            
            # print(result_dct)
            # print(list(result_dct.values()))
            print(geometric_mean(list(result_dct.values())))
            performances += [geometric_mean(list(result_dct.values()))]
        print(performances)
        

def predict_params(perfvector: list, device_clock_speed: int, num_compute_units: int, input_size: tuple):
    """
    Predicts the optimal parameters for a given device and input size.

    :param: perfvector: a vector of k values, in the format: (k < 10)
    :param: device_clock_speed: the device clock speed, in MHz
    :param: num_compute_units: the number of compute units
    :param: input_size: the input size (M, N, K)

    :return: the optimal parameter configuration for this device and input size [of the Portability Tuned parameters]
    """
    pass

def symbolic_reg(subset, parameter_variants):
    """
    Perform symbolic regression to find the equation mapping parameter benchmark and input to actual runtime.

    :param: perfvector: a vector of k values, in the format: (k < 10)
    """
    # set the column names [e.g. 'param_config1', 'input1', 'runtime1', ..., 'given_input', 'output_config']
    column_names = []
    parameter_names = subset.get_parameter_names()
    print(parameter_names)
    for i in range(len(parameter_variants.str_repr())):
        for parameter_name in parameter_names:
            column_names += [parameter_name + str(i)]
            break
        column_names += 'M'+str(i), 'N'+str(i), 'K'+str(i), 'runtime'+str(i)
    column_names += ['M_input', 'N_input', "K_input"]
    # column_names += ['device_core_clock_speed']
    column_names += ['device']
    column_names += ['output_config']

    df = pd.DataFrame(columns=column_names)
    rows = []

    # create a row containing k parameter configurations, the input size, and the runtime
    for i in range(30000):
        print(i)
        row = []
        device = subset.devices[i % len(subset.devices)]
        # for paramc in parameter_variants.str_repr():
        entr = subset.get_random_entry(device=device)
        found = False
        if entr is not None:
            # get the runtime for this parameter configuration
            for p, r in random.shuffle(list(entr.results.items()))  : # unadjusted_results # shuffle
                # check if p is str
                if isinstance(p, str):
                    found = True
                    pc = paramc.split(',')
                    for val in pc:
                        row += [val]
                    for val in entr.input:
                        row += [val]
                    row += [r]
                    break
        if entr is None or not found:
            pc = paramc.split(',')
            for val in pc:
                row += [val]
            if entr is not None:
                for val in entr.input:
                    row += [val]
            else:
                row += [np.nan, np.nan, np.nan]
            row += [np.nan]

        # get result entry from this device
        entr2 = subset.get_random_entry(device=device)
        if entr2 is not None:
            # get the runtime for this parameter configuration
            p, r = entr2.get_best_param(adj=True)
            for val in entr2.input:
                row += [val]
            # row += [entr2.device[1]]
            # row += [i % len(subset.devices)]
            row += [r]
            for paramc, j in zip(parameter_variants.str_repr(), range(len(parameter_variants.str_repr()))):
                if paramc == p:
                    # row += [j+1]
                    row += [r]
            rows.append(row)

    # remove any duplicate rows
    df = pd.DataFrame(rows, columns=column_names)
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(-1)
    # df = df.dropna()
    print(df)
    # separate the 'output_config' column
    y_values = df['output_config']
    # get all of df except output_config column
    data = df.drop('output_config', axis=1)
    data = data.drop('device', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, y_values, test_size=0.7, random_state=1) # 70% training and 30% test

    print(X_test, y_test)

    a = input("Press enter to continue")

    #X_train = df[(df['M_input'] != 1024) | (df['N_input'] != 1024) | (df['K_input'] != 1024)]
    #X_test = df[(df['M_input'] == 1024) & (df['N_input'] == 1024) & (df['K_input'] == 1024)]
    
    #y_train = X_train['output_config']
    #y_test = X_test['output_config']
    #X_train = X_train.drop('output_config', axis=1)
    #X_test = X_test.drop('output_config', axis=1)

    import warnings
    from sklearn.exceptions import DataConversionWarning

    # take the best of 10 trainings of the decision tree
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning)
        performances = []   
        for _ in range(1):
            # clf = DecisionTreeClassifier()
            function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'inv',
                'max', 'min', 'sin', 'tan']
            est_gp = SymbolicRegressor(population_size=10000,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.6, p_subtree_mutation=0.2,
                           p_hoist_mutation=0.1, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.00001, random_state=0, n_jobs=4)
            # est_gp.fit(X_train, y_train)
            clf = est_gp.fit(X_train,y_train)
            # y_pred = clf.predict(X_test)
            # print(y_pred)
            # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            score_gp = est_gp.score(X_test, y_test)
            print(f"Regression Score: {score_gp}")
            print(f"Program: {est_gp._program}")

            """

            # identical training and testing split as the one above
            X_train2, X_test2, y_train2, y_test2 = train_test_split(df, y_values, test_size=0.7, random_state=1) # 70% training and 30% test
            X_test2.columns = df.columns
            #print(df.columns)

            #X_train2 = df[(df['M_input'] != 1024) | (df['N_input'] != 1024) | (df['K_input'] != 1024)]
            #X_test2 = df[(df['M_input'] == 1024) & (df['N_input'] == 1024) & (df['K_input'] == 1024)]
            #y_train2 = X_train2['output_config']
            #y_test2 = X_test2['output_config']

            result_dct = dict()
            for rw1, rw2 in zip(X_test.iterrows(), X_test2.iterrows()):
                
                # num = clf.predict(rw)
                # get the predict config from the row
                num = clf.predict([rw1[1]])
                device = subset.devices[rw2[1]['device']]
                input = (rw2[1]['M_input'], rw2[1]['N_input'], rw2[1]['K_input'])
                sub2 = subset.make_subset(subset.kernel, [device], [input], subset.arguments)
                #print(parameter_variants.str_repr()[num[0]-1], device, input)
                #print(parameter_variants.parameters[num[0]-1])
                kvc = sub2.evaluate(Parameters([parameter_variants.parameters[num[0]-1]]))        
                for k, v in kvc[0].items():
                    result_dct[k] = v
            
            # print(result_dct)
            # print(list(result_dct.values()))
            print(geometric_mean(list(result_dct.values())))
            performances += [geometric_mean(list(result_dct.values()))]
        print(performances)
            """