import pandas as pd
import json
from statistics import mean


# store json in dictionary
def read_database(filename='clblast_database_new_pruned.json'):
    """
    Parses JSON database and returns as dictionary.
    """
    with open(filename, encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile)
        return df['sections'].to_dict()

# store params in list
def params_dict_to_list(dct):
    """
    Converts a dictionary of parameters to a list of parameters.
    """
    return [a for a in dct.values()]

# useful for double-checking results
def evaluate_performance(input_kernels, df):
    performances = dict() # stores performances of devices
    valids = dict() # stores devices where parameters are valid
    seen_devices = set() # stores devices which are known to these parameters

    every_perf = [] # stores performances for all devices (including ones not recorded with these parameters)
    all_d = set() # all devices (including ones not recorded with these parameters)
    l = 0 # ?

    MIN_RUNTIME = 80

    print("starting search")

    # add results_formatted
    for section in df:
        kf = df[section]["kernel_family"]
        p = df[section]["precision"]
        if "arg_alpha" in df[section] and int(p) == 32:
            alpha = df[section]["arg_alpha"]
        else:
            continue
        if kf == "xgemm" and "i" not in alpha and float(alpha) == 2:

            if "clblast_device_architecture" in df[section]:
                d = (df[section]["clblast_device_architecture"], df[section]["clblast_device_name"], df[section]["device_core_clock"])
                seen_devices.add(d)
            else:
                continue

            clock = df[section]["device_core_clock"]
            all_d.add(d)
            l += 1

            # store parameter names
            params = df[section]["parameter_names"]
            # store min value
            minval = 100000000000

            new = False

            # adjust runtimes relative to oracle
            for result in df[section]["results"]:
                param_str = result[0]
                runtime = result[1]
                if runtime < minval:
                    minval = runtime

            # for each, find if this is within range. add to set if it is
            for result in df[section]["results"]:
                param_str = result[0]
                adj_runtime = minval * 100 / result[1]  # % of optimal perf
                every_perf.append(adj_runtime)
                if param_str in input_kernels:

                    if d in performances:
                        if adj_runtime > performances[d]:
                            performances[d] = adj_runtime
                    else:
                        performances[d] = adj_runtime
                        valids[d] = True

                    if adj_runtime > MIN_RUNTIME:
                        # print(param_str, adj_runtime, d)
                        pass

    print("printing performances")
    sorted_perf = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1], reverse=True)}
    for a in sorted_perf:
        print(a, sorted_perf[a])

    optimal = [performances[a] for a in performances if performances[a] > MIN_RUNTIME]

    print("not included were: ")
    for a in seen_devices.difference(set(performances.keys())):
        print(a)


    for a in all_d:
        print(f"- [{a[1]}, {a[2]}]")

    print(l, len(all_d))

    print(f"found {len(optimal)} counts of > {MIN_RUNTIME}% performance, of {len(seen_devices)} total devices. The parameters were valid on {len(valids)}. Avg performance = {mean(performances.values())}, random perf = {mean(every_perf)}")

""" PER DEVICE, CLBLAST PARAMS - 1024, 1024, 1024 """
# input_dct = {'HD500': '1,4,1,1,16,16,64,8,8,64,0,0,0,0,2,2','Iris Pro': '1,16,1,1,16,16,64,8,8,64,0,0,0,0,4,1','Quadro P5000': '0,1,32,2,8,32,128,16,8,128,1,1,1,1,2,2','Quadro P5000': '0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,1','Mali-G71': '0,1,32,2,16,16,64,8,8,64,0,0,0,0,4,4','Radeon RX Vega': '0,1,32,2,8,8,64,16,16,64,1,1,0,0,4,4'}

input_dct = {'Param0': '0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,4', 'Param1': '0,1,32,2,8,8,32,8,8,32,1,1,0,0,4,4', 'Param2': '0,1,16,2,8,8,32,8,16,128,1,0,1,1,4,8'}

input_kernels = params_dict_to_list(input_dct)

# example usage
evaluate_performance(input_kernels, read_database("../datasets/database_means.json"))
