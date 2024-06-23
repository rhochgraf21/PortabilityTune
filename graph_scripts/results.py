# main imports
import statistics
import hashlib
import time
import sklearn.tree as sktree
import math
import itertools
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
import ast
import random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier as skNeighborsClassifier
from sklearn.model_selection import train_test_split
from statistics import geometric_mean
from matplotlib.ticker import ScalarFormatter, MultipleLocator, LogLocator
import seaborn as sns
import scipy.stats as stats
import dill
import os

# import the folders where tuning_kernels and portabilitytune are.
import sys

sys.path.append("../tuning_kernels")
sys.path.append("../ptuner")

# imports from tuning_kernels
import utils
import models
import dataset
import kernel_tree_classifier

# imports from ptuner
from src.plot import make_graphs
from src.db import TuningDatabase
from src.metrics import TuningMetric
from src.tuner import PortabilityTuner
from src.multi import MultiTuner
from src.poly import CT

# Data Files
DEVICES = ["HD500", "Iris", "Mali", "Vega", "Quadro"]
DEVICES_INPUTS = [
    ("Intel(R) Gen9 HD Graphics NEO", 750),
    ("Intel(R) Gen9 HD Graphics NEO", 1150),
    ("Mali-G71", 5),
    ("Radeon RX Vega", 1630),
    ("Quadro P5000", 1733),
]
DEVICE_MAP = {k: v for k, v in zip(DEVICES_INPUTS, DEVICES)}
DATA_FILES = ["intel1.csv", "intel7.csv", "mali.csv", "vega.csv", "quadro.csv"] # old, unused
DATA_FILES_MAP = {k: v for k, v in zip(DEVICES, DATA_FILES)} # old, unused

# Methods
CLUSTER_METHODS = ["KMeans"] # DecisionTree, KMeans, "TopN", "KMeans", "PCAKMeans", "Spectral"] # SPECIFY CLUSTER METHODS HERE
OTHER_METHODS = ["PortabilityTune"]
ALL_METHODS = CLUSTER_METHODS + OTHER_METHODS

# MODELS
MODELS = [
        # models.TopN,
        # models.DecisionTree,
        models.KMeans,
        # models.PCAKMeans,
        # models.Spectral,
        # models.HDBScan,
        ]
MODELS_MAP = {k: v for k, v in zip(CLUSTER_METHODS, MODELS)}

# Normalizations
NORMS = ["scale"] # "rawcutoff"]

# Number Of Kernels
NUM_KERNELS = [3, 5]

# Function to calculate ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def parse_args(config_filename):
    """
    Parses the configuration file.
    """

    with open(config_filename, "r") as config_file:
        config_data = yaml.safe_load(config_file)

        # parse tuning metrics
        config_data["tuning_metric"] = TuningMetric.get_metric(
            config_data["tuning_metric"])

        # parse the PortabilityTune() arguments
        try:
            config_data["devices"] = tuple(tuple(i)
                                           for i in config_data["devices"])
            config_data["db_devices"] = tuple(tuple(i)
                                           for i in config_data["db_devices"])
            config_data["inputs"] = tuple(tuple(i)
                                          for i in config_data["inputs"])
            config_data["arguments"] = tuple(
                tuple(i) for i in config_data["arguments"])
            config_data["baseline_inputs"] = tuple(
                tuple(i) for i in config_data["baseline_inputs"])
            config_data["baseline_arguments"] = tuple(
                tuple(i) for i in config_data["baseline_arguments"])
            global NUM_KERNELS
            NUM_KERNELS = config_data["num_kernels"]
        except:
            raise ValueError(
                "Invalid config.yml arguments. See the docs for example formatting.")

        if not config_data["devices"]:
            raise ValueError("PortabilityTune() requires at least one device.")

        if not config_data["inputs"]:
            raise ValueError("PortabilityTune() requires at least one input.")

        if not config_data["baseline_inputs"]:
            raise ValueError("Baseline requires at least one input.")

        if not "data_files" in config_data:
            raise ValueError("Clustering requires one or more data files (one per device).")

        return config_data

def split_dataset(rawdata):
    train_dataset, test_dataset=kernel_tree_classifier.split_dataset(rawdata, dup=True) # dup=True duplicates the full dataset
    tup=(train_dataset, test_dataset)
    return tup

def load_lawson_dataset(data_file, args = None):
    rawdata=dataset.load_cached(data_file, args=args)
    return rawdata

def get_lawson(data, method, norm, n_c, geo=True, label=False):
    model=MODELS_MAP[method]
    train, test = data
    print(train)
    train = kernel_tree_classifier.normalize_data(train, norm)
    print(train)

    # randomize the order of the dataset
    shuffle_indx = np.random.permutation(len(train.features))
    shuffle_data = dataset.DataSet(features=train.features.iloc[shuffle_indx].reset_index(drop=True), normalized=train.normalized.iloc[shuffle_indx].reset_index(drop=True),
                          values=train.values.iloc[shuffle_indx].reset_index(drop=True))

    csv_output=[]
    m=model(shuffle_data, n_c)
    labels=m.classes

    print(f"Clustering with {method} and {norm} normalization yielded", utils.geom_mean(utils.get_perfect_errors_for(labels, test)))

    if geo:
        error=utils.geom_mean(utils.get_perfect_errors_for(labels, test))
        return error
    else:
        if label:
            return labels
        # data = (train, test)
        error = utils.get_perfect_errors_for(labels, test, dct=geo)
        return error

    # Hack to remove HDBScan cache, as it doesn't know about different
    # files or normalization schemes
    # models.reset_hdbscan_cache()

    return error

def get_ptuner(args, db, k, geo=True, tuner=False, label=False, multi=False, select=0, num_runs=30, combined=False):
    if not multi:
        num_runs = 1
    args_str = str(args['devices']) + str(args['inputs']) + str(args['arguments']) + str(args['baseline_inputs']) + str(args['baseline_arguments']) + str(args['tuning_metric'].get_names()) + str(args['minimum_coverage']) + str(args['tuning_time']) + str(args['parallel_evals']) + str(args['kernel']) + str(args['database_path']) + str(args['clblast_database_path']) + str(args['db_devices']) + str(args['data_files'])
    filename = str(hashlib.sha256(str(args_str).encode('utf-8')).hexdigest()) + str(k) + str(geo) + str(tuner) + str(label) + str(multi) + str(select) + str(num_runs) + ".pkl"
    filepath = "pickles/" + filename
    pickle_exists = os.path.exists(filepath)
    if pickle_exists:
        with open(filepath, 'rb') as f:
            print("reading from", filepath)
            return dill.load(f)
    else:
        # perform the tuning with the given arguments
        tune = None
        if multi:
            tune=MultiTuner(args["kernel"], args['devices'], args["inputs"], args["arguments"], k,
                                args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"], select=select, num_runs=num_runs)
        else:
            tune=PortabilityTuner(args["kernel"], args['devices'], args["inputs"], args["arguments"], k,
                                args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])

        if tuner:
            with open(filepath, 'wb') as f:
                dill.dump(tune, f)
            return tune

        if label:
            with open(filepath, 'wb') as f:
                dill.dump(tune.output, f)
            return tune.output

        if multi and combined:
            with open(filepath, 'wb') as f:
                dill.dump(tune.runtimes, f)
            return tune.runtimes

        # return the geometric mean?
        if geo:
            # get reciprocals of all performance info
            reciprocal_perf=np.reciprocal(
                list(tune.optimal_settings.values()))
            # calculate geomean performance
            performance=geometric_mean(reciprocal_perf)
            # return performance
            with open(filepath, 'wb') as f:
                dill.dump(performance, f)
            return performance

        # no, return all performance results in a dict
        else:
            # return the optimal settings, take the reciprocal of each performance value
            # print(type(tuner.optimal_settings))
            with open(filepath, 'wb') as f:
                dill.dump({(entry.device, entry.input): tune.optimal_settings[entry] for entry in tune.optimal_settings}, f)
            return {(entry.device, entry.input): tune.optimal_settings[entry] for entry in tune.optimal_settings}

def get_baseline(args, db, k, geo=True, tuner=False):
    print("Tuning Baseline")
    # perform the tuning with the given arguments
    tune=PortabilityTuner(args["kernel"], args['devices'], args["baseline_inputs"], args["baseline_arguments"], 1,
                            args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])

    # get performance on full subset
    runtimes = tune.runtimesForSubset(args['devices'], args['inputs'], args['arguments'])

    if tuner:
        return tune

    if geo:
        # get reciprocals of all performance info
        reciprocal_perf=np.reciprocal(
            list(runtimes.values()))
        # calculate geomean performance
        performance=geometric_mean(reciprocal_perf)
        # return performance
        return performance
    else:
        # return the optimal settings, take the reciprocal of each performance value
        return {(entry.device, entry.input): runtimes[entry] for entry in runtimes}

def get_lawson_results(rows, data_file, args, geo=True, label=False, multi=False, combined=False):
    # Get Data for Clustering
    data = load_lawson_dataset(data_file)
    for method in CLUSTER_METHODS:
        for norm in NORMS:
            for k in NUM_KERNELS:
                a = 1
                if multi:
                    a = 30
                performances = []
                for _ in range(a):
                    split = split_dataset(data)
                    # data=load_lawson_dataset(data_file)
                    performance=get_lawson(
                                split, method, norm, k, geo, label)
                    if combined:
                        if performance < 10000000000:
                            performances.append(performance)
                    else:
                        row=(method + "-" + norm, k, performance)
                        rows.append(row)
                if combined:
                    row=(method + "-" + norm, k, performances)
                    rows.append(row)
    return rows

def get_ptuner_results(rows, data_file, args, device=None, geo=True, tuner=False, name="", baseline=True, multi=False, select=0, combined=False):
    device_args = args
    if device is not None:
        device_args['devices'] = [device]
    device_args['database_path'] = data_file
    print(device_args['database_path'])
    db = TuningDatabase(device_args)
    for n_k in NUM_KERNELS:
        print(f"[GRAPH] Starting Portabiltiy Tune with Arguments {args['devices']}, {args['inputs']}, {n_k}")
        # print(db.unpruned_data)
        performance=get_ptuner(args, db, n_k, geo=geo, tuner=tuner, multi=multi, select=select, combined=combined)
        if name:
            row=[name, n_k, performance]
        else:
            row=["PortabilityTune", n_k, performance]
        rows.append(row)
        if device is not None and baseline:
            # CLBlast baseline also calculated through PortabilityTune()
            performance=get_baseline(args, db, n_k, geo=geo, tuner=tuner)
            row=["CLBlast", 1, performance]
            rows.append(row)
    if device is None and baseline:
        # CLBlast baseline also calculated through PortabilityTune()
        performance=get_baseline(args, db, n_k, geo=geo, tuner=tuner)
        row=["CLBlast", 1, performance]
        rows.append(row)
    return rows

def get_all_entries(args):
    """
    Gets all possible entries for this dataset [device-portable]
    """
    entries = []
    for tup in itertools.product(args['devices'], args['inputs']):
        entries.append(tup)
    return entries

def display_bar_graph(rows, title="", filename="out.pdf", yaxis="Performance (Relative to Oracle)"):
    rows_list = []

    # extract unique values
    names = sorted(set(row[0] for row in rows))
    kernels = sorted(set(row[1] for row in rows))

    result = [[None for _ in kernels] for _ in names]

    # populate the results
    for row in rows:
        a_index = names.index(row[0])
        b_index = kernels.index(row[1])
        result[a_index][b_index] = row[2] #[float(a) for a in row[2]] # float(row[2]) #

    # each list is a row
    df1=pd.DataFrame(
            result, index=names, columns=kernels)
    ax = df1.plot.bar(rot=0, figsize=(17, 10))
    ax.set_title(title)
    ax.legend(title='Number of Variants')
    plt.show()
    plt.savefig(filename)
    plt.clf()
    print(df1)

def display_ip_scatter(df, device_unhash):
    sns.set()

    # Define explicit order for hue and style
    unique = list(df['method'].unique())
    print(len(unique))
    palette = dict(zip(unique, sns.color_palette("bright", n_colors=len(unique))))

    grid = sns.relplot(data=df, x='input', y='performance', hue='method', style='n_k', col_wrap=1, palette=palette, col='device_label', kind='line', facet_kws={'sharey': False, 'sharex': True, "legend_out": True}, clip_on=True, markers=True, height=2)
    grid.set_axis_labels("Input (m,n,k)", "Slowdown (S/O)")

    for ax in grid.axes.flatten():
        title = ax.get_title()
        col = int(title.split('=')[-1].strip())
        ax.set_title(DEVICE_MAP[device_unhash[int(col)]])
        props = {"rotation" : 90}
        plt.setp(ax.get_xticklabels(), **props)

    # plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space

    plt.show()

def main():

    # get the configuration filename
    try:
        config_filename=sys.argv[1]
    except:
        print("Missing argument: please enter a YAML configuration filename, eg config.yml.")
        quit()

    # parse the configuration args
    args=parse_args(config_filename)

    # rows in form: METHOD, N_C, NORM, PERFORMANCE
    rows_list=[]

    # device-portable results
    if 'portable' in args['graph_types']:
        print("Tuning Device-Portable")
        data_file = args['data_files'][0]
        db_path = args['database_path']
        rows=[]

        rows = get_ptuner_results(rows, db_path, args, geo=True, multi=True, combined=True)

        device_names = "".join([str(device[0])+' '+str(device[1])+' ' for device in args['devices']])
        device_names_short = "".join([(str(device[0])+str(device[1])).replace(' ','') for device in args['devices']])
        display_bar_graph(rows, f"Tuning Methods On {device_names}, All Inputs", f"{device_names_short}_portable.pdf")

    # device-fleet results
    if 'fleet' in args['graph_types']:
        print("Tuning Device-Portable Fleet")
        data_file = args['data_files'][0]
        db_path = args['database_path']
        db = TuningDatabase(args)
        rows=[]

        # first, get the results for ptuner
        saved_tuner = None
        i = 0
        tuners = {}
        global NUM_KERNELS
        for n_k in NUM_KERNELS:
            tuner = MultiTuner(args["kernel"], args['devices'], args["inputs"], args["arguments"], n_k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
            tuners[("PortabilityTune", n_k)] = tuner
            if i == 0:
                saved_tuner = tuner
            i += 1

        # then, get the results for clustering
        for method in CLUSTER_METHODS:
            for norm in NORMS:
                for k in NUM_KERNELS:
                    tuner = CT(MODELS_MAP[method], norm, k, data_file, args)
                    tuners[(method + "-" + norm, k)] = tuner

        # get the runtimes for each tuner
        runtimes = []
        for (name, n_k), tuner in tuners.items():
            print(tuner.parameters)
            runtimes.append([name, n_k, [1/saved_tuner.runs[0].runtimeOnSubset(args['devices'], args['inputs'], args['arguments'], parameters=r, coverage=False, speedup=False, all=False, dct=False) for r in tuner.parameters]])

        # manually override tuning params
        clblast_params = { 'HD500': '1,4,1,1,16,16,64,8,8,64,0,0,0,0,2,2', 'Iris Pro': '1,16,1,1,16,16,64,8,8,64,0,0,0,0,4,1', 'Quadro P5000': '0,1,32,2,8,32,128,16,8,128,1,1,1,1,2,2', 'Quadro P5000': '0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,1', 'Mali-G71': '0,1,32,2,16,16,64,8,8,64,0,0,0,0,4,4', 'Radeon RX Vega': '0,1,32,2,8,8,64,16,16,64,1,1,0,0,4,4' }
        clblast_runtime = 1/saved_tuner.runs[0].runtimeOnSubset(args['devices'], args['inputs'], args['arguments'], parameters=clblast_params, coverage=False, speedup=False, all=False, dct=False)
        runtimes.append(["CLBlast", 5, clblast_runtime])

        # make a dataframe from the runtimes
        df = pd.DataFrame(runtimes, columns=["Method", "Kernels", "Runtimes"])
        # now give each runtime its own row
        df = df.explode("Runtimes")
        # df = pd.DataFrame([["Decision Tree", 3, lawson_3_expected, lawson_3_upper, lawson_3_lower], ["Decision Tree", 5, lawson_5_expected, lawson_5_upper, lawson_5_lower], ["Portability Tune", 3, ptuner_3_expected, ptuner_3_upper, ptuner_3_lower], ["Portability Tune", 5, ptuner_5_expected, ptuner_5_upper, ptuner_5_lower]], columns=["Method", "Kernels", "Expected", "Upper", "Lower"])
        # now, plot the data
        ax = sns.barplot(x="Method", y="Runtimes", hue="Kernels", data=df)
        # set the y-axis label
        ax.set_ylabel("Tasks/Second")
        # show the plot
        plt.show()

    # device-specific tunings
    if 'specific' in args['graph_types']:
        # must provide a data_file for each device
        if len(args['devices']) != len(args['data_files']):
            raise ValueError("Must provide a data_file for each device.")
        # tune for each device
        for device, data_file in zip(args['devices'], args['data_files']):
            rows=[]

            print(f"Tuning Device-Specific for device {device}")

            # Get Data for Clustering
            rows = get_lawson_results(rows, data_file, args)
            rows = get_ptuner_results(rows, data_file, args, device)
            display_bar_graph(rows, f"Tuning Methods On {device[0]}-{device[1]}, All Inputs", f"{device[0].replace(' ','')}-{device[1]}_specific.pdf")

    # device-portable tuning, with result for every input shown
    if 'ipscatter' in args['graph_types']:

        print("Tuning IP Scatter")
        data_file = args['data_files'][0]
        db_path = args['database_path']
        lawson_rows=[]
        ptuner_rows=[]

        # Get Data
        print("Getting Cluster Data")
        lawson_rows = get_lawson_results(lawson_rows, data_file, args, geo=False)

        print("Getting Portability Tune Data")
        ptuner_rows = get_ptuner_results(ptuner_rows, db_path, args, geo=False, multi=True, tuner=False)

        # get the entry results (for debugging)
        """
        all_entries = get_all_entries(args)
        for entry in all_entries:
            device, input = entry
            print(f"Device: {device}, Input: {input}")
        """

        # get the device labels
        device_unhash = {}
        for device in args['devices']:
            device_unhash[dataset.alt_hash(device[0] + " " + str(device[1]))] = device

        # convert to final rows in form (method, k, (m,n,k), label)
        combined_rows = []

        # convert rows
        for dtype, n_k, d in ptuner_rows:
            for (device, input), performance in d.items():
                method = dtype
                input = input
                device_label = dataset.alt_hash(device[0] + " " + str(device[1]))
                combined_rows.append((method, n_k, str(input), device_label, performance))

        for dtype, n_k, t in lawson_rows:
            for (m, n, k, b, l), performance in t.items():
                combined_rows.append((dtype, n_k, str((m,n,k)), l, 1/performance))

        combined_rows.sort(key = lambda x: x[1]) # sort by n_k
        combined_rows.sort(key = lambda x: math.prod(ast.literal_eval(x[2]))) # sort by input size (m*n*k)
        combined_rows.sort(key = lambda x: x[0]) # sort by method
        combined_rows.sort(key = lambda x: x[3]) # sort by device

        # print the rows
        for row in combined_rows:
            print(row[0], row[1], row[2], device_unhash[row[3]], row[4])

        df = pd.DataFrame(combined_rows, columns=['method', 'n_k', 'input', 'device_label', 'performance'])

        # display the scatterplot
        display_ip_scatter(df, device_unhash)

    if 'boxplot' in args['graph_types']:
        # get the data
        db = TuningDatabase(args)
        entries = db.make_subset(args['kernel'], args['devices'], args['inputs'], args['arguments']).get_entries_format()
        rows = []
        devices = set()
        for entry in entries:
            for result in entry.results.values():
                devices.add(DEVICE_MAP[entry.device])
                rows.append((DEVICE_MAP[entry.device], result))
        df = pd.DataFrame(rows, columns=['device', 'performance'])

        ax = sns.boxenplot(data=df, x="performance", y="device", log_scale=10) #kind="kde")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Slowdown Relative to Oracle (Log Scale)')
        ax.set_ylabel('GPU Device')
        plt.show()

    if 'tuningtime' in args['graph_types']:
        data = """
Seconds	PortabilityTune 10	PortabilityTune 4	PortabilityTune 1	KMeans 4	KMeans 10	DecisionTree 4	DecisionTree 10
10	1.32	1.646	1.67	1.258251642	1.199116998	1.204238921	1.142499353
20	1.27	1.2831	1.67	1.258251642	1.199116998	1.204238921	1.142499353
30	1.23	1.2307	1.67	1.258251642	1.199116998	1.204238921	1.142499353
40	1.23	1.2244	1.67	1.258251642	1.199116998	1.204238921	1.142499353
50	1.20	1.2244	1.67	1.258251642	1.199116998	1.204238921	1.142499353
60	1.19	1.2244	1.67	1.258251642	1.199116998	1.204238921	1.142499353
70	1.18	1.2244	1.67	1.258251642	1.199116998	1.204238921	1.142499353
80	1.18	1.2244	1.67	1.258251642	1.199116998	1.204238921	1.142499353
90	1.18	1.2189	1.67	1.258251642	1.199116998	1.204238921	1.142499353
100	1.18	1.2189	1.67	1.258251642	1.199116998	1.204238921	1.142499353
110	1.15	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
120	1.15	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
130	1.14	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
140	1.14	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
150	1.14	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
160	1.13	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
170	1.13	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
180	1.13	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
190	1.13	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
200	1.13	1.2005	1.67	1.258251642	1.199116998	1.204238921	1.142499353
210	1.13	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
220	1.13	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
230	1.13	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
240	1.13	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
250	1.13	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
260	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
270	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
280	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
290	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
300	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
310	1.11	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
320	1.10	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
330	1.10	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
340	1.10	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
350	1.10	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
360	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
370	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
380	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
390	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
400	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
410	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
420	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
430	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
440	1.09	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
450	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
460	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
470	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
480	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
490	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
500	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
510	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
520	1.08	1.1981	1.67	1.258251642	1.199116998	1.204238921	1.142499353
530	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
540	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
550	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
560	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
570	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
580	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
590	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
600	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
610	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
620	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
630	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
640	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
650	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
660	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
670	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
680	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
690	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
700	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
710	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
720	1.08	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
730	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
740	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
750	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
760	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
770	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
780	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
790	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
800	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
810	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
820	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
830	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
840	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
850	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
860	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
870	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
880	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
890	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
900	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
910	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
920	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
930	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
940	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
950	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
960	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
970	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
980	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
990	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
1000	1.07	1.1927	1.67	1.258251642	1.199116998	1.204238921	1.142499353
"""
        data = data.replace('\t', ',')
        from io import StringIO
        df = pd.read_csv(StringIO(data))

        # Melting the DataFrame to make it suitable for Seaborn
        df_melted = df.melt('Seconds', var_name='columns', value_name='value')

        # Plotting using Seaborn
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='Seconds', y='value', hue='columns')
        # set y axis label
        plt.ylabel('Geomean Slowdown Relative to Oracle')
        plt.xlabel('Tuning Time (Seconds)')
        # set legend label
        plt.legend(title='Method | Number of Variants')
        # place legend in middle right
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.show()

    if 'unseenarchci' in args['graph_types']:
        """
        WARNING: Only run when len(NUM_KERNELS) == 1, ie only [5] variants or only [10] variants.
        """

        num_variants = NUM_KERNELS[0]

        db_path = args['database_path']

        # get the results data
        rows = []
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 5 devices, all inputs", tuner=True, baseline=False, multi=True)  # all 5, all inputs
        inp = args['inputs']
        args['inputs'] = ((1024, 1024, 1024),)
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 5 devices, M=N=K=1024", tuner=True, baseline=False, multi=True)  # all 5, 1024
        db_path = args['clblast_database_path']
        dev = args['devices']
        args['devices'] = args['db_devices']
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 71 devices", tuner=True, baseline=False, multi=True)  # all 5, 1024
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 10 devices", tuner=True, baseline=False, multi=True, select=10)  # all 5, 1024
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 7 devices", tuner=True, baseline=False, multi=True, select=7)  # all 5, 1024
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 6 devices", tuner=True, baseline=False, multi=True, select=6)  # all 5, 1024
        rows = get_ptuner_results(rows, db_path, args, name="PortabilityTune, 5 devices", tuner=True, baseline=False, multi=True, select=5)  # all 5, 1024
        args['devices'] = dev
        args['inputs'] = inp

        # parse the tuners from the result rows
        ptuner_all_5 = rows[0][2]
        ptuner_1024_5 = rows[1][2]
        ptuner_db_5 = rows[2][2]
        ptuner_db_5_10 = rows[3][2]
        ptuner_db_5_7 = rows[4][2]
        ptuner_db_5_6 = rows[5][2]
        ptuner_db_5_5 = rows[6][2]

        runtimes = []
        coverages = []

        # select the tuners to use in the evaluation
        tuners = [ptuner_all_5, ptuner_db_5, ptuner_db_5_10, ptuner_db_5_7, ptuner_db_5_5] # ptuner_db_5 (71), ptuner_db_5_30, ptuner_db_5_10
        indices = [0, 2, 3, 4, 6]
        # tuners = [ptuner_1024_3, ptuner_1024_5, ptuner_all_3, ptuner_all_5, ptuner_db_3, ptuner_db_5, ptuner_db_3_50, ptuner_db_5_50, ptuner_db_3_30, ptuner_db_5_30, ptuner_db_3_10, ptuner_db_5_10, ptuner_db_3_5, ptuner_db_5_5]

        # get the runtimes for each tuner
        for a in tuners:
            runtime, coverage = a.runtimeOnSubset(args['devices'], args['inputs'], args['arguments'], db=ptuner_all_5.getDB(), coverage=True, speedup=True, all=True, dct=True)
            runtimes.append(runtime)
            coverages.append(coverage)

        # skip rows that are not in the indices, otherwise add expected runtime and confidence interval
        data = []
        i = 0
        j = 0

        for r in rows:
            label = r[0]
            k = r[1]
            perf = r[2]
            if i not in indices:
                print("skipping ", i)
                i += 1
                continue
            data.append([label, k, runtimes[j], coverages[j]]) # * 100 / len(args['db_devices'])])
            data.append([label, k, runtimes[j], coverages[j]]) # * 100 / len(args['db_devices'])])
            data.append([label, k, runtimes[j], coverages[j]]) # * 100 / len(args['db_devices'])])
            i += 1
            j += 1

        # add results for clblast
        clblast_params = { 'HD500': '1,4,1,1,16,16,64,8,8,64,0,0,0,0,2,2',
 'Iris Pro': '1,16,1,1,16,16,64,8,8,64,0,0,0,0,4,1',
 'Quadro P5000': '0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,1',
 'Mali-G71': '0,1,32,2,16,16,64,8,8,64,0,0,0,0,4,4',
 'Radeon RX Vega': '0,1,32,2,8,8,64,16,16,64,1,1,0,0,4,4' }
        r, c = ptuner_1024_5.runtimeOnSubset(args['devices'], args['inputs'], args['arguments'], db=ptuner_all_5.getDB(), parameters=clblast_params, speedup=True, coverage=True, all=True, dct=True)
        data.append(["CLBlast", num_variants, r, c * 100 / len(args['db_devices'])])

        sns.set()

        df = pd.DataFrame(data, columns=["label", "k", "time", "coverage"])

        # plot the data, using slowdown relative to oracle instead of speedup, by taking reciprocals of the runtimes
        # df["time"] = df["time"].apply(lambda x: np.reciprocal(x))

        violin_data = [(entry[0], entry[1], 1/times[0], 1/times[1], 1/times[2]) for entry in data for times in entry[2]]
        violin_df = pd.DataFrame(violin_data, columns=['label', 'k', 'upper', 'median', 'lower'])
        # allow k=5 and k=NUM_KERNELS[0]
        violin_df = violin_df[(violin_df['k'] == NUM_KERNELS[0]) | (violin_df['k'] == num_variants)]
        palette = sns.color_palette("husl", n_colors=len(violin_df['label'].unique())+1)

        print(violin_df)

        for i, category in enumerate(violin_df['label'].unique()):
            subset = violin_df[violin_df['label'] == category]
            y_median, x_median = ecdf(subset['median'])
            y_upper, x_upper = ecdf(subset['upper'])
            y_lower, x_lower = ecdf(subset['lower'])

            # calculate area under the curve for median
            auc_median = np.trapz(y_median, x_median)

            # Plot ECDF for median
            plt.plot(x_median, y_median, label=f'{category}; AUC = {auc_median:.2f}', color=palette[i])

            # Add error bounds
            plt.fill_between(x_median, np.interp(x_median, x_lower, y_lower),
                     np.interp(x_median, x_upper, y_upper), color=palette[i], alpha=0.2)


        plt.xlabel('Proportion')
        plt.ylabel('Slowdown Relative To Oracle')
        plt.legend(title='Category')

        # make the plot logscale
        plt.xscale('log')

        ax = plt.gca()
        # Set y-axis tick labels manually
        #x = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 1]
        #y = [a for a in range(1, 20)]
        # get the y-ticks
        #x = ax.get_yticks()
        #ax.set_yticks(y)
        #ax.set_yticklabels(['{:.1f}'.format(i) for i in y])

        plt.show()

        input()

main()
