# main imports
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier as skNeighborsClassifier
from sklearn.model_selection import train_test_split
from statistics import geometric_mean

# command line arg parsing
import sys

# imports from tuning_kernels
import tuning_kernels.utils as utils
import tuning_kernels.models as models
import tuning_kernels.dataset as dataset
import tuning_kernels.kernel_tree_classifier as kernel_tree_classifier

# imports from ptuner
from ptuner.src.plot import make_graphs
from ptuner.src.db import TuningDatabase
from ptuner.src.metrics import TuningMetric
from ptuner.src.tuner import PortabilityTuner

# Data Files
DEVICES = ["Iris Pro", "HD500", "Mali-G71", "Radeon RX Vega", "Quadro P5000"]
DATA_FILES = ["intel1.csv", "intel7.csv", "mali.csv", "vega.csv", "quadro.csv"]
DATA_FILES_MAP = {k: v for k, v in zip(DEVICES, DATA_FILES)}

# Methods
CLUSTER_METHODS = ["TopN", "DecisionTree", "KMeans", "PCAKMeans", "Spectral"]
OTHER_METHODS = ["PortabilityTune"]
ALL_METHODS = CLUSTER_METHODS + OTHER_METHODS

# MODELS
MODELS = [
        models.TopN,
        models.DecisionTree,
        models.KMeans,
        models.PCAKMeans,
        models.Spectral,
        # models.HDBScan,
        ]
MODELS_MAP = {k: v for k, v in zip(CLUSTER_METHODS, MODELS)}

# Normalizations
NORMS = ["rawcutoff", "scale"]

# Number Of Kernels
NUM_KERNELS = [5, 6, 7] # should be overriden by config.yml


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
        
        if not "num_kernels" in config_data:
            raise ValueError("Clustering requires one or more number of kernels.")

        return config_data

def load_lawson_dataset(data_file):
    rawdata=dataset.load_cached(data_file)
    train_dataset, test_dataset=kernel_tree_classifier.split_dataset(rawdata)

    tup=(train_dataset, test_dataset)

    return tup

def get_lawson(data, method, norm, n_c, geo=False):
    model=MODELS_MAP[method]
    train, test=data
    train = kernel_tree_classifier.normalize_data(train, norm)
    csv_output=[]
    m=model(train, n_c)
    labels=m.classes
    error=utils.geom_mean(utils.get_perfect_errors_for(labels, test))

    # Hack to remove HDBScan cache, as it doesn't know about different
    # files or normalization schemes
    models.reset_hdbscan_cache()

    return error

def get_ptuner(args, db, k, geo=False):
    # perform the tuning with the given arguments
    tuner=PortabilityTuner(args["kernel"], args['devices'], args["inputs"], args["arguments"], k,
                            args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])    
    
    # return the geometric mean?
    if geo:
        # get reciprocals of all performance info
        reciprocal_perf=np.reciprocal(
            list(tuner.optimal_settings.values()))
        # calculate geomean performance
        performance=geometric_mean(reciprocal_perf)
        # return performance
        return performance
    
    # no, return all performance results in a dict
    else:
        # return the optimal settings, take the reciprocal of each performance value
        return {k: 1/v for k,v in tuner.optimal_settings}

def get_baseline(args, db, k, geo=False):
    # perform the tuning with the given arguments
    tuner=PortabilityTuner(args["kernel"], args['devices'], args["baseline_inputs"], args["baseline_arguments"], 1,
                            args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
    if geo:
        # get reciprocals of all performance info
        reciprocal_perf=np.reciprocal(
            list(tuner.optimal_settings.values()))
        # calculate geomean performance
        performance=geometric_mean(reciprocal_perf)
        # return performance
        return performance
    else:
        # return the optimal settings, take the reciprocal of each performance value
        return {k: 1/v for k,v in tuner.optimal_settings}

def get_lawson_results(rows, data_file, args):
    # Get Data for Clustering
    data=load_lawson_dataset(data_file)
    for method in CLUSTER_METHODS:
        for norm in NORMS:
            for k in NUM_KERNELS:
                performance=get_lawson(
                        data, method, norm, k)
                row=(method + "-" + norm, k, performance)
                rows.append(row)
    return rows

def get_ptuner_results(rows, data_file, args, device=None, geo=False):
    device_args = args
    if device is not None:
        device_args['devices'] = [device]
    db = TuningDatabase(device_args)
    for n_k in NUM_KERNELS:
        print(f"[GRAPH] Starting Portabiltiy Tune with Arguments {args['devices']}, {args['inputs']}, {n_k}")
        print(db.unpruned_data)
        performance=get_ptuner(args, db, n_k, geo)
        row=["PortabilityTune", n_k, performance]
        rows.append(row)
        if device is not None:
            # CLBlast baseline also calculated through PortabilityTune()
            performance=get_baseline(args, db, n_k, geo)
            row=["CLBlast", 1, performance]
            rows.append(row)
    if device is None:
        # CLBlast baseline also calculated through PortabilityTune()
        performance=get_baseline(args, db, n_k, geo)
        row=["CLBlast", 1, performance]
        rows.append(row)
    return rows

def display_graph(rows):
    rows_list = []

    # extract unique values
    names = sorted(set(row[0] for row in rows))
    kernels = sorted(set(row[1] for row in rows))

    result = [[None for _ in kernels] for _ in names]

    # populate the results
    for row in rows:
        a_index = names.index(row[0])
        b_index = kernels.index(row[1])
        result[a_index][b_index] = float(row[2])

    # each list is a row
    df1=pd.DataFrame(
            result, index=names, columns=kernels)
    df1.plot.bar(rot=0)
    plt.show()
    plt.clf()
    input("Close the figure and press a key to continue")
    print(df1)

def main():

    # get the configuration filename
    config_filename=sys.argv[1]

    # parse the configuration args
    args=parse_args(config_filename)

    # rows in form: METHOD, N_C, NORM, PERFORMANCE
    rows_list=[]

    if 'portable' in args['graph_types']:
        # device-portable tuning
        data_file = args['data_files'][0]
        rows=[]

        rows = get_lawson_results(rows, data_file, args)
        rows = get_ptuner_results(rows, data_file, args)
        display_graph(rows)

    if 'specific' in args['graph_types']:
        # do a tuning for a single device
        for device, data_file in zip(args['devices'], args['data_files']):
            rows=[]

            # Get Data for Clustering
            rows = get_lawson_results(rows, data_file, args)
            rows = get_ptuner_results(rows, data_file, args, device)
            display_graph(rows)
    
    if 'ipscatter' in args['graph_types']:
        # do a tuning for a scatterplot - returning all results, not just a geomean
        pass

main()
