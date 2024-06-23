from src.db import TuningDatabase, Subset
from src.poly import PolyTuner, CT, PT, BT
from src.graphs import plot_dev_portable, plot_dev_specific
"""
Runs a PortabilityTune() instance and creates a graph.
"""

import ast
import yaml
import sys

from src.metrics import TuningMetric
from src.db import TuningDatabase
from src.plot import make_graphs

import sys
sys.path.append('../tuning_kernels')
import utils
import models
import dataset
import kernel_tree_classifier
import src.db as db
import itertools
import decide

level_2 = itertools.product([512,2048,4096,8192],[512,2048,4096,8192])
level_3 = itertools.product([256,512,1024,4096],[256,512,1024,4096],[256,512,1024,4096])

def parse_args(config_filename):
    """
    Parses the configuration file.
    """

    with open(config_filename, "r") as config_file:
        config_data = yaml.safe_load(config_file)

    # parse tuning metrics
    config_data["tuning_metric"] = TuningMetric.get_metric(config_data["tuning_metric"])
    
    # parse output path2
    # config_data["graph_path"] = sys.argv[2]

    # parse the PortabilityTune() arguments
    try:
        config_data["devices"] = tuple(tuple(i) for i in config_data["devices"])
        config_data["db_devices"] = tuple(tuple(i) for i in config_data["db_devices"])
        config_data["inputs"] = tuple(tuple(i) for i in config_data["inputs"])
        config_data["arguments"] = tuple(tuple(i) for i in config_data["arguments"])
        config_data["baseline_inputs"] = tuple(tuple(i) for i in config_data["baseline_inputs"])
        config_data["baseline_arguments"] = tuple(tuple(i) for i in config_data["baseline_arguments"])
    except:
        raise ValueError("Invalid config.yml arguments. See the docs for example formatting.")

    if not config_data["devices"]:
        raise ValueError("PortabilityTune() requires at least one device.")

    if not config_data["inputs"]:
        raise ValueError("PortabilityTune() requires at least one input.")

    if not config_data["baseline_inputs"]:
        raise ValueError("Baseline requires at least one input.")

    return config_data


def main():
    """
    Runs a PortabilityTune() instance and creates graph(s).
    """
    
    # get the configuration filename
    config_filename = sys.argv[1]

    # parse the configuration args
    args = parse_args(config_filename)

    # read the database file
    db = TuningDatabase(args)
    d = args['database_path']
    i = args['inputs']
    args['database_path'] = args['clblast_path']
    args['inputs'] = args['baseline_inputs']
    db_cl = TuningDatabase(args)
    args['inputs'] = i
    args['database_path'] = d

    # create the subset for tuning
    subset = db.make_subset(args['kernel'], args["devices"], args["inputs"], args["arguments"])
    #subset_cl = db_cl.make_subset(args['kernel'], args["db_devices"], args["baseline_inputs"], args["arguments"])

    #print(subset.dataframe)

    # run the tuning
    # models.KMeans
    # args['devices'] = args['db_devices']
    #tune = PT(args, subset_cl, 5, num_runs=3)
    tune = CT(models.DecisionTree, 'scale', args, subset, 10, num_runs=3)
    tune.do()
    
    # args['devices'] = [('Radeon RX Vega', 1630)]

    print(tune.get_expected_parameters())
    # subset = subset.make_subset(args['kernel'], args["devices"], args["inputs"], args["arguments"], tune.get_expected_parameters())

    print(subset)
    #subset_cl = subset_cl.make_subset(args['kernel'], args["db_devices"], args["baseline_inputs"], args["arguments"], tune.get_expected_parameters())
    #print(subset.dataframe)

    # decide.build_tree(subset, tune.get_expected_parameters())
    # print(tune.expected)

    # decide.build_tree(subset_cl, tune.get_expected_parameters())
    # decide.symbolic_reg(subset_cl, tune.get_expected_parameters())

    """
    # evaluate the tuning results
    print("parameters: ", tune.get_expected_parameters().str_repr())
    result = subset.evaluate(tune.get_expected_parameters())
    print(result)
    print(tune.expected)
    """

    # make the graphs
    plot_dev_portable(args, subset)
    plot_dev_specific(args, subset)

    # graphs to make:
    # 1. fleet performance
    # 2. device specific
    # 3. portable

if __name__ == "__main__":
    main()
