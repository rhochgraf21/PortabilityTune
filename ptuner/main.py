"""
Runs a PortabilityTune() instance and creates a graph.
"""

import ast
import yaml
import sys

from src.metrics import TuningMetric
from src.db import TuningDatabase
from src.plot import make_graphs


def parse_args(config_filename):
    """
    Parses the configuration file.
    """

    with open(config_filename, "r") as config_file:
        config_data = yaml.safe_load(config_file)

    # parse tuning metrics
    config_data["tuning_metric"] = TuningMetric.get_metric(config_data["tuning_metric"])
    
    # parse output path2
    config_data["graph_path"] = sys.argv[2]

    # parse the PortabilityTune() arguments
    try:
        config_data["devices"] = tuple(tuple(i) for i in config_data["devices"])
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

    # make the graphs
    make_graphs(args, db)


if __name__ == "__main__":
    main()
