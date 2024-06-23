from src.tuner import PortabilityTuner
from src.multi import MultiTuner
from src.db import TuningDatabase
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from enum import Enum

DEVICES = ["HD500", "Iris", "Mali", "Vega", "Quadro"]
DEVICES_INPUTS = [
    ("Intel(R) Gen9 HD Graphics NEO", 750),
    ("Intel(R) Gen9 HD Graphics NEO", 1150),
    ("Mali-G71", 5),
    ("Radeon RX Vega", 1630),
    ("Quadro P5000", 1733),
]
DEVICE_MAP = {k: v for k, v in zip(DEVICES_INPUTS, DEVICES)}

class Graph(Enum):
    """
    Enumerates the types of graphs that can be made.
    """
    PORTABLE = "portable"
    SPECIFIC = "specific"


def make_graphs(args: dict, db: TuningDatabase):
    """
    Makes the graphs specified in the configuration file.
    """
    for i in range(len(args["graph_types"])):
        if Graph.PORTABLE.value == args["graph_types"][i].lower():
            pTunerDevicePortableBar(
                args, db, args["graph_titles"][i], args["graph_labels"][i])
        if Graph.SPECIFIC.value == args["graph_types"][i].lower():
            pTunerDeviceSpecificBar(
                args, db, args["graph_titles"][i], args["graph_labels"][i])

    # save the graphs to a pdf
    # adapted from devgarg05
    pdf = PdfPages(args["graph_path"])
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format='pdf')
    pdf.close()



def pTunerDeviceSpecificBar(args: dict, db: TuningDatabase, title: str, ylabel: str):
    """
    Generates a bar graph that compares Portability Tuning results (device specific).
    """

    # find the best kernel combinations for each device
    device_results = dict()

    for device in args["devices"]:
        results = list()
        for k in args["num_kernels"]:
            tuner = PortabilityTuner(
                args["kernel"], [device], args["inputs"], args["arguments"], k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
            results.append(tuner.best_runtime)
        # run baseline tuning
        print("\n\n\n\n Running baseline tuning \n\n\n")
        tuner = PortabilityTuner(
            args["kernel"], [device], args["baseline_inputs"], args["baseline_arguments"], k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
        best_runtime = tuner.runtimeOnSubset([device], args["inputs"], args["arguments"])
        print("\n\n\n\n Baseline tuning complete \n\n\n")
        results.append(best_runtime)
        device_results[device] = results

    print(device_results)

    # consolidate the results into a dataframe
    index = [device[0] for device in args["devices"]]
    columns = [k for k in args["num_kernels"]]
    columns.append("CLBlast")
    dfa = pd.DataFrame(device_results.values(), index=index, columns=pd.Index(columns))

    # plot the runtimes for each device
    dfa.plot.bar(rot=0)
    plt.suptitle(title)
    plt.gca().get_legend().set_title("Number of Variants")
    plt.xlabel("Device")
    plt.ylabel(ylabel)
    if args["oracle_adjust"]:
        plt.ylim(bottom=1.0)
    fig = plt.gcf()
    plt.plot()


def pTunerDevicePortableBar(args: dict, db: TuningDatabase, title: str, ylabel: str):
    """
    Generates a bar graph that compares Portability Tuning results (device specific).
    """

    # find the best kernel combinations, device portable
    outputs = dict()
    up = dict()
    low = dict()
    tuners = list()
    for k in args["num_kernels"]:
        tuner = MultiTuner(args["kernel"], [device for device in args["devices"]], args["inputs"], args["arguments"], k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"], num_runs=10)
        # PortabilityTuner(args["kernel"], [device for device in args["devices"]], args["inputs"], args["arguments"], k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
        for device in args["devices"]:
            runtime, coverage = tuner.runtimeOnSubset([device], args["inputs"], args["arguments"], dct=False, coverage=True)
            upper, median, lower = runtime
            if device not in outputs:
                outputs[device] = list()
                up[device] = list()
                low[device] = list()
            outputs[device].append(median)
            up[device].append(upper-median)
            low[device].append(median-lower)
            # tuners.append(tuner)

    # run baseline tunings
    cover_inputs = {('Intel(R) Gen9 HD Graphics NEO', 750): 43,
                    ('Intel(R) Gen9 HD Graphics NEO', 1150): 55,
                    ('Mali-G71', 5): 60,
                    ('Radeon RX Vega', 1630): 64,
                    ('Quadro P5000', 1733): 64}
    
    clblast_params = { ('Intel(R) Gen9 HD Graphics NEO', 750): '1,4,1,1,16,16,64,8,8,64,0,0,0,0,2,2',
 ('Intel(R) Gen9 HD Graphics NEO', 1150): '1,16,1,1,16,16,64,8,8,64,0,0,0,0,4,1',
 ('Mali-G71', 5): '0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,1',
 ('Radeon RX Vega', 1630): '0,1,32,2,16,16,64,8,8,64,0,0,0,0,4,4',
 ('Quadro P5000', 1733): '0,1,32,2,8,8,64,16,16,64,1,1,0,0,4,4' }

    for device in args["devices"]:
        tuner = MultiTuner(args["kernel"], [device], args["inputs"], args["baseline_arguments"], 1, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"], cover_inputs=cover_inputs, num_runs=1, baseline_inputs=args["baseline_inputs"])
        # PortabilityTuner(args["kernel"], [device], args["baseline_inputs"], args["baseline_arguments"], k, args["tuning_metric"], args["minimum_coverage"], db, args["tuning_time"], args["parallel_evals"])
        runtime, coverage = tuner.runtimeOnSubset([device], args["inputs"], args["arguments"], dct=False, coverage=True)
        upper, median, lower = runtime
        outputs[device].append(median)
        up[device].append(upper-median)
        low[device].append(median-lower)
        tuners.append(tuner)

    # consolidate the results into a dataframe
    results = [outputs[device] for device in args["devices"]]
    index = [DEVICE_MAP[device] for device in args["devices"]]
    columns = [k for k in args["num_kernels"]]
    columns.append("1 - CLBlast")
    dfa = pd.DataFrame(results, index=index, columns=columns)

    print(dfa)
    print(low)
    print(up)
    print(outputs)


    a = 0
    for device in args['devices']:
        print(f"Device: {device}")
        print(f"Best Runtimes: {outputs[device]}")
        print(f"Upper Confidence Bounds: {up[device]}")
        print(f"Lower Confidence Bounds: {low[device]}")
        print(f"Best Parameters: {tuners[a].optimal_settings}")
        print(f"Best Parameters: {tuners[a].output}")
        a += 1

    dfa.plot.bar(rot=0, yerr=[(low[device], up[device]) for device in args["devices"]])

    # plot the runtimes for each device
    # title the legend with the name "Number of Variants"
    plt.gca().get_legend().set_title("Number of Variants")
    # plt.suptitle(title)
    plt.xlabel("Device")
    plt.ylabel(ylabel)
    if args["oracle_adjust"]:
        plt.ylim(bottom=1.0)
    fig = plt.gcf()
    plt.plot()

    # breakpoint()