from src.metrics import TuningMetric
from src.tuner import PortabilityTuner
from src.multi import MultiTuner
from src.db import TuningDatabase
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from enum import Enum
from src.db import TuningDatabase, Subset
from src.poly import PolyTuner, CT, PT, BT
import seaborn as sns

DEVICES = ["HD500", "Iris", "Mali", "Vega", "Quadro"]
DEVICES_INPUTS = [
    ("Intel(R) Gen9 HD Graphics NEO", 750),
    ("Intel(R) Gen9 HD Graphics NEO", 1150),
    ("Mali-G71", 5),
    ("Radeon RX Vega", 1630),
    ("Quadro P5000", 1733),
]
DEVICE_MAP = {k: v for k, v in zip(DEVICES_INPUTS, DEVICES)}

def device_name(device):
    return DEVICE_MAP[device]

def tuning_metric_eval(tuning_metric, runtimes_dct):
    return tuning_metric.evaluate_metric(list(runtimes_dct.values()), None, None, all_info=runtimes_dct)

def plot_dev_specific(args, subset):
    # get the tuning metric
    tuning_metric = args['tuning_metric']

    # get the baseline results by device
    runtimes = dict() # store list of runtimes by device
    columns = ['CLBlast']
    for k in args['num_kernels']:
        columns.append(f"PortabilityTune K={k}")

    for device in args['devices']:
        
        device_nm = device_name(device)

        specific_k = dict()

        # deepcopy dictionary "args"
        args2 = dict()
        for key in args:
            args2[key] = args[key]
        args2['devices'] = [device]

        # create the subset for the baseline
        subset_baseline = subset.make_subset(args['kernel'], [device], args['inputs'], args['arguments'])        

        # get the baseline parameters
        baseline = BT(args2, subset_baseline, 1, num_runs=1)
        baseline.do()

        # store the baseline results by device
        runtimes[device_nm] = list()

        # get the device baseline runtime
        dbr = tuning_metric_eval(tuning_metric, subset_baseline.evaluate(baseline.get_expected_parameters())[0])
        runtimes[device_nm].append(dbr)

        # get the specific parameters
        # device portable parameters
        for k in args['num_kernels']:
            r = []
            dev_specific = PT(args2, subset_baseline, int(k), num_runs=30)
            dev_specific.do()
            for params in dev_specific.get_all_params():
                dev_specific_runtime, c = subset_baseline.evaluate(params)
                dpr = tuning_metric_eval(tuning_metric, dev_specific_runtime)
                r.append(dpr)
            runtimes[device_nm].append(r)
    
    # process the dictionary
    processed_data = []
    for name, value in runtimes.items():
        for method, val in zip(columns, value):
            if method == 'CLBlast':
                processed_data.append([name, method, val])
            else:
                for v in val:
                    processed_data.append([name, method, v])

    df = pd.DataFrame(processed_data, columns=['Device', 'Method', 'Runtime'])
    print(df)
    # print all unique 'method' in df
    print(df['Method'].unique())
    print(runtimes)

    # plot the graph
    # ax = df.plot.bar(rot=0)
    sns.set(font_scale=1)
    sns.set_style("ticks")
    # plt.style.use('ggplot')
    plt.grid(False)

    b = sns.barplot(data=df, x='Device', y='Runtime', hue='Method')
    #b.set_xlabel("Device",fontsize=30)
    #b.set_ylabel("Geomean Slowdown Over Oracle",fontsize=20)
    # plt.gca().get_legend().set_title("Number of Variants")
    plt.xlabel("Device")
    plt.ylabel("Geomean Slowdown Over Oracle")
    if args["oracle_adjust"]:
        plt.ylim(bottom=1.0)
    fig = plt.gcf()
    # plt.show()
    plt.tight_layout()
    plt.savefig("graphs/specific_xgemm.pdf", format='pdf')
    plt.clf() # clear the plot


    # plt.plot()

def plot_dev_portable(args, subset):

    # get the tuning metric
    tuning_metric = args['tuning_metric']

    portable_k = dict()

    # device portable parameters
    for k in args['num_kernels']:
        dev_portable = PT(args, subset, int(k), num_runs=30)
        dev_portable.do()
        portable_k[k] = dev_portable

    baseline_d = dict()

    # get the baseline results by device
    runtimes = dict() # store list of runtimes by device
    columns = ['CLBlast']
    for k in args['num_kernels']:
        columns.append(f"PortabilityTune K={k}")

    for device in args['devices']:
        
        device_nm = device_name(device)

        # deepcopy dictionary "args"
        args2 = dict()
        for key in args:
            args2[key] = args[key]
        args2['devices'] = [device]

        # create the subset for the baseline
        subset_baseline = subset.make_subset(args['kernel'], [device], args['inputs'], args['arguments'])        
        # get the baseline parameters
        baseline = BT(args2, subset_baseline, 1, num_runs=1)
        baseline.do()
        # store the baseline results by device
        baseline_d[device] = baseline

        runtimes[device_nm] = list()

        # get the device baseline runtime
        dbr = tuning_metric_eval(tuning_metric, subset_baseline.evaluate(baseline.get_expected_parameters())[0])
        runtimes[device_nm].append(dbr)

        # get the device portable runtime
        for k in args['num_kernels']:
            r = []
            # get the device portable runtime
            for params in portable_k[k].get_all_params():
                dev_portable_runtime, c = subset_baseline.evaluate(params)
                print(dev_portable_runtime)
                print(params)
                print(subset_baseline)
                dpr = tuning_metric_eval(tuning_metric, dev_portable_runtime)
                r.append(dpr)
            # store the device portable runtimes
            runtimes[device_nm].append(r)
    
    # process the dictionary
    processed_data = []
    for name, value in runtimes.items():
        for method, val in zip(columns, value):
            if method == 'CLBlast':
                processed_data.append([name, method, val])
            else:
                for v in val:
                    processed_data.append([name, method, v])

    df = pd.DataFrame(processed_data, columns=['Device', 'Method', 'Runtime'])
    print(df)
    print(runtimes)

    # plot the graph
    # ax = df.plot.bar(rot=0)
    sns.set(font_scale=1)
    sns.set_style("ticks")
    # plt.style.use('ggplot')
    plt.grid(False)
    b = sns.barplot(data=df, x='Device', y='Runtime', hue='Method')
    #b.set_xlabel("Device",fontsize=30)
    #b.set_ylabel("Geomean Slowdown Over Oracle",fontsize=20)
    # plt.gca().get_legend().set_title("Number of Variants")
    plt.xlabel("Device")
    plt.ylabel("Geomean Slowdown Over Oracle")
    if args["oracle_adjust"]:
        plt.ylim(bottom=1.0)
    fig = plt.gcf()
    plt.tight_layout()
    plt.savefig("graphs/portable_xgemm.pdf", format='pdf')
    plt.clf()  # clear plot

