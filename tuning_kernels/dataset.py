import pandas as pd

import pickle
import os

from collections import namedtuple
from hashlib import md5

DataSet = namedtuple('DataSet', ['features', 'normalized', 'values'])


def alt_hash(x):
    return int(md5(x.encode()).hexdigest(), 16)

def load_from_csv(filename, args = None):
    data = pd.read_csv(filename)
    data['config'] = data.apply(lambda x: '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        x.GEMMK, x.KREG, x.KWG, x.KWI, x.MDIMA, x.MDIMC, x.MWG, x.NDIMB, x.NDIMC,
        x.NWG, x.SA, x.SB, x.STRM, x.STRN, x.VWM, x.VWN),axis=1)
    """
    data['config'] = data.apply(lambda x: '{}_{}_{}:{}_{}_{}'.format(
        x.row_tile, x.acc_tile, x.col_tile, x.wg_a, x.wg_b, x.wg_c),
                                axis=1)
    """
    data['label'] = data['label'].apply(lambda x: alt_hash(x)) # hash the device labels
    print(data)

    # now filter the data
    if args is not None:
        # filter so the Dataset only contains dataframes with the specified devices and kernel
        data = data[data['label'].isin([alt_hash(str(a[0]) + " " + str(a[1])) for a in args['devices']])]
        # data = data[data['kernel_family'].isin(args['kernels'])]
        # filter so the Dataset only contains dataframes with the specified input. the tuple (data['m'], data['n'], data['k']) must be in args['inputs']
        data = data[data.apply(lambda x: (int(x['m']), int(x['n']), int(x['k'])) in args['inputs'], axis=1)]

    # TODO: undo GEMM hardcode
    pivot = data.pivot_table(index=['m', 'k', 'n', 'batch', 'label'], columns=['config'], values='rate_s').reset_index() # mean_ns for lawson data
    features = pivot[['m', 'k', 'n', 'batch', 'label']]
    values = pivot.drop(['m', 'k', 'n', 'batch', 'label'], axis=1)
    normalized = values.div(values.max(axis=1), axis=0)
    values = values.fillna(0)
    normalized = normalized.fillna(0)
    #print("output normalized")
    #normalized.to_csv('out.csv')
    return features, normalized, values


def _save_to_pickle(filename, features, normalized, values):
    df_dict = {
        'features': features,
        'normalized': normalized,
        'values': values,
    }
    with open(filename, 'wb') as f:
        pickle.dump(df_dict, f)


def _load_from_pickle(filename):
    with open(filename, 'rb') as f:
        dfs = pickle.load(f)
    return DataSet(dfs['features'], dfs['normalized'], dfs['values'])


def load_cached(filename, args=None):
    root_file, _ = os.path.splitext(filename)
    pickle_file = root_file + '.pkl'
    pickle_exists = os.path.exists(pickle_file + "1") # disable loading from cache
    if pickle_exists:
        return _load_from_pickle(pickle_file)
    else:
        a, b, c = load_from_csv(filename, args=args)
        _save_to_pickle(pickle_file, a, b, c)
    return DataSet(a, b, c)

def filter_dataset(dataset, args):
    # filter so the Dataset only contains dataframes with the specified kernel, devices, inputs, and arguments

    pass


def combine(dataset1, dataset2):
    feat = pd.concat([dataset1.features,
                      dataset2.features]).reset_index(drop=True)
    norm = pd.concat([dataset1.normalized,
                      dataset2.normalized]).reset_index(drop=True)
    values = pd.concat([dataset1.values,
                        dataset2.values]).reset_index(drop=True)
    return DataSet(feat, norm, values)


def from_values(feat, norm, val):
    return DataSet(feat.reset_index(drop=True), norm.reset_index(drop=True),
                   val.reset_index(drop=True))
