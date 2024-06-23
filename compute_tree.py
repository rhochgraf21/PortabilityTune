import argparse

import sklearn.tree as sktree

import dataset
import models
import utils


def load_results(filenames):
    assert len(filenames) > 0
    data = dataset.load_cached(filenames[0])
    for i in range(len(filenames) - 1):
        data = dataset.combine(data, dataset.load_cached(filenames[i]))

    data = dataset.DataSet(
        features=data.features,
        normalized=data.normalized.applymap(lambda x: x
                                                     if x > 0.9 else 0),
        values=data.values,
    )
    return data


def prune_kernels(data, num_kernels):
    model = models.PCAKMeans(data, num_kernels)
    return model.classes


def augment_features(feats):
    def get_feats(m, k, n, b, l):
        return (m, k, n, b, m * n, b * m * n, k * n, k * m, m / k, n / k,
                m * n / k, l)

    feats['m * n'] = feats['m'] * feats['n']
    feats['batch * m * n'] = feats['batch'] * feats['m'] * feats['n']
    feats['k * n'] = feats['k'] * feats['n']
    feats['k * m'] = feats['k'] * feats['m']
    feats['m / k'] = feats['m'] / feats['k']
    feats['n / k'] = feats['n'] / feats['k']
    feats['m * n / k'] = feats['m'] * feats['n'] / feats['k']

    feats['m % 2'] = feats['m'] % 2
    feats['m % 4'] = feats['m'] % 4
    feats['m % 8'] = feats['m'] % 8
    feats['k % 2'] = feats['k'] % 2
    feats['k % 4'] = feats['k'] % 4
    feats['k % 8'] = feats['k'] % 8
    feats['n % 2'] = feats['n'] % 2
    feats['n % 4'] = feats['n'] % 4
    feats['n % 8'] = feats['n'] % 8

    return feats, get_feats


def get_targets_for_given_configs(dataset, labels):
    limited_norm = dataset.normalized[labels]
    return limited_norm.idxmax(axis=1)


def train_classifier(data, kernels):
    #model = sktree.DecisionTreeClassifier(random_state=0)
    model = sktree.DecisionTreeClassifier(random_state=0,
                                          #max_features=4,
                                          max_depth=6,
                                          #min_samples_split=3,
                                          #min_samples_leaf=4,
                                          )
    augment_features(data.features)
    labels = get_targets_for_given_configs(data, kernels)
    model.fit(data.features, labels)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames',
                        nargs='+',
                        metavar='file',
                        help='CSV file containin benchmark results')
    parser.add_argument('--num-kernels',
                        type=int,
                        default=8,
                        help='Number of kernels to select')
    args = parser.parse_args()

    data = load_results(args.filenames)
    kernels = prune_kernels(data, args.num_kernels)
    model = train_classifier(data, kernels)
    func = utils.tree_to_code('select', model, data.features.columns, kernels)
    print(func)
    tiles = set()
    for kernel in kernels:
        tiles.add(tuple(kernel.split(':')[0].split('_')))

    for tile in sorted(tiles):
        print('generate_matmul_impl(_sources {0[0]} {0[1]} {0[2]})'.format(tile))

if __name__ == '__main__':
    main()
