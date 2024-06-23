import pandas as pd
import csv
import sys
import random

sys.path.append("../tuning_kernels")

import utils
import models
import dataset
import kernel_tree_classifier
from hashlib import md5

# define the inputs PTuner looks for
VALID_INPUTS = ('arg_m', 'arg_n', 'arg_k')

def alt_hash(x):
    return int(md5(x.encode()).hexdigest(), 16)

class Parameters:
    """
    A Parameters object stores the parameters of a tuning run.
    Parameters are stored as a list of tuple of parameter values.
    """
    def __init__(self, parameters):
        self.parameters = []
        for config in parameters:
            values = None
            if type(config) is str:
                # determine if the parameters are in the form of a string split by _ or ,
                if "_" in config:
                    values = config.split("_")
                else:
                    values = config.split(",")
            if type(config) is tuple:
                values = config
            self.parameters.append(tuple(int(p) for p in values))
    
    def str_repr(self, separator=','):
        """
        Returns a string representation of the parameters.
        The values will be separated by the given separator.
        """
        parameters = []
        for config in self.parameters:
            config_str = ""
            for val in config:
                config_str += str(val) + separator
            if len(config_str) > 0:
                config_str = config_str[:-1]
            parameters.append(config_str)
        return parameters
    
    def __str__(self) -> str:
        return f"Parameters({self.parameters})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Subset:
    """
    A Subset is a subset of the tuning database that is within the scope of the tuning query.
    """
    def __init__(self, kernel, devices, inputs, arguments, entries, params=None):
        self.kernel = kernel
        self.devices = devices
        self.inputs = inputs
        self.arguments = arguments
        self.entries = entries
        self.params = params
        self.dataframe = Subset.__make_df(entries)
        # create a dict of entries by device
        self.entries_by_dev = {device: [entry for entry in self.entries if entry.device == device] for device in self.devices}
        # create a set of all parameter names contained in this subset
        self.parameter_names = list()
        for entry in self.entries:
            for param in entry.parameter_names:
                if param not in self.parameter_names:
                    self.parameter_names.append(param)
    
    def get_parameter_names(self):
        """
        Returns the parameter names contained in this subset.
        """
        return self.parameter_names

    def get_random_entry(self, device=None):
        """
        Returns a random entry from the subset.
        """
        if device is None:
            # select a random entry from self.entries
            return random.choice(self.entries)
        else:
            # select a random entry from self.entries with the given device
            if device not in self.entries_by_dev:
                return None
            entries = self.entries_by_dev[device]
            if len(entries) <= 0:
                return None
            return random.choice(entries)

    def get_all_params(self):
        """
        Returns all parameters contained with this Subset.
        """
        return TuningDatabase.get_all_params(self.entries)
    
    def make_subset(self, kernel, devices, inputs, arguments, params=None):
        """
        Creates a Subset from the current subset.
        """
        pruned_entries = []
        for entry in self.entries:
            if TuningDatabase.in_subset(entry, kernel, devices, inputs, arguments):
                if params is not None:
                    new_entry = entry.prune_params(params)
                    if new_entry is not None:
                        pruned_entries.append(new_entry)
                else:
                    pruned_entries.append(entry)
        return Subset(kernel, devices, inputs, arguments, pruned_entries, params=params)

    def __make_df(entries):
        """
        Make a dataframe from the entries.
        """
        new_rows = []
        for entry in entries:
            if not isinstance(entry, Entry):
                raise Exception("Entries must be Entry objects.")
            new_rows.extend(entry.get_dct_repr())
        df = pd.DataFrame(new_rows)
        # use pd.concat to add the new rows to the dataframe
        return df

    def get_dataset_format(self) -> dataset.DataSet:
        """
        Convert Subset to a DataSet format.
        Adapted from tuning_kernels/dataset.py
        """
        # print(self.dataframe)
        pivot = self.dataframe.pivot_table(index=['m', 'k', 'n', 'batch', 'label'], columns=['config'], values='rate_s').reset_index()
        features = pivot[['m', 'k', 'n', 'batch', 'label']]
        values = pivot.drop(['m', 'k', 'n', 'batch', 'label'], axis=1)
        normalized = values.div(values.max(axis=1), axis=0)
        values = values.fillna(0)
        normalized = normalized.fillna(0)
        ds = dataset.DataSet(features, normalized, values)
        # return the dataset twice, since we don't have a test set
        # TODO: add test set splitting, if desired
        return (ds, ds)
    
    def get_entries_format(self) -> list:
        """
        Convert Subset to a list of Entry objects.
        """
        return self.entries
    
    def __iter__(self):
        return iter(self.entries)

    def get_best_params(self, baseline_input, tolerance):
        """
        Returns the best kernel for a given input.
        The best parameters are the ones with the lowest runtime on the baseline input, where
        the params also achieve a coverage >= tolerance on the inputs.
        """
        best_params = None
        best_runtime = None
        for entry in self.entries:
            if entry.input == baseline_input:
                for params, runtime in entry.results.items():
                    if best_runtime is None or runtime < best_runtime and self.evaluate(Parameters([params]))[1] >= tolerance:
                        best_runtime = runtime
                        best_params = [params]
        if best_params is None:
            raise Exception("No best parameters found for the given baseline input that achieve the desired coverage.")
        return best_params

    def evaluate(self, parameters: Parameters) -> tuple:
        """
        Returns the runtimes dict and coverage for a given set of parameters.
        """
        # create a dict of the performance results
        # key: entry, value = runtime
        performance_dct = dict()
        for entry in self.entries:
            # print(parameters.str_repr(), entry.results.keys())
            best_runtime_for_entry = None
            for param in parameters.str_repr():
                if param in entry.results:
                    if best_runtime_for_entry is None or entry.results[param] < best_runtime_for_entry:
                        best_runtime_for_entry = entry.results[param]
            if best_runtime_for_entry is not None:
                performance_dct[entry] = best_runtime_for_entry
        # get the coverage (as a % of the total entries)
        coverage = len(performance_dct.keys()) * 100 / len(self.entries)
        return performance_dct, coverage

    def __len__(self):
        return len(self.entries)

    def __str__(self) -> str:
        return f"Subset({self.kernel}, {self.devices}, {self.inputs}, {self.arguments}, {self.entries})"
    
    def __repr__(self) -> str:
        return self.__str__()


class TuningDatabase:
    """
    A TuningDatabase reads in a database of tuning results and makes Entry objects.
    """
    def __init__(self, args) -> None:
        self.__args = args  # store the args to know about oracle_adjust, arguments, etc.
        self.__data_filename = self.__args['database_path']
        print(f"[TUNER] Reading database from {self.__data_filename}")
        self.unpruned_data = self.__read_database_df(self.__data_filename)

    def make_subset(self, kernel, devices, inputs, arguments):
        """ 
        Returns a list of Entry objects corresponding to the requested subset
        """
        subset = self.__prune_data(kernel, devices, inputs, arguments)
        if len(subset) <= 0:
            raise Exception("No entries found in database for the given tuning query. Check your tuning arguments.")
        return subset
    
    def makeSubset(self, kernel, devices, inputs, arguments):
        """ For compatibility """
        return self.make_subset(kernel, devices, inputs, arguments)

    def __prune_data(self, kernel, devices, inputs, arguments):
        """
        Reads and prunes the dataset from the filename to include only valid entries that are within the scope of the
        tuning query.
        """
        pruned_deserialized_data = list()
        for entry in self.unpruned_data:
            obj = None
            if isinstance(entry, Entry):
                obj = entry
            else:
                obj = Entry(entry, self.__args)
            if TuningDatabase.in_subset(obj, kernel, devices, inputs, arguments):
                pruned_deserialized_data.append(obj)
        subset = Subset(kernel, devices, inputs, arguments, pruned_deserialized_data)
        return subset

    @staticmethod
    def in_subset(entry, kernel, devices, inputs, arguments):
        """
        Determines if an Entry is in the portability tuning query.
        :param entry: an Entry object
        :return: True if the entry is in the query bounds, False otherwise
        """        
        # for each argument:
        #  check the argument name is in the entry
        #  check the entry value is in the argument values of the query
        for arg in arguments:
            t = False
            for earg in entry.arguments:
                if arg[0] in earg:
                    for val in arg[1]:
                        if str(val) in earg:
                            t = True
                        for a in earg:
                            if a.replace('.','',1).isdigit():
                                if float(a) == val:
                                    t = True
            if not t:
                return False
        return kernel == entry.kernel and entry.device in devices and entry.input in inputs

    @staticmethod
    def get_all_params(entries):
        """
        Returns all parameters from the pruned Entries.
        """
        all_params = set()
        for entry in entries:
            for paramset in entry.results.keys():
                if paramset not in all_params:
                    all_params.add(paramset)
        return all_params

    def __read_database_df(self, filename):
        """ 
        Read a database from a given filename.
        """
        # store json in df
        df = pd.read_json(filename)
        df = df['sections']  # remove the sections header
        return df
    

class Entry(object):
    def __init__(self, entry, args, kernel_family=None, inputs=None, device=None, parameter_names=None, results=None, unadj_results=None, arguments=None):
        """
        Creates an Entry object.

        An entry is made up of the following:

        The kernel family of the entry,
        The inputs of the entry as a tuple (m,n,k) (n and k are optional)
        The device of the entry, as a tuple of its name and clock speed
        The results of the entry, as a dict in form {"parameters": runtime}
        The arguments of the entry, as a list of tuples in form [(arg_name, arg_value)]
        """
        if kernel_family is not None:
            self.kernel = kernel_family
            self.input = inputs
            self.device = device
            self.parameter_names = parameter_names if parameter_names else args["parameter_names"]
            self.results = results if results else {}
            self.unadjusted_results = unadj_results if unadj_results else {}
            self.arguments = arguments if arguments else []
        else:
            # set the attributes
            self.kernel = entry['kernel_family']
            self.input = list()
            for arg in VALID_INPUTS:
                if arg in entry:
                    self.input.append(int(entry[arg]))
            self.device = (entry['clblast_device_name'],
                            int(entry['device_core_clock']))
            self.input = tuple(self.input)
            self.arguments = list()
            self.parameter_names = entry['parameter_names']
            for arg in args["arguments"]:
                if arg[0] in entry:
                    self.arguments.append((arg[0], entry[arg[0]]))
            # set the results
            self.results = dict()
            best_result = None
            for result in entry['results']:
                if best_result is None or result[1] < best_result:
                    best_result = result[1]
                self.results[result[0]] = result[1]
            # adjust relative to oracle if requested
            self.unadjusted_results = {}
            for result in entry['results']:
                self.unadjusted_results[result[0]] = result[1]
            if args["oracle_adjust"]:
                for result in entry['results']:
                    self.results[result[0]] = result[1] / best_result 

    def get_best_param(self, adj=False):
        """
        Returns the best parameter configuration for this entry.
        """
        best_param = None
        best_runtime = None
        a = self.results if adj else self.unadjusted_results
        for param, runtime in a.items():
            if best_runtime is None or runtime < best_runtime:
                best_runtime = runtime
                best_param = param
        return best_param, best_runtime

    def get_dct_repr(self):
        """
        Returns a list of dcts representing the entries.
        
        Each dct is in the form:
        | m | n | k | batch | label | mean_ms | rate_s | parameter_a | parameter_b | parameter_x....| 
        """
        results = []
        # precompute m, n, k, batch, label
        m, n, k = None, None, None
        if len(self.input) == 1:
            n = self.input[0]
        elif len(self.input) == 2:
            m = self.input[0]
            n = self.input[1]
        else:
            m = self.input[0]
            n = self.input[1]
            k = self.input[2]
        for config, runtime in self.unadjusted_results.items():
            dct = {}
            dct['m'] = m
            dct['n'] = n
            dct['k'] = k
            dct['batch'] = 1
            device = f"{self.device[0]} {self.device[1]}"
            dct['label'] = alt_hash(device)
            dct['mean_ms'] = runtime
            dct['rate_s'] = 1000 / runtime
            dct['config'] = config
            parameter_names = self.parameter_names
            parameter_values = [int(a) for a in config.split(',')]
            for i in range(len(parameter_names)):
                dct[parameter_names[i]] = parameter_values[i]
            results.append(dct)
        return results

    def __str__(self) -> str:
        return f"Entry({self.kernel}, {self.input}, {self.device})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def prune_params(self, parameter_variants):
        """
        Prunes the parameters to only include the ones in the Entry.
        Returns a new Entry object with only the parameters in parameter_variants.

        :param: parameter_variants: a list of parameter variants to keep, as Parameter objects
        """
        new_results = dict()
        new_results_unadj = dict()
        for paramset, runtime in self.results.items():
            if paramset in parameter_variants.str_repr():
                new_results[paramset] = runtime
        for paramset, runtime in self.unadjusted_results.items():
            if paramset in parameter_variants.str_repr():
                new_results_unadj[paramset] = runtime
        if len(new_results) <= 0:
            return None
        return Entry(None, None, kernel_family=self.kernel, inputs=self.input, device=self.device, parameter_names=self.parameter_names, results=new_results, unadj_results=new_results_unadj, arguments=self.arguments)