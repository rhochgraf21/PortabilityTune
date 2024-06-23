import statistics
import random
import opentuner
from opentuner import MeasurementInterface, EnumParameter, Result
from opentuner import ConfigurationManipulator

# import the folders where tuning_kernels and portabilitytune are.
import sys
sys.path.append("../tuning_kernels")
import utils
import models
import dataset
import kernel_tree_classifier
import src.db as db
import json

ERROR = 10**10

class PolyTuner():
    def __init__(self, args, subset, k, num_runs=30, select=0) -> None:
        """
        Base Class for performing Portability Tunings.

        Args:
        args: A dictionary of arguments for the tuning run.
        k: The number of outputs to tune for.
        num_runs: The number of tuning runs to perform.
        select: The number of devices to randomly select for each tuning run. 
                If 0, then all devices will be used.
        """
        self.kernel = args["kernel"]
        self.devices = args["devices"]
        self.inputs = args["inputs"]
        self.arguments = args["arguments"]
        self.__subset = subset
        self.subset = subset
        self.num_outputs = k
        self.tolerance = args["minimum_coverage"]
        self.tuning_metric = args["tuning_metric"]
        self.tuning_time = args["tuning_time"]
        self.oracle_adjust = args["oracle_adjust"]
        self.result_db_path = args["results_path"]
        self.num_runs = num_runs
        self.select = select
        self.parameters = []
        self.runtimes = []
        self.best_runtime = None
    
    def do(self):
        """
        Runs tunings with the given arguments.
        """
        
        # check if tuning is necessary (i.e. the tuning run has already been done and is in the database)
        # if so, get the tuning results from the database
        results = self.get_from_json()
        if results is not None:
            self.optimal_settings = results["median_parameters"]
            self.expected = results["median_runtime"]
            self.runtimes = results["all_runtimes"]
            self.parameters = [db.Parameters(a) for a in results["all_parameters"]]
            self.__calculate_statistics()
            return

        # otherwise, run the tuning

        for _ in range(self.num_runs):
            dev = self.select_devices()
            if dev != self.devices:
                self.subset = self.__subset.make_subset(self.kernel, dev, self.inputs, self.arguments)
            try:
                print(f"Running tuning run {_ + 1}/{self.num_runs}")
                runtime, parameters = self.tune()
                self.runtimes.append(runtime)
                self.parameters.append(parameters)
            except AssertionError as e:
                print(self.kernel, dev, self.inputs, self.arguments, self.num_outputs)
                print(f"Failed to run tuning run {_ + 1}/{self.num_runs} due to error " + str(e))
                continue
        self.__calculate_statistics()  # calculate statistics about the tuning run
        self.write_to_json()  # write the tuning results to a json file
    
    def __calculate_statistics(self):
        """
        Calculates key statistics about the tuning run.
        """
        # now that we have all the runtimes, get the expected runtime and confidence bands
        self.expected = statistics.median(self.runtimes)
        # pick the settings that gave the expected runtime, or the ones closest to it
        min_distance = float('inf')
        for i in range(len(self.runtimes)):
            distance = abs(self.runtimes[i] - self.expected)
            if distance < min_distance:
                min_distance = distance
                self.optimal_settings = self.parameters[i]
            if self.runtimes[i] == self.expected:
                self.optimal_settings = self.parameters[i]
                break
        # print(len(self.optimal_settings))
        print(self.runtimes)
        self.deviation = statistics.stdev(self.runtimes) if len(self.runtimes) > 1 else 0
        self.upper = statistics.mean(self.runtimes) + 1.96 * self.deviation / self.num_runs**0.5 if len(self.runtimes) > 1 else self.expected
        self.lower = statistics.mean(self.runtimes) - 1.96 * self.deviation / self.num_runs**0.5 if len(self.runtimes) > 1 else self.expected

    def select_devices(self):
        """
        Randomly selects the devices to use for this tuning run.
        """
        # get a random shuffle of the devices, if selecting at least 1 randomly
        if self.select >= 1:
            shuffler = list(self.devices)
            random.shuffle(shuffler)
            shuffler = shuffler[:self.select]
            return shuffler
        # otherwise, use all devices
        return self.devices
    
    def tune(self):
        """
        Runs a single tuning run with the given arguments.
        
        Returns:
        runtime: The runtime of the tuning run.
        parameters: The parameters that gave the runtime.
        """
        pass

    def get_expected_parameters(self):
        """
        Returns the parameters that gave the expected runtime.
        """
        return self.optimal_settings
    
    def get_all_params(self):
        """
        Returns all selected parameter configurations
        """
        return self.parameters

    def __convert_tuple_to_lst(self, tup):
        """
        Recursively converts all tuples to lists.
        """
        if isinstance(tup, tuple):
            return [self.__convert_tuple_to_lst(i) for i in tup]
        elif isinstance(tup, tuple) or isinstance(tup, list) or isinstance(tup, set) or isinstance(tup, frozenset) or isinstance(tup, dict):
            a = []
            for i in tup:
                a.append(self.__convert_tuple_to_lst(i))
            return a
        else:
            return tup
    

    def get_db_key(self):
        """
        Returns the key for the tuning run in the database.
        """
        # assemble the key
        key = dict()
        key["tuning_type"] = str(type(self).__name__)
        if hasattr(self, "method"):
            key["method"] = str(self.method.__name__)
        if hasattr(self, "norm"):
            key["norm"] = str(self.norm)
        key["kernel"] = self.kernel
        key["devices"] = self.__convert_tuple_to_lst(self.devices)
        key["inputs"] = self.__convert_tuple_to_lst(self.inputs)
        key["arguments"] = self.__convert_tuple_to_lst(self.arguments)
        key["tuning_time"] = self.tuning_time
        key["tuning_metric"] = str(type(self.tuning_metric).__name__)
        key["minimum_coverage"] = self.tolerance
        key["num_runs"] = self.num_runs
        key["select"] = self.select
        key["oracle_adjust"] = self.oracle_adjust
        key['k'] = self.num_outputs
        return key

    def write_to_json(self):
        """
        Write the tuning results to a json file.
        """
        # KEY: kernel, devices, arguments, tuning time, tuning metric, tolerance, num runs, select, oracle adjust
        # VALUE: tuning results: expected parameters, expected parameters runtime, all runtimes, all parameters
        key = self.get_db_key()

        # assemble the values
        value = dict()
        value["median_parameters"] = self.optimal_settings.str_repr()
        value["median_runtime"] = self.expected
        value["all_runtimes"] = self.runtimes
        value["all_parameters"] = [a.str_repr() for a in self.parameters]

        # write to the database
        dct = dict()
        dct["tuning_arguments"] = key
        dct["tuning_results"] = value

        with open(self.result_db_path, 'r') as file:
            db = json.load(file)
            # if the key is already in the database, update the value
            sections = db["sections"]
            sections.append(dct)
            with open(self.result_db_path, 'w') as file:
                json.dump(db, file, indent=4)
        
    def get_from_json(self):
        """
        Get the tuning results from a json file.
        """
        key = self.get_db_key()

        with open(self.result_db_path) as file:
            db = json.load(file)
            for section in db["sections"]:
                if section["tuning_arguments"] == key:
                    return section["tuning_results"]
            return None


class CT(PolyTuner):
    
    def __init__(self, method, norm, args, subset, k, num_runs=30, select=0) -> None:
        """
        Class for performing Portability Tunings using Clustering/DT.

        Args:
        method: The method to use for tuning, e.g. models.DecisionTree or models.KMeans 
        args: A dictionary of arguments for the tuning run.
        k: The number of outputs to tune for.
        num_runs: The number of tuning runs to perform.
        select: The number of devices to randomly select for each tuning run. 
                If 0, then all devices will be used.
        """
        super().__init__(args, subset, k, num_runs, select)
        self.method = method
        self.norm = norm


    def tune(self):
        """
        Runs a single tuning run with the given arguments.
        """
        model = self.method
        train, test = self.subset.get_dataset_format()
        train = kernel_tree_classifier.normalize_data(train, self.norm)
        m = model(train, self.num_outputs)
        labels = set(c for c in m.classes)
        runtime = None
        try:
            # rank by tuning metric if possible, otherwise use geometric mean
            # runtime = self.tuning_metric.evaluate_metric(utils.get_perfect_errors_for(labels, test).dropna(), None, None)
            runtime_dct, coverage = self.subset.evaluate(db.Parameters(labels))
            runtime = self.tuning_metric.evaluate_metric(list(runtime_dct.values()), None, None, all_info=runtime_dct)
        except AssertionError as e:
            # if we can't use the tuning metric, use geometric mean
            print(f"Could not use tuning metric due to error {e}, using geometric mean instead.")
            # print(utils.get_perfect_errors_for(labels, test))
            runtime = 1/statistics.geometric_mean(utils.get_perfect_errors_for(labels, test).dropna())
        # print(db.Parameters(labels))
        return runtime, db.Parameters(labels)
    


class PT(PolyTuner, MeasurementInterface):

    # self, kernel, devices, inputs, arguments, num_outputs, tuning_metric, tolerance,
    # db, tuning_time, parallel_evals, tune=True, cover_inputs=None, 
    # baseline_inputs=None, baseline_arguments=None

    def __init__(self, args, subset, k, num_runs=30, select=0) -> None:
        super().__init__(args, subset, k, num_runs, select)

        # initialize variables
        self.parallel_compile = True
        self.max_coverage = 0
        # self.tolerance = args["minimum_coverage"] * len(self.inputs) * len(self.devices)

        # Tuning Time and OpenTuner args
        sys.argv = list()
        sys.argv.append("--parallelism=" + str(8))
        sys.argv.append("--stop-after=" + str(self.tuning_time))
        sys.argv.append("--no-dups")
        sys.argv.append("--database=sqlite://")

        # parse opentuner arguments
        self.__argparser = opentuner.default_argparser()
        MeasurementInterface.__init__(self, self.__argparser.parse_args())

    def tune(self):
        self.__tune()
        return self.runtime, db.Parameters(self.output.values())

    def __tune(self):
        """
        Starts the OpenTuner tuning run.
        """
        from opentuner.tuningrunmain import TuningRunMain
        # return PortabilityTuner.main(self.__argparser.parse_args())
        return TuningRunMain(self, self.__argparser.parse_args()).main()

    def manipulator(self):
        """
        Define the search space of the tuning using all possible parameter combinations.
        """
        manipulator = ConfigurationManipulator()

        for _ in range(self.num_outputs):
            manipulator.add_parameter(EnumParameter(
                'Param' + str(_), self.subset.get_all_params()))
            
        return manipulator

    def save_final_config(self, configuration):
        """
        Called at the end of tuning. Saves the best configuration found.
        """
        self.output = configuration.data
        self.runtime = self.best_runtime
        # reset the best runtime for the next tuning run
        self.best_runtime = None
        self.settings = None
        self.max_coverage = 0
    
    def run(self, desired_result, input, limit):
        """
        Simulates a kernel run from the dataset.
        """
        cfg = desired_result.configuration.data
        optimal_settings = dict()
        counts = 0

        # select the best runtime from any of the parameter sets for each entry
        params = []
        for pname in cfg:  # for each parameter name in the configuration
            param = cfg[pname]  # get the parameter from its name
            params.append(param)

        # get the best runtime for each entry
        optimal_settings, coverage = self.subset.evaluate(db.Parameters(params))
        counts = coverage

        # handle first run
        if self.best_runtime is None:
            self.settings = optimal_settings
            self.best_runtime = ERROR  # large constant
        
        # evaluate the performance
        runtime = self.tuning_metric.evaluate_metric(list(optimal_settings.values(
            )), list(self.settings.values()), self.best_runtime, all_info=optimal_settings)    # evaluate the runtimes on the metric

        # if the number of entries evaluated is less than the tolerance, return an error
        # print(counts)
        if counts > self.max_coverage:
           self.max_coverage = counts

        if counts < self.tolerance:
            # print(f"coverage was {counts} but needed {self.tolerance}, max so far is {self.max_coverage}")
            return opentuner.resultsdb.models.Result(state='ERROR', time=ERROR)
        
        # if the runtime is better than the best known runtime, update the best runtime and optimal settings
        if (self.best_runtime is None or runtime < self.best_runtime): # and not len(optimal_settings.values()) < len(self.__pruned_data):
            self.best_runtime = runtime
            self.settings = optimal_settings
            return opentuner.resultsdb.models.Result(state='OK', time=runtime)
        else:
            return opentuner.resultsdb.models.Result(state='ERROR', time=ERROR)
        
class BT(PolyTuner):
    """ 
    Runs Baseline Tunings.
    A Baseline tuning is a tuning run that tunes for the best runtime on a single device for a predetermined input.
    """

    level_1_input = (4194304)
    level_2_input = (2048, 2048)
    level_3_input = (1024, 1024, 1024)

    level_1 = ["xaxpy", "xdot"]
    level_2 = ["xgemv", "xger", "invert", "copy", "pad", "transpose", "padtranspose"]
    level_3 = ["xgemm", "xgemm_direct"]

    def __init__(self, args, subset, k, num_runs=1, select=0) -> None:
        super().__init__(args, subset, k, num_runs, select)
        if len(self.devices) != 1:
            raise ValueError("Baseline Tuning requires exactly one device.")
        if k != 1:
            raise ValueError("Baseline Tuning requires exactly one output.")
        if select != 0:
            raise ValueError("Baseline Tuning cannot select devices.")
        if self.kernel in self.level_1:
            self.baseline_inputs = self.level_1_input
        elif self.kernel in self.level_2:
            self.baseline_inputs = self.level_2_input
        elif self.kernel in self.level_3:
            self.baseline_inputs = self.level_3_input

    def tune(self):
        """
        Runs a single tuning run with the given arguments.
        """
        # get the best kernel for the given input, with at least self.tolerance coverage on the given self.subset
        best_params = self.subset.get_best_params(self.baseline_inputs, self.tolerance)
        best_params = db.Parameters(best_params)
        # get the runtime for the best kernel
        runtime_dct, coverage = self.subset.evaluate(best_params)
        runtime = self.tuning_metric.evaluate_metric(list(runtime_dct.values()), None, None, all_info=runtime_dct)
        return runtime, best_params