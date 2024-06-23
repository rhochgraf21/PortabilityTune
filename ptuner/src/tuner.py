import opentuner
from opentuner import MeasurementInterface, EnumParameter, Result
from opentuner import ConfigurationManipulator
from itertools import product
from src.metrics import TuningMetric
from src.db import TuningDatabase
import sys

ERROR = 10**10

class PortabilityTuner(MeasurementInterface):
    def __init__(self, kernel, devices, inputs, arguments, num_outputs, tuning_metric, tolerance, db, tuning_time, parallel_evals, tune=True, cover_inputs=None, baseline_inputs=None, baseline_arguments=None):
        """
        Initializes a PortabilityTuner.
        """
        self.parallel_compile = True

        # Tuning Time and OpenTuner args
        sys.argv = list()
        sys.argv.append("--parallelism=" + str(parallel_evals))
        sys.argv.append("--stop-after=" + str(tuning_time))
        sys.argv.append("--no-dups")
        sys.argv.append("--database=sqlite://")

        # parse opentuner arguments
        self.__argparser = opentuner.default_argparser()
        super().__init__(self.__argparser.parse_args())
        
        # PortabilityTune arguments
        self.__kernel = kernel
        self.__devices = devices
        self.__inputs = inputs  
        self.__arguments = arguments
        self.__num_outputs = num_outputs
        self.__cover_inputs = cover_inputs
        self.__baseline_inputs = baseline_inputs
    
        # stores list of valid entries as Entry objects.
        self.__db = db
        self.__pruned_data = self.__db.makeSubset(self.__kernel, self.__devices, self.__inputs, self.__arguments)
        self.data = self.__pruned_data
        self.__all_params = TuningDatabase.get_all_params(self.__pruned_data)
        
        # evaluates the performance of a given set of parameters
        self.metric = tuning_metric
        
        # the minimum number of supported device-input pairs to be included
        self.tolerance = tolerance * len(self.__inputs) * len(self.__devices)
        self.best_runtime = None
        self.output = None
        self.optimal_settings = None  # stores the runtimes of each Entry

        self.max_coverage = 0

        # run the tuning
        if tune:
            self.tune()

    def __getstate__(self):
        """
        Called when the tuner is pickled. Sets the state of the tuner.
        """
        state = {}
        # pickle all the things
        state['kernel'] = self.__kernel
        state['devices'] = self.__devices
        state['inputs'] = self.__inputs
        state['arguments'] = self.__arguments
        state['num_outputs'] = self.__num_outputs
        state['db'] = self.__db
        state['pruned_data'] = self.__pruned_data
        state['all_params'] = self.__all_params
        state['metric'] = self.metric
        state['tolerance'] = self.tolerance
        state['best_runtime'] = self.best_runtime
        state['optimal_settings'] = self.optimal_settings
        state['output'] = self.output
        state['max_coverage'] = self.max_coverage
        return state
    
    def __setstate__(self, state):
        # unpickle all the things
        self.__kernel = state['kernel']
        self.__devices = state['devices']
        self.__inputs = state['inputs']
        self.__arguments = state['arguments']
        self.__num_outputs = state['num_outputs']
        self.__db = state['db']
        self.__pruned_data = state['pruned_data']
        self.__all_params = state['all_params']
        self.metric = state['metric']
        self.tolerance = state['tolerance']
        self.best_runtime = state['best_runtime']
        self.optimal_settings = state['optimal_settings']
        self.output = state['output']
        self.max_coverage = state['max_coverage']

    def tune(self):
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

        for _ in range(self.__num_outputs):
            manipulator.add_parameter(EnumParameter(
                'Param' + str(_), self.__all_params))
            
        return manipulator

    def save_final_config(self, configuration):
        """
        Called at the end of tuning. Saves the best configuration found.
        """
        self.output = configuration.data
        print("the optimal params were: ", self.output)
        print(self.__db)
        print(self.data)

    def run(self, desired_result: tuple, input, limit):
        """
        Simulates a kernel run from the dataset.
        """
        cfg = desired_result.configuration.data
        optimal_settings = dict()
        counts = 0

        # select the best runtime from any of the parameter sets for each entry
        for pname in cfg:  # for each parameter name in the configuration
            param = cfg[pname]  # get the parameter from its name
            for entry in self.__pruned_data:  # for each entry
                if param in entry.results:  # if the param set was run
                    runtime = entry.results[param]  # find the runtime
                    # print(param, entry, runtime)
                    if entry in optimal_settings:  # if the runtime < best known runtime for this entry, update it
                        if runtime < optimal_settings[entry]:
                            optimal_settings[entry] = runtime
                    else:
                        optimal_settings[entry] = runtime
                        counts += 1

        # handle first run
        if self.best_runtime is None:
            self.optimal_settings = optimal_settings
            self.best_runtime = ERROR  # large constant
        
        # evaluate the performance
        runtime = self.metric.evaluate_metric(list(optimal_settings.values(
            )), list(self.optimal_settings.values()), self.best_runtime, all_info=optimal_settings)    # evaluate the runtimes on the metric
        # print(runtime)

        """ If calculating baseline, then we need to remove all entries that are not in the baseline inputs from optimal_settings """
        adjusted = {}
        if self.__baseline_inputs is not None:
            # remove all entries that are not in the baseline inputs from optimal_settings
            for entry in self.optimal_settings:
                if entry.input in self.__baseline_inputs:
                    adjusted[entry] = self.optimal_settings[entry]
            # now calculate the runtime
            self.optimal_settings = adjusted
            runtime = self.metric.evaluate_metric(list(optimal_settings.values(
            )), list(self.optimal_settings.values()), self.best_runtime, all_info=optimal_settings)            

        # counts
        # print(counts, self.tolerance)
        # print("RUNTIME: ", runtime, "; COUNTS: ", counts, len(optimal_settings.values()), len(self.__pruned_data))

        # if the number of entries evaluated is less than the tolerance, return an error
        # print(counts)
        if counts > self.max_coverage:
           self.max_coverage = counts
        
        if self.__cover_inputs is not None:
            if counts < self.__cover_inputs[self.__devices[0]]:
                return opentuner.resultsdb.models.Result(state='ERROR', time=ERROR)

        if counts < self.tolerance:
            # print(f"coverage was {counts} but needed {self.tolerance}, max so far is {self.max_coverage}")
            return opentuner.resultsdb.models.Result(state='ERROR', time=ERROR)
        
        # if the runtime is better than the best known runtime, update the best runtime and optimal settings
        if (self.best_runtime is None or runtime < self.best_runtime): # and not len(optimal_settings.values()) < len(self.__pruned_data):
            self.best_runtime = runtime
            self.optimal_settings = optimal_settings
            return opentuner.resultsdb.models.Result(state='OK', time=runtime)
        else:
            # print("randomerror")
            return opentuner.resultsdb.models.Result(state='ERROR', time=ERROR)

    def getDB(self):
        """
        Returns the tuning database.
        """
        return self.__db

    def runtimeOnSubset(self, devices, inputs, arguments, db=None, coverage=False, speedup=False, parameters=None, tuning_metric=None, all=False, dct=False):
        """
        Gives the optimal runtime of this tuner's selected parameters on a given subset of data.
        """
        runtimes = list()
        runtimes_dict = dict()

        seen_devices = set()

        map = dict()

        params = self.output.values()

        if parameters is not None:
            if isinstance(parameters, dict):
                params = parameters.values()
            if isinstance(parameters, list):
                params = parameters
            print("using params", params)

        metric = self.metric if not tuning_metric else tuning_metric 
        subset=None

        if db is not None:
            subset = db.makeSubset(self.__kernel, devices, inputs, arguments)
        else:
            # make a new subset
            subset = self.__db.makeSubset(self.__kernel, devices, inputs, arguments)

        #print(params)

        #print([param for param in subset[0].results])

        #print(params[0] in subset[0].results)

        # for every entry, if the entry is for the device, find the best param for the entry, take that runtime
        print("printing entries")
        for entry in subset:
            # print(entry)
            # print([param for param in entry.results])

            if TuningDatabase.in_subset(entry, self.__kernel, devices, inputs, arguments):
                times = [entry.results[param]
                        for param in params if param in entry.results]
                if len(times) > 0:
                    if parameters is not None:
                        pass
                        # print(entry, min(times))
                    if speedup:
                        runtimes.append(1/min(times))
                        map[entry] = 1/min(times)
                        runtimes_dict[entry] = 1/min(times)
                    else:
                        runtimes.append(min(times))
                        map[entry] = min(times)
                        runtimes_dict[entry] = min(times)
                    if entry.device not in seen_devices:
                        seen_devices.add(entry.device)
                    #if min(times) < 1.1111:
                    #    print(entry, min(times))
                    # print(entry.input, entry.device, min(times))
        
        # if no runtimes, there was an error
        if len(runtimes) == 0:
            raise Exception("Data Mismatch")
        
        if all:
            if dct:
                if coverage:
                    return runtimes_dict, len(seen_devices)
                else:
                    return runtimes_dict
            else:
                if coverage:
                    return runtimes, len(seen_devices)
                else:
                    return runtimes
                

        print("evaluated to: ", metric.evaluate_metric(runtimes, None, None, all_info = map))
        if coverage:
            return metric.evaluate_metric(runtimes, None, None, all_info = map), len(seen_devices)
        else:
            print(runtimes_dict)
            return metric.evaluate_metric(runtimes, None, None, all_info = map)
        
    def runtimesForSubset(self, devices, inputs, arguments, db=None):
          """
          Gives the optimal runtimes of this tuner's selected parameters on a given subset of data.
          """
          runtimes = dict()

          subset=None

          if db is not None:
              subset = db.makeSubset(self.__kernel, devices, inputs, arguments)
          else:
              # make a new subset
              subset = self.__db.makeSubset(self.__kernel, devices, inputs, arguments)

          # make a new subset
          # subset = self.__db.makeSubset(self.__kernel, devices, inputs, arguments)

          # for every entry, if the entry is for the device, find the best param for the entry, take that runtime
          for entry in subset:
              if TuningDatabase.in_subset(entry, self.__kernel, devices, inputs, arguments):
                  times = [entry.results[param]
                          for param in self.output.values() if param in entry.results]
                  if len(times) > 0:
                      runtimes[entry] = min(times)
                      # print(entry, min(times))
                  else:
                      print("no times for ", entry)
                      # print(entry.input, entry.device, min(times))
          
          # if no runtimes, there was an error
          if len(runtimes) == 0:
              raise Exception("Data Mismatch")

          return runtimes