from src.tuner import PortabilityTuner
from src.db import Entry
import statistics
from collections import defaultdict
import random


class MultiTuner():
    def __init__(self, kernel, devices, inputs, arguments, num_outputs, tuning_metric, tolerance, db, tuning_time, parallel_evals, select=0, num_runs=30, cover_inputs=None, baseline_inputs=None) -> None:
        self.runs = []
        self.optimal_settings = {}  
        for _ in range(num_runs):
            # deep copy tuple
            dev = devices[:]
            try:
                # get a random shuffle of the devices, if selecting at least 1 randomly
                if select >= 1:
                    shuffler = list(dev)
                    random.shuffle(shuffler)
                    dev = shuffler[:select]
                    print("shuffled devices", dev, "from", devices, "to", shuffler)
                print(f"Running tuning run {_ + 1}/{num_runs}")
                tune = PortabilityTuner(kernel, dev, inputs, arguments, num_outputs, tuning_metric, tolerance, db, tuning_time, parallel_evals, cover_inputs=cover_inputs, baseline_inputs=baseline_inputs)
                self.runs.append(tune)
            except ZeroDivisionError as e:
                print(kernel, dev, inputs, arguments, num_outputs, tuning_metric, tolerance, db, tuning_time, parallel_evals)
                print(f"Failed to run tuning run {_ + 1}/{num_runs} due to error " + str(e))
                continue
        self.expected = statistics.median([tune.best_runtime for tune in self.runs])
        # pick the settings that gave the expected runtime, or the ones closest to it
        min_distance = float('inf')
        for tune in self.runs:
            distance = abs(tune.best_runtime - self.expected)
            if distance < min_distance:
                min_distance = distance
                self.optimal_settings = tune.optimal_settings
            if tune.best_runtime == self.expected:
                self.optimal_settings = tune.optimal_settings
                break            
        self.deviation = statistics.stdev([tune.best_runtime for tune in self.runs]) if num_runs > 1 else 0
        self.upper = statistics.mean([tune.best_runtime for tune in self.runs]) + 1.96 * self.deviation / num_runs**0.5 if num_runs > 1 else self.expected
        self.lower = statistics.mean([tune.best_runtime for tune in self.runs]) - 1.96 * self.deviation / num_runs**0.5 if num_runs > 1 else self.expected
        self.runtimes = [tune.best_runtime for tune in self.runs]
        self.parameters = [tune.output for tune in self.runs]

    def getDB(self):
        return self.runs[0].getDB()

    def runtimeOnSubset(self, devices, inputs, arguments, db=None, coverage=False, speedup=False, parameters=None, tuning_metric=None, all=False, dct=False, all_runs=False):
        """
        Return a list of the runtimes for this subset, one per random tuning run
        """

        if dct is not True:
              if coverage:
                  # for every run, get its performance on the subset
                  runtimes = []
                  coverages = []
                  for t in self.runs:
                      try:
                          r, c = t.runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, parameters=parameters, tuning_metric=tuning_metric, all=all, dct=dct)
                          runtimes.append(r) # a list of dicts
                          coverages.append(c)
                      except:
                          continue
                  # get the median runtime of all on the subset
                  median = statistics.median(runtimes) 
                  # get the upper confidence interval
                  upper_confidence = statistics.mean(runtimes) + 1.96 * statistics.stdev(runtimes) / len(runtimes)**0.5 if len(runtimes) > 1 else median
                  # get the lower confidence interval
                  lower_confidence = statistics.mean(runtimes) - 1.96 * statistics.stdev(runtimes) / len(runtimes)**0.5 if len(runtimes) > 1 else median

                  return (upper_confidence, median, lower_confidence), coverages
              else:
                  raise NotImplementedError
              
        if dct is True:
            if parameters:
                if coverage:
                    r, c = self.runs[0].runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, parameters=parameters, tuning_metric=tuning_metric, all=all, dct=False)
                    runtimes = []
                    for val in r:
                        runtimes.append((val, val, val))
                    return runtimes, c
                else:
                    r = self.runs[0].runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, parameters=parameters, tuning_metric=tuning_metric, all=all, dct=False)
                    runtimes = []
                    for val in r:
                        runtimes.append((val, val, val))
                    return runtimes
            else:
                if not all_runs:
                    runtimes = []
                    coverages = []
                    for t in self.runs:
                        try:
                            if coverage:
                                r, c = t.runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, tuning_metric=tuning_metric, all=all, dct=dct)
                            else:
                                r = t.runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, tuning_metric=tuning_metric, all=all, dct=dct)
                            runtimes.append(r) # a list of dicts
                            coverages.append(c)
                        except:
                            continue

                    
                    print("printing runtimes", runtimes)

                    runtimes_default = defaultdict(list)
                    for d in runtimes:
                        for k, v in d.items():
                            print(k, v)
                            runtimes_default[str(k)].append(v)

                    print(runtimes_default)
                    #print(len(self.runs))
                    #print(self.runs)

                    runtimes = []
                    
                    for k, v in runtimes_default.items():
                        # upper confidence interval
                        upper_confidence = statistics.mean(v) + 1.96 * statistics.stdev(v) / len(v)**0.5 if len(v) > 1 else statistics.mean(v)
                        # median
                        median = statistics.median(v)
                        # lower confidence interval
                        lower_confidence = statistics.mean(v) - 1.96 * statistics.stdev(v) / len(v)**0.5 if len(v) > 1 else statistics.mean(v)

                        tup = tuple([upper_confidence, median, lower_confidence])
                        
                        runtimes.append(tup)

                        # runtimes.append(tuple([upper_confidence, median, lower_confidence])) 
                        print(f"[MULTI] {k}: {upper_confidence}, {median}, {lower_confidence} ({len(v)} runs)")
                    
                    """
                    runtimes = []
                    for k, v in runtimes_default.items():
                        runtimes.extend(v)
                    """

                    print(runtimes)

                    return runtimes, coverages
                if all_runs:
                    runtimes = []
                    coverages = []
                    for t in self.runs:
                        try:
                            if coverage:
                                r, c = t.runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, tuning_metric=tuning_metric, all=all, dct=dct)
                            else:
                                r = t.runtimeOnSubset(devices, inputs, arguments, db=db, coverage=coverage, speedup=speedup, tuning_metric=tuning_metric, all=all, dct=dct)
                            runtimes.append(r) # a list of dicts
                            coverages.append(c)
                        except:
                            continue

                    runtimes_default = defaultdict(list)
                    for d in runtimes:
                        # print(d)
                        for k, v in d.items():
                            runtimes_default[str(k)].append(v)

                    runtimes = []
                    for k, v in runtimes_default.items():
                        runtimes.append(v)
                    
                    print(runtimes)
                    

                    return runtimes, coverages
                    

        else:
            raise NotImplementedError