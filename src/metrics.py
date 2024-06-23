from abc import abstractmethod
from statistics import geometric_mean, harmonic_mean, mean, median
from scipy.stats import mannwhitneyu

"""
This file defines Cost Functions that can be used to evaluate utility of parameters.
"""

class TuningMetric():
    @abstractmethod
    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        pass

    @staticmethod
    @abstractmethod
    def get_names():
        pass
    
    @staticmethod
    def get_metric(metric_name):
        for subclass in TuningMetric.__subclasses__():
            if metric_name.lower() in subclass.get_names():
                return subclass()
        raise ValueError("Invalid metric name: " + metric_name)


class GeometricMean(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["geomean", "geometric_mean"]
    
    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        return geometric_mean([a for a in runtimes if a != 0 and a is not None and a != float('inf')])
    
class Median(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["median", "med"]
    
    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        return median(runtimes)

class DummyMetric(TuningMetric):
    
    @staticmethod
    @abstractmethod
    def get_names():
        return ["dummy", "dummy_metric"]
    
    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        return 1
    

class PerformancePortability(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["performance_portability", "perfport"]

    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        return harmonic_mean(runtimes)


class KruskalTest(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["mwu", "mannwhitneyu"]

    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        if current_best_runtime is None:
            return 10**10
        statistic, p_value = mannwhitneyu(runtimes, args)
        if p_value < 0.05:
                    return current_best_runtime - 1
        else:
            return current_best_runtime
        
class MaximizeCount(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["maximize_count", "maxcount"]

    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        count = 0
        for runtime in runtimes:
            if runtime < 1.176:
              # print(runtime)
              count += 1
        # print(count)
        if count > 0:
            # print(count)
            return 1/count

        else:
            return 10**10
        
class FleetPerformance(TuningMetric):

    @staticmethod
    @abstractmethod
    def get_names():
        return ["fleet", "fleetperf", "fleet_performance"]

    def evaluate_metric(self, runtimes: list, args: list, current_best_runtime: float, all_info=None):
        # [sum( # of device d / sum( # of inputs i ) for all inputs i ) for all devices d]
        """ 
        Returns the runtime-per-app for a fleet of devices.
        """
        if all_info is not None:
            devices = set([entry.device for entry in all_info])
            inputs = set([entry.input for entry in all_info])
            fleet = 0
            for device in devices:
                device_rate = 0
                app_time = 0
                device_times = set()
                for input in inputs:
                    for entry in all_info:
                        if entry.device == device and entry.input == input:
                            if all_info[entry] is not None:
                                app_time += all_info[entry]
                            device_times.add(all_info[entry])
                            # print("the runtime for input ", input, " on device ", device, " is ", all_info[entry])
                if app_time != 0:
                    device_rate += 1/app_time
                    # add in the radeon rx vega twice more to account for the three rx devices
                    #if device == ("Radeon RX Vega", 1630):
                    #    device_rate += 2/app_time
                    # print("the rate on device ", device, " is ", device_rate, "with app time ", app_time, " and device times ", device_times)
                fleet += device_rate * 1000 # multiply by 1000 to represent tasks/s not tasks/ms
            # print("the fleet performance is ", fleet)
            return 1/fleet
        else:
            raise Exception("Fleet Performance requires information about all inputs and devices")