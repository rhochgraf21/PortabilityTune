# Graph Scripts

This folder contains the script used to generate the graphs in the ICS submission.

Files:

- `pickles/`: for storing tuning results (as tuner objects). tunings can take a long time, it's worth having.
- `config.yml`: set the configuration options for a tuning: inputs, devices, runtime adjustment to oracle, which type of graph to generate, etc.
- `device_specific.py`: an older version of `results.py` kept for reference (non-functional).
- `results.py`: usage; `python results.py config.yml`. Used for generating many of the result figures in the table. more or less, set the inputs/devices/graph type, and run the script. It may take a while to run. Supported graph types:
  - `fleet`: tunes over multiple devices for the collective *task rate*. outputs a bar graph of tasks/second versus tuning method (Decision Tree, PortabilityTune, etc), fig 4 in ICS submission.
  - `specific`: tunes specific to each given device in `devices`, but across all `inputs` for varying number of variant. (similar to fig 2 in ICS submission.)
  - `portable`: tunes across all given devices and inputs; (similar to fig 3 in ICS submission.) 
  - `ipscatter`: show the performance of each tuning method on every input, for each given device. fig 5 in ICS submission.
  - `tuning_time`: generates a graph showing tuning performance vs tuning time. data taken from a prior recorded tuning (ignores inputs, devices, etc). fig 8 in ICS submission.
  - `unseenarchci`: generates a graph showing the performance of a set of portability-tuned variants (median shown, 95% CI drawn) vs. the proportion of devices and inputs covered at at least that level of performance. fig 7 in the ICS submission.
  - `boxplot`: a boxenplot showing the distribution of all performance results (not in ICS submission)