import itertools
import subprocess
import argparse
import pathlib
import shutil

kernel_level = {
    "xaxpy": 1,
    "xdot": 1,
    "xger": 2,
    "xgemv": 2,
    "transpose": 2,
    "padtranspose": 2,
    "invert": 2,
    "copy": 2,
    "pad": 2,
    "xgemm": 3,
    "xgemm_direct": 3,
}

kernels = list(kernel_level.keys())

DeepBench_inputs = [
[5124,	700,	2048],
[35,	700,	2048],
[5124,	700,	2560],
[35,	700,	2560],
[5124,	1500,	2048],
[35,	1500,	2048],
[5124,	1500,	2560],
[35,	1500,	2560],
[7680,	1,	2560],
[7680,	2,	2560],
[7680,	4,	2560],
[3072,	1,	1024],
[3072,	2,	1024],
[3072,	4,	1024],
[64,	1,	1216],
[512,	1,	500000],
[1024,	1,	500000],
[512,	2,	500000],
[1024,	2,	500000],
[512,	4,	500000],
[1024,	4,	500000],
[1024,	700,	512],
[7680,	1500,	2560],
[6144,	1500,	2048],
[4608,	1500,	1536],
[8448,	1500,	2816],
[3072,	1500,	1024],
[7680,	3000,	2560],
[6144,	3000,	2048],
[4608,	3000,	1536],
[8448,	3000,	2816],
[3072,	3000,	1024],
[7680,	6000,	2560],
[6144,	6000,	2048],
[4608,	6000,	1536],
[8448,	6000,	2816],
[3072,	6000,	1024],
[6144,	1,	2048],
[4608,	1,	1536],
[8448,	1,	2816],
[6144,	2,	2048],
[4608,	2,	1536],
[8448,	2,	2816],
[6144,	4,	2048],
[4608,	4,	1536],
[8448,	4,	2816],
[128,	1500,	1280],
[3072,	1500,	128],
[128,	1,	1024],
[3072,	1,	128],
[176,	1500,	1408],
[4224,	1500,	176],
[128,	1,	1408],
[4224,	1,	128],
[512,	1500,	2816],
[512,	1500,	2048],
[512,	1500,	2560],
[512,	1500,	1530],
[1024,	1500,	2816],
[1024,	1500,	2048],
[1024,	1500,	2560],
[1024,	1500,	1530],
[512,	1,	512],
[1024,	1,	512],
[512,	3000,	2816],
[512,	3000,	2048],
[512,	3000,	2560],
[512,	3000,	1530],
[1024,	3000,	2816],
[1024,	3000,	2048],
[1024,	3000,	2560],
[1024,	3000,	1530],
[512,	2,	512],
[1024,	2,	512],
[512,	6000,	2816],
[512,	6000,	2048],
[512,	6000,	2560],
[512,	6000,	1530],
[1024,	6000,	2816],
[1024,	6000,	2048],
[1024,	6000,	2560],
[1024,	6000,	1530],
[512,	4,	512],
[1024,	4,	512]]


def level_one_inputs():
    """
    Returns a list of inputs for level one kernels.
    """
    return list(itertools.combinations([2097152, 4194304, 16777216], 3))

def level_two_inputs():
    """
    Returns a list of inputs for level one kernels.
    """
    return list(itertools.combinations([512, 2048, 4096, 8192], 3))

def level_three_inputs():
    """
    Returns a list of inputs for level one kernels.
    """
    # return list(itertools.combinations([256, 512, 1024, 4096], 3))
    return DeepBench_inputs

input_by_level = {
    1: level_one_inputs,
    2: level_two_inputs,
    3: level_three_inputs,
}

inputs_by_level = {
    1: ["-n"],
    2: ["-m", "-n"],
    3: ["-m", "-n", "-k"],
}
def get_inputs_for_kernel(kernel: str) -> list:
    """
    Returns a list of inputs to be used in the tuning process.
    """
    return input_by_level[kernel_level[kernel]]()

def round_to_nearest(x: int, base: int):
    """ Round an integer x to the nearest non-zero multiple of base. """
    a = base * round(x/base)
    if a > 0:
        return a
    else:
        return base

def pad_input(kernel, input):
    """ Pad m and n to nearest 128 multiple for the xgemm kernel (this is what clblast will do automatically) """
    if kernel == "xgemm":
        return [round_to_nearest(input[0], 128), round_to_nearest(input[1], 128), input[2]]
    else:
        return input

def run_binary(kernel: str, input: list, device_id: int):
    """
    Runs the tuning binary for the given kernel with the input.
    """

    # pad inputs as needed
    input = pad_input(input)
    # collect the command line arguments
    binary_name = "./clblast_tuner_" + kernel + " -d " + str(device_id)
    input_mix = inputs_by_level[kernel_level[kernel]]
    input_list = [input + " " + str(value) + " " for input, value in zip(input_mix, input)]
    inputs_string = "".join(input_list)

    # run the binary
    args = binary_name + " " + inputs_string
    popen = subprocess.Popen(args.split(" "), stdout=subprocess.PIPE)
    try:
        popen.wait(timeout=60*15) # wait 15 minutes maximum
    except:
        pass

def move_json(kernel, input, directory):
    """
    Moves the json file to the appropriate directory and renames by input.
    """
    input_str = "".join(["_"+str(i) for i in input])
    for src_file in pathlib.Path(".").glob(f"*{kernel}*.json"):
        shutil.move(src_file, directory + "/" + pathlib.Path(src_file).stem + input_str + ".json")

def main():
    """
    For each kernel, get the relevant inputs and run the tuning binary on them.
    """
    # parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("directory", help="The directory to move json files to.", type=str)
    argparser.add_argument("device", help="The opencl device index.", type=int)
    args = argparser.parse_args()

    # create specified directory, if it does not already exist
    pathlib.Path(args.directory).mkdir(parents=True, exist_ok=True)

    kernels = ["xgemm", "xgemm_direct"]

    for kernel in kernels:
        inputs = get_inputs_for_kernel(kernel)
        for input in inputs:
            run_binary(kernel, input, args.device)
            move_json(kernel, input, args.directory)

if __name__ == "__main__":
    main()