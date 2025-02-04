import os
import argparse
import json5
from time import time
import torch
from entrypoints.Parameters import Parameters
from entrypoints.Kernel import Kernel
from utilities.seed import set_seed


def device_synchronize(input: dict):
    if input["device"] == "cuda":
        torch.cuda.synchronize()


def main(input: dict):

    if input.get("seed", None) != None:
        set_seed(input["seed"])

    device_synchronize(input)
    time_begin = time()
    para = Parameters(input)
    kernel = Kernel(para)
    device_synchronize(input)
    time_finish = time()
    print("-"*50+f"\nTime used for initialization = {time_finish-time_begin:.3f} s.\n"+"-"*50)

    device_synchronize(input)
    time_begin = time()
    kernel.run()
    device_synchronize(input)
    time_finish = time()
    print("-"*50+f"\nTime used for training = {time_finish-time_begin:.3f} s.\n"+"-"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.inputfile)
    with open(args.inputfile, "r") as f:
        inputfile = json5.load(f)

    main(inputfile)
