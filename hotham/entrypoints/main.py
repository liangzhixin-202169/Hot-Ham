import os
import argparse
import json5
import yaml
from time import time
import torch
import torch.distributed as dist
from ..entrypoints.Parameters import Parameters
from ..entrypoints.Kernel import Kernel
from ..utilities.seed import set_seed
# torch.multiprocessing.set_start_method("spawn", force=True)


def device_synchronize(input: dict):
    if input["device"] == "cuda":
        torch.cuda.synchronize()


def init_distribution(para: Parameters):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if para["device"] == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="gloo")
    else:
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")

    para["rank"] = rank
    para["local_rank"] = local_rank
    para["device"] = device
    para["world_size"] = dist.get_world_size()
    # para["lr"] = para["lr"]*para["world_size"]


def main():
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.inputfile)
    if args.inputfile.endswith(".json"):
        with open(args.inputfile, "r", encoding='utf-8') as f:
            input = json5.load(f)
    elif args.inputfile.endswith(".yaml"):
        with open(args.inputfile, "r", encoding='utf-8') as f:
            input = yaml.safe_load(f)

    # seet random seed
    if input.get("seed", None) != None:
        set_seed(input["seed"])

    # initialize model. read dataset
    device_synchronize(input)
    time_begin = time()
    para = Parameters(input)
    init_distribution(para)
    kernel = Kernel(para)
    device_synchronize(input)
    time_finish = time()
    if para.rank == 0:
        print("-"*50+f"\nTime used for initialization = {time_finish-time_begin:.3f} s.\n"+"-"*50)

    # run
    device_synchronize(input)
    time_begin = time()
    kernel.run()
    device_synchronize(input)
    time_finish = time()
    if para.rank == 0:
        print("-"*50+f"\nTime used for training = {time_finish-time_begin:.3f} s.\n"+"-"*50)

    # finish
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
