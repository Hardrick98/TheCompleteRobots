import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--interaction","-i",type=str)

args = parser.parse_args()

path = args.interaction

folders = [f for f in os.listdir(path) if "." not in f]

for f in folders:
    data_path = os.path.join(path,f,"data")
    for i in os.listdir(data_path):
        if not i.endswith(".pkl"):
            os.remove(os.path.join(data_path,i))