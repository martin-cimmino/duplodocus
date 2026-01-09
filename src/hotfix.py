import glob
import os
import random
import shutil


def move_hotnodes(d):
    fs = [_ for _ in glob.glob("/mnt/raid0/jacc_hotnodes/*") if _.endswith(".jsonl.zst")]
    for src in fs:
        dst = os.path.join(d, os.path.basename(src))
        shutil.move(src, dst)


def move_output(d):
    fs = [_ for _ in glob.glob("/mnt/raid0/jacc_output/*") if _.endswith(".jsonl.zst")]
    for src in fs:
        dst = os.path.join(d, os.path.basename(src))
        shutil.move(src, dst)


if __name__ == "__main__":
    SUFFIX = "part_%s" % random.randint(0, 2**64)
    OUTPUT_DIR = os.path.join("/mnt/raid0/jacc_output/", SUFFIX)
    os.mkdir(OUTPUT_DIR)
    HOTNODE_DIR = os.path.join("/mnt/raid0/jacc_hotnodes/", SUFFIX)
    os.mkdir(HOTNODE_DIR)
    while True:
        move_hotnodes(HOTNODE_DIR)
        move_output(OUTPUT_DIR)
