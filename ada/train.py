#!/usr/bin/env python3

import sys 
import os
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np

from config import *

def main(argv):
	args = parse_cl_args(argv)
	agent = set_up_sim(args)
	# TODO: Train agent and log results

	agent.train()

if __name__ == "__main__":
	main(sys.argv)