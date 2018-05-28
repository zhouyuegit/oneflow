#!/usr/bin/env python3

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

def get_cmd_args():
    arg_parser = argparse.ArgumentParser(description = 'Plots the result of evaluation.')
    arg_parser.add_argument('kernel', help='the kernal name of evaluation result')
    return arg_parser.parse_args()


def main():
    args = get_cmd_args()
    fname = args.kernel + '_mp_lite.log'
    with open(fname, 'r') as fh:
        idx, model_ratio_y = 1, []
        for ef_line in fh.readlines():
            match_obj = re.search('\s(\S+)\.\S+%$', ef_line)
            idx += 1
            model_ratio_y.append(int(match_obj.group(1)))

    fname = args.kernel + '_dp_lite.log'
    with open(fname, 'r') as fh:
        idx, data_ratio_y = 1, []
        for ef_line in fh.readlines():
            match_obj = re.search('\s(\S+)\.\S+%$', ef_line)
            idx += 1
            data_ratio_y.append(int(match_obj.group(1)))

    parallelism_x = np.flip(np.arange(1, idx, 1), 0)
    plt.plot(parallelism_x, data_ratio_y, 'bo', parallelism_x, model_ratio_y, 'r^')
    plt.axis([0, 35, 0, 100])
    plt.gca().invert_xaxis()
    plt.xlabel('Parallelism degree')
    plt.ylabel('Achieved GFLOPS ratio to the peak performance')
    plt.title('Data parallel(blue) vs model parallel(red) for fully connected layer')
    plt.show()


if __name__ == '__main__':
    main()

