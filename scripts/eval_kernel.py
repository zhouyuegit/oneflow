#!/usr/bin/env python3

import argparse
import re
import os

def get_cmd_args():
    arg_parser = argparse.ArgumentParser(description = \
'Evaluate the kernel and plot the result.')
    arg_parser.add_argument('kernel', help='the kernel name to be evaluated')
    arg_parser.add_argument('--eval_type', help='the type of evaluation, \
including parallelism...')
    return arg_parser.parse_args()

class Evaluator:
    def __init__(self, cmd_args):
        log_dir = '../build/eval/'
        bin_dir = '../build/bin/'
        if not os.path.isdir(log_dir):
            os.system('mkdir ' + log_dir)
        self.__cmd_args = cmd_args
        self.__kernel_exe = bin_dir + cmd_args.kernel + '_eval'
        self.__mp_log = log_dir + cmd_args.kernel + '_mp.log'
        self.__mpl_log = log_dir + cmd_args.kernel + '_mp_lite.log'
        self.__dp_log = log_dir + cmd_args.kernel + '_dp.log'
        self.__dpl_log = log_dir + cmd_args.kernel + '_dp_lite.log'

    def work(self):
        if self.__cmd_args.eval_type == 'parallelism':
            self.__parallelism();
        else:
            self.__parallelism();

    # Alias -->
    # mp: model parallel   dp: data parallel
    def __parallelism(self):
        m, n, k, max_parallel = 64, 8192, 1024 ,32
        os.system('rm -f ' + self.__mp_log)
        os.system('rm -f ' + self.__dp_log)

        # model parallel
        for i in range(max_parallel, 0, -1):
            single_wt = int(n / i)
            cmd="nvprof --devices 0 --metrics flop_sp_efficiency \
{} -in_rows={} -wt_unit={} >> {} 2>&1"\
            .format(self.__kernel_exe, m, single_wt, self.__mp_log)
            os.system(cmd)
            print('Evaluating fully_connected_kernel, model parallelism '+ str(i))
        os.system('grep flop_sp_efficiency {} > {}'\
        .format(self.__mp_log, self.__mpl_log))

        # data parallel
        for i in range(max_parallel, 0, -1):
            in_rows = int(m / i)
            cmd="nvprof --devices 0 --metrics flop_sp_efficiency \
{} -in_rows={} -wt_unit={} >> {} 2>&1"\
            .format(self.__kernel_exe, in_rows, n, self.__dp_log)
            os.system(cmd)
            print('Evaluating fully_connected_kernel, data parallelism '+ str(i))
        os.system('grep flop_sp_efficiency {} > {}'\
        .format(self.__dp_log, self.__dpl_log))

def main():
    cmd_args = get_cmd_args()
    evaluator = Evaluator(cmd_args)
    evaluator.work()

if __name__ == '__main__':
    main()
