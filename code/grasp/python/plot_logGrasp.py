'''

'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt


from grasp.vis_grasp import TrainTestMonitor as TTMon
from tkinter import Tk #from Tkinter import Tk
from tkinter import filedialog  as tkFileDialog# import tkFileDialog

def main(args):
    if not os.path.exists(args.log_dir):
        tkroot = Tk()
        tkroot.withdraw()
        args.log_dir = tkFileDialog.askdirectory(title='select log folder', initialdir='../logs', mustexist=True)
        tkroot.destroy()
    assert(os.path.exists(args.log_dir))
    ttm = TTMon(args.log_dir,plot_extra=True,plot_train=True, stages = 1) #plot_extra=args.plot_extra!=0
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('-d','--log_dir',type=str, default='', help='log folder')
    parser.add_argument('-e','--plot_extra',type=int, default=0, help='plot training accuracy and test loss')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)



