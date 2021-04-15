

import os
import sys
import argparse
import glog as logger
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.autograd
from grasp.graspdata import  DefaultTestSet, DefaultTrainSet,

from grasp.models import *
from grasp.traintester_grasp import TrainTester
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True


parser = argparse.ArgumentParser(sys.argv[0], description='DeepFaceAlignment')

parser.add_argument('-e','--epoch',type=int,default=50,
                    help='training epochs')
parser.add_argument('--batch-size',type=int,default=16,
                    help='training batch size')
parser.add_argument('--test-batch-size',type=int,default=8,
                    help='testing batch size')
parser.add_argument('--lr',type=float,default=0.0001,
                    help='learning rate')
parser.add_argument('--momentum',type=float,default=0.5,
                    help='Solver momentum')
parser.add_argument('--weight-decay',type=float,default=1e-5,
                    help='weight decay')
parser.add_argument('--log-dir',type=str,default='logs/tmp',
                     help='log folder to save training stats as numpy files')
parser.add_argument('--verbose_per_n_batch',type=int,default=2,
                    help='log training stats to console every n batch (<=0 disables training log)')


args = parser.parse_args(sys.argv[1:])
args.script_folder = os.path.dirname(os.path.abspath(__file__))

args.cuda = torch.cuda.is_available() 

print(str(args))
sys.stdout.flush()

#### Main
print(torch.cuda.current_device())

net = GraspNet()#, init_path = args.init_path)

loss_fn = GraspLoss()
 

solver = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay) #rmsprop  Adam

runner = TrainTester(net=net, solver=solver, total_epochs=args.epoch,
    cuda=args.cuda, log_dir=args.log_dir, verbose_per_n_batch=args.verbose_per_n_batch)

kwargs = {'num_workers':8, 'pin_memory':False} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader( DefaultTrainSet(), batch_size=args.batch_size, shuffle=False, **kwargs)
train_loader = torch.utils.data.DataLoader( DefaultTrainSet(), batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(DefaultTestSet(), batch_size=args.test_batch_size, shuffle=False, **kwargs)


runner.run(train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn)
logger.info('Done!')

