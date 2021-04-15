'''
visualization.py in ineepFaceAlignment

author  : wmou, modified based on pointoctnet of cfeng
created : 2/18/18 5:15 AM
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, proj3d

from skimage import io

from sklearn import metrics
import pdb
from sklearn import metrics



class TrainTestMonitor(object):

    def __init__(self, log_dir, plot_loss_max=4., plot_extra=False, plot_train=False, stages = 2):
        assert(os.path.exists(log_dir))
        print(log_dir)

        stats_test_acc = np.load(os.path.join(log_dir, 'stats_test_acc.npz'))  # test acc of each epoch of all stages
        stats_train_acc = np.load(os.path.join(log_dir, 'stats_train_acc.npz'))  # train acc of each epoch of all stages
        
        num_epochs=stats_test_acc['epoch_val'].shape[0]
        epochs =range(1,num_epochs+1)

        test_acc=stats_test_acc['epoch_val'][:,0]-epochs
        train_acc=stats_train_acc['epoch_val'][:,0]-epochs
        print('test acc   ---- ', test_acc[-1])

        ########plot train and test accuracy for each epochs####
        color_set = ['r','b','g','k','y','m','c']
        plt.plot(epochs, train_acc, '-', label='training', color=color_set[0], linewidth=2)
        plt.plot(epochs, test_acc, '-', label='testing', color=color_set[1], linewidth=2)
        plt.xlabel('epochs')
        plt.ylabel('Recognition Accuracy')
        plt.legend(loc='lower right', framealpha=0.8)

        plt.show()
        


        #-----------------------------plot loss of both training and testing-------------------------------------------#

        loss_train_running_average_total = np.load(os.path.join(log_dir, 'stats_train_running_average.npz')) # running loss of J1 + J2
        loss_test_average = np.load(os.path.join(log_dir, 'stats_test_average.npz'))  # test loss of each epoch


        train_loss = loss_train_running_average_total['iter_loss']
        test_loss = loss_test_average['iter_loss']
        pdb.set_trace()
        plt.plot(train_loss[:, 0], train_loss[:, 1], '-', label='train_running_loss', color=color_set[0], linewidth=2)
        plt.plot(test_loss[:, 0], test_loss[:, 1], '-', label='test_loss', color=color_set[1], linewidth=2)

        #plt.ylim([0, 1])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(os.path.basename(log_dir) + '_loss')
        plt.legend(loc='upper right', framealpha=0.8)

        
        plt.show()


        #loss - batches
        loss_train_batch=np.load(os.path.join(log_dir, 'batches_loss_train.npy')) # num_epoch X (num_sampe/batch_size)
        loss_test_batch=np.load(os.path.join(log_dir, 'batches_loss_test.npy')) # num_epoch X (num_sampe/batch_size)
        loss_train_re=np.concatenate(loss_train_batch,0)
        loss_test_re=np.concatenate(loss_test_batch,0)

        # loss_train_re=loss_train_re[:1000]
        # loss_test_re =loss_test_re[0:1000]

        # plt.plot(range(0,loss_train_re.shape[0]),loss_train_re , '-', label='train_batch_loss', color=color_set[0], linewidth=2)
        # plt.plot(range(0,loss_test_re.shape[0]),loss_test_re, '-', label='test_batch_loss', color=color_set[1], linewidth=2)

        # #plt.ylim([0, 1])
        # plt.xlabel('batches')
        # plt.ylabel('loss')
        # plt.title(os.path.basename(log_dir) + '_loss')
        # plt.legend(loc='lower right', framealpha=0.8)       
        #plt.show()
        
        pdb.set_trace()

        #######precision recall roc########
        gt_train=np.load(os.path.join(log_dir, 'gt_train.npy')) #num_epoch X num_sample
        gt_test=np.load(os.path.join(log_dir, 'gt_test.npy'))

        pred_train=np.load(os.path.join(log_dir, 'pred_train.npy'))
        pred_test=np.load(os.path.join(log_dir, 'pred_test.npy'))

        score_train=np.load(os.path.join(log_dir, 'train_score.npy')) # num_sampeX1X1
        score_train=score_train[:,0,0]
        score_test=np.load(os.path.join(log_dir, 'test_score.npy')) # num_sampeX1X1
        score_test=score_test[:,0,0]

        #score_train_all=np.load(os.path.join(log_dir, 'train_score_all.npy')) # num_epoch X num_sampe X 1 X 1
        #score_test_all=np.load(os.path.join(log_dir, 'test_score_all.npy')) # num_epoch X num_sampe X 1 X 1
        

        # #####precision for each epoch tp/(tp+fp)
        #'''
        pr_train_eps=[]
        pr_test_eps=[]

        for i in range(num_epochs):

            pr_train=metrics.precision_score(gt_train[i,:], pred_train[i,:])
            pr_test=metrics.precision_score(gt_test[i,:], pred_test[i,:])

            pr_train_eps.append(pr_train)
            pr_test_eps.append(pr_test)
        #'''


        # #####Recall for each epoch tp / (tp + fn)
        #'''
        rc_train_eps=[]
        rc_test_eps=[]

        for i in range(num_epochs):

            rc_train=metrics.recall_score(gt_train[i,:], pred_train[i,:])
            rc_test=metrics.recall_score(gt_test[i,:], pred_test[i,:])

            rc_train_eps.append(rc_train)
            rc_test_eps.append(rc_test)
        #'''

        #####ROC curve######

        fpr, tpr, thresholds = metrics.roc_curve(gt_test[gt_test.shape[0]-1,:], score_test, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()






