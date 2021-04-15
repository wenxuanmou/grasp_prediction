'''
author  : wmou, modified based on traintester.py in DeepFaceAlignment
created : 2/12/18 8:11 AM
'''

import os
import sys
import time
import glog as logger
import numpy as np
from matplotlib import pyplot as plt # for visualizing the output

import torch
from torch.autograd import Variable
# TEST-MOD: import cPickle # import _pickle as cPickle for python3;  import cPickle for python2
if( 3 > sys.version_info.major ):
    import cPickle;
else:
    import _pickle as cPickle
# TEST-MOD-END
import errno
import pdb

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

def check_exist_or_mkdirs(path):
    '''thread-safe mkdirs if not exist'''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Stats(object):

        def __init__(self):
            self.iter_loss = []
            self.epoch_val = []
        def push_loss(self, iter, loss):
            self.iter_loss.append([iter, loss])

        def push_val(self, epoch, val):
            self.epoch_val.append([epoch]+val)

        def push_push(self, epoch, val):
            self.push_val(epoch, val)

        def push(self, iter, loss):
            #self.push_accuracy(iter)
            self.push_loss(iter, loss)

        def save(self, file):
            np.savez_compressed(
                file,
                iter_loss=np.asarray(self.iter_loss))
        def save_val(self, file):
            np.savez_compressed(
                file,
                epoch_val=np.asarray(self.epoch_val))


class TrainTester(object):

    def __init__(self, net, solver, total_epochs, cuda,log_dir,
                 verbose_per_n_batch=1, iter_size=1, snapshot_dir=None, epoch_per_snapshot=1):
        self.net, self.solver, self.total_epochs, self.cuda = net, solver, total_epochs, cuda
        self.log_dir, self.verbose_per_n_batch, self.iter_size = log_dir, verbose_per_n_batch, iter_size
        check_exist_or_mkdirs(log_dir)


        if epoch_per_snapshot is None:
            epoch_per_snapshot = int(self.total_epochs*0.25) #default snapshot approximately 4 times in total
        if snapshot_dir is not None:
            check_exist_or_mkdirs(snapshot_dir)
        else:
            snapshot_dir = log_dir
        self.snapshot_dir, self.epoch_per_snapshot = snapshot_dir, epoch_per_snapshot

        #self.num_epochs_save = self.total_epochs
        self.done = False
        self.train_iter = 0
        self.test_iter = 0
        self.train_epoch = 0


        self.stats_train_batch = Stats()
        self.stats_train_running = Stats()
        self.stats_test = Stats()
        self.stats_train_batch_average = Stats()
        self.stats_train_running_average = Stats()
        self.stats_test_average = Stats()

        self.stats_test_acc = Stats()
        self.stats_train_acc = Stats()

        self.running_loss = None


        self.running_factor = 0.95
        self.epoch_callbacks = [self.save_stats, self.snapshot]

   
        self.gt_train=[]
        self.pred_train=[]
        self.score_train=[]

        self.gt_test=[]
        self.pred_test=[]
        self.score_test=[]


        self.batches_loss=[]
        self.batches_loss_test=[]





    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    logger.warn('epoch_callback[{}] failed.'.format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults['lr']  # 
        lr = base_lr * (total_step - step + 1.) / total_step
        for param_group in self.solver.param_groups:
            param_group['lr'] = lr


    def train(self, epoch, loader, loss_fn):

        self.net.train() #Sets the module in training mode.
        total_step = self.total_epochs * len(loader)
        finished_step = (epoch-1) * len(loader)
        loss_sum, batch_loss, loss_sum_running = 0.0, 0.0, 0.0
        acc_set_sum=0.0

       
        pred_eps=[]
        gt_eps=[]
        tt=[]
        y_score=[]
        batches_loss_per_epoch=[]

        for batch_idx, batch in enumerate(loader):
            self.adjust_lr_linear(batch_idx + finished_step, total_step)
            #img0_train = batch['im']
            img = batch['imNorm']
            label = batch['label'] #label
            filename = batch['filename']
            tt=filename

            if self.cuda:
                img = img.cuda()
                label = label.cuda()

            img = Variable(img)
            label = Variable(label)
            # training step
            if batch_idx % self.iter_size == 0:
                self.solver.zero_grad()
            # Forward propagation
            #pdb.set_trace()
            output = self.net.forward(img) #  B x 1 is a list            
            #compute the classification accuracy
            gt =  label  # B x 1 x 1
            output0 = torch.sigmoid(output)#B x 1 x 1
            v_pred=(output0.detach().cpu().numpy()>0.5)*1 #B x 1 x 1
            v_pred = v_pred[:,0,0] #B x 1
            v_gt=gt.cpu().numpy()[:,0,0] #B x 1
            acc_batch=np.sum((v_gt==v_pred)*1.0)/len(v_gt)
            acc_set_sum += acc_batch

            pred_eps.append(v_pred) # predicted results of the epoch
            gt_eps.append(v_gt) # gt of the epoch
            
            #compute the loss using binary cross entropy
            loss = loss_fn(output,gt) 

            # Backpropagation
            loss.backward()

            if batch_idx % self.iter_size == 0:
                self.solver.step()

            batch_loss = loss.item()
            loss_sum += batch_loss
            batches_loss_per_epoch.append(batch_loss)

            if self.running_loss is None:
                self.running_loss = batch_loss
            else:
                self.running_loss = self.running_factor*self.running_loss \
                                    + (1-self.running_factor)*batch_loss
            loss_sum_running += self.running_loss

            gt_eps_=np.concatenate(gt_eps,0)
            pred_eps_=np.concatenate(pred_eps,0)
            acc_eps=np.sum(gt_eps_==pred_eps_)*1.0/gt_eps_.shape[0]

            # collect stats
            self.train_iter += 1
            self.stats_train_batch.push(self.train_iter, loss=batch_loss)
            self.stats_train_running.push(self.train_iter, loss=self.running_loss)

            
            
            if self.verbose_per_n_batch > 0 and (batch_idx+1) % self.verbose_per_n_batch == 0:

                logger.info((
                        'Epoch={:<3d} [{:3.0f}% of {:<5d}] ' +
                        'Loss(Batch)={:.2f},Loss(Running)={:.2f}, acc={:.4f}').format(
                    epoch, 100. * (batch_idx+1) / len(loader), len(loader.dataset),
                    batch_loss, self.running_loss, acc_eps))
        
        #if epoch==self.total_epochs:
            y_t=output0.detach().cpu().numpy()
            y_score.append(y_t)
        y_score=np.concatenate(y_score,0) 
        self.score_train.append(y_score)
        np.save(os.path.join(self.log_dir,'train_score'),y_score)     
        
        self.batches_loss.append(batches_loss_per_epoch)

        self.stats_train_batch_average.push(epoch, loss=loss_sum / float(len(loader)))
        self.stats_train_running_average.push(epoch, loss=loss_sum_running / float(len(loader)))
        #print(tt)
        gt_eps_=np.concatenate(gt_eps,0)
        pred_eps_=np.concatenate(pred_eps,0)
        acc_eps=np.sum(gt_eps_==pred_eps_)*1.0/gt_eps_.shape[0]
        self.stats_train_acc.push_push(epoch, val=acc_eps)
        self.gt_train.append(gt_eps_)
        self.pred_train.append(pred_eps_)
        
        np.save(os.path.join(self.log_dir,'gt_train'),self.gt_train)
        np.save(os.path.join(self.log_dir,'pred_train'),self.pred_train)
        np.save(os.path.join(self.log_dir,'train_score_all'),self.score_train)


        logger.info('Train set (epoch={:<3d}): Loss(LastBatch,Average)={:.3f},{:.3f},'
                'acc_epoch ={:.4f}, acc_epoch ={:.4f} '.
                format(epoch, batch_loss, loss_sum / float(len(loader)), acc_set_sum / float(len(loader)), acc_eps))

    def test(self, epoch, loader, loss_fn):
        self.net.eval() #Sets the module in evaluation mode.
        loss_fn.size_average = False
        test_loss = 0.
        acc_set_sum=0.0
    

        pred_eps_test=[]
        gt_eps_test=[]
        ttt=[]
        batch_idx=-1
        y_score=[]
        batches_loss_per_epoch=[]

        for batch in loader:
            batch_idx=batch_idx+1
          
            img = batch['imNorm']
            label = batch['label']
            filename = batch['filename']
            ttt=filename

            if self.cuda:
                img = img.cuda()
                label = label.cuda()
                
            img = Variable(img)
            label = Variable(label)

            output = self.net.forward(img)
            gt =  label  # B x 1 x 1
            output0 = torch.sigmoid(output)#B x 1 x 1

            v_pred=(output0.detach().cpu().numpy()>0.5)*1 #B x 1 x 1
            v_pred = v_pred[:,0,0] #B x 1
            v_gt=gt.cpu().numpy()[:,0,0] #B x 1
            acc_batch=np.sum((v_gt==v_pred)*1.0)/len(v_gt)
            acc_set_sum += acc_batch

            pred_eps_test.append(v_pred) # predicted results of the epoch
            gt_eps_test.append(v_gt) # gt of the epoch
            
            #compute the loss using binary cross entropy
            loss_batch = loss_fn(output,gt) #       
            batch_loss = loss_batch.item()
            test_loss += batch_loss
            batches_loss_per_epoch.append(batch_loss)
           
            self.test_iter += 1
            self.stats_test.push(self.test_iter, loss = loss_batch.data)

            gt_eps_t=np.concatenate(gt_eps_test,0)
            pred_eps_t=np.concatenate(pred_eps_test,0)
            acc_eps_t=np.sum(gt_eps_t==pred_eps_t)*1.0/gt_eps_t.shape[0]

            if (batch_idx) % 200 == 0:

                logger.info((
                        'Epoch={:<3d} [{:3.0f}% of {:<5d}] ' +
                        'Loss(Batch)={:.2f}, acc={:.4f}').format(
                    epoch, 100. * (batch_idx) / len(loader), len(loader.dataset),
                    batch_loss,  acc_eps_t))

            y_t=output0.detach().cpu().numpy()
            y_score.append(y_t)
        self.batches_loss_test.append(batches_loss_per_epoch)
        y_score=np.concatenate(y_score,0) 
        self.score_test.append(y_score)
        np.save(os.path.join(self.log_dir,'test_score'),y_score)
        # predict results for each epoch
        gt_eps_=np.concatenate(gt_eps_test,0)
        pred_eps_=np.concatenate(pred_eps_test,0)
        acc_eps=np.sum(gt_eps_==pred_eps_)*1.0/gt_eps_.shape[0]
        self.stats_test_average.push(epoch, loss = test_loss / float(len(loader)))
        self.stats_test_acc.push_push(epoch, val = acc_eps)

                
        self.gt_test.append(gt_eps_)
        self.pred_test.append(pred_eps_)       
        np.save(os.path.join(self.log_dir,'gt_test'),self.gt_test)
        np.save(os.path.join(self.log_dir,'pred_test'),self.pred_test)
        np.save(os.path.join(self.log_dir,'test_score_all'),self.score_test)

        np.save(os.path.join(self.log_dir,'test_acc_eps'),acc_eps)

        logger.info('Test set  (epoch={:<3d}): AvgLoss={:.2f}, acc={:.4f}, acc_eps={:.4f}, '
                .format(epoch, test_loss / float(len(loader)),acc_set_sum / float(len(loader)),acc_eps))

        loss_fn.size_average = True 

 
    def save_stats(self):
        self.stats_train_running.save(os.path.join(self.log_dir, 'stats_train_running.npz'))
        self.stats_train_batch.save(os.path.join(self.log_dir, 'stats_train_batch.npz'))
        self.stats_test.save(os.path.join(self.log_dir, 'stats_test.npz')) # the same as train_batch
        self.stats_test_average.save(os.path.join(self.log_dir, 'stats_test_average.npz'))
        self.stats_train_batch_average.save(os.path.join(self.log_dir, 'stats_train_batch_average.npz'))
        self.stats_train_running_average.save(os.path.join(self.log_dir, 'stats_train_running_average.npz'))


        self.stats_test_acc.save_val(os.path.join(self.log_dir, 'stats_test_acc.npz'))
        self.stats_train_acc.save_val(os.path.join(self.log_dir, 'stats_train_acc.npz'))

    def snapshot(self, fname=None):
        if self.train_epoch % self.epoch_per_snapshot != 0:
            return

        if fname is None:
            fname = os.path.join(self.snapshot_dir, 'param_epoch={}.pkl'.format(self.train_epoch))
        if fname=='':
            fname = os.path.join(self.snapshot_dir, 'param_final.pkl')
        torch.save(self.net.state_dict(), f=fname, pickle_module=cPickle)
        logger.info('saved :'+fname)

    def count_parameter_num(self, verbose=False):
        cnt = 0
        params = self.net.state_dict() #Returns a dictionary containing a whole state of the module.
        for k in params.keys():
            p = params[k]
            sz = p.size()
            cntp = np.prod(sz)
            cnt += cntp
            if verbose:
                print('{}: {}={}'.format(k,tuple(sz),cntp))
        if verbose:
            logger.info('Number of parameters={}'.format(cnt))
            sys.stdout.flush()
        return cnt


    def run(self, train_loader, test_loader, loss_fn):
        logger.check_eq(self.done, False, 'Done already!')
        if self.cuda:
            self.net.cuda()
            print('run() is using cuda')
        else:
            print('run() is not using cuda')

        logger.info('Network Architecture:')
        print(str(self.net))
        sys.stdout.flush()
        self.count_parameter_num(True)

        logger.info('{} Hyperparameters:'.format(self.solver.__class__.__name__))

        sys.stdout.flush()
        self.count_parameter_num(True)

        #self.test(epoch=0, loader=test_loader, loss_fn=loss_fn)
        for epoch in range(1, self.total_epochs+1): # Main loop of runner. Test after each training epoch.
            self.train_epoch = epoch
            time0 =time.time()
            self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
            print('time for one epoch is: ', time.time() - time0)

            if (epoch-1) % self.epoch_per_snapshot == 0:
                self.test(epoch=epoch, loader=test_loader, loss_fn=loss_fn)
            self.invoke_epoch_callback()

        np.save(os.path.join(self.log_dir,'batches_loss_train'),self.batches_loss)
        np.save(os.path.join(self.log_dir,'batches_loss_test'),self.batches_loss_test)
        
        self.snapshot('')
        self.save_stats()
        self.done=True


    def run_test(self, test_loader, loss_fn):
        logger.check_eq(self.done, False, 'Done already!')
        if self.cuda:
            self.net.cuda()
            print('run() is using cuda')
        else:
            print('run() is not using cuda')

        #assert(0)

        logger.info('Network Architecture:')
        print(str(self.net))
        sys.stdout.flush()
        self.count_parameter_num(True)

        logger.info('{} Hyperparameters:'.format(self.solver.__class__.__name__))

        sys.stdout.flush()
        self.count_parameter_num(True)

        #self.test(epoch=0, loader=test_loader, loss_fn=loss_fn)
        #for epoch in range(1, self.total_epochs+1): # Main loop of runner. Test after each training epoch.
        time0 =time.time()

        self.test(epoch=1, loader=test_loader, loss_fn=loss_fn)
        print('time for one epoch is: ', time.time() - time0)
        self.invoke_epoch_callback()


        #self.snapshot('')
        #self.save_stats()
        self.done=True


    
