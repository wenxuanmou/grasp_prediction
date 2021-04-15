import os
import numpy as np

import random
import pdb

# dir_path ='/home/g85734wm/Desktop/grasp/grasp/data/grasp_data'
# date_folds = os.listdir(dir_path) # ['20191113', '20191114']

# imPath=[]
# count_success=0
# count_fail=0
# for foldername in date_folds:
#   sub_folds=os.listdir(os.path.join(dir_path,foldername))
#   for sub_foldername in sub_folds:
#       #print('---------------------------- sub_foldername   ',sub_foldername)
#       if not sub_foldername.endswith('.npy'):
#           files = os.listdir(os.path.join(dir_path,foldername,sub_foldername))
#           if 'during.png' in files:
#               imPath.append(os.path.join('grasp_data',foldername,sub_foldername,'during.png'))
#               if 'true_amplitude.npy' in files:
#                   #grasp_label=1
#                   count_success=count_success+1
#                   #np.save(os.path.join(dir_path,foldername,sub_foldername,'grasp_label'), grasp_label)
#               else:
#                   #grasp_label=0
#                   count_fail=count_fail+1
#                   #np.save(os.path.join(dir_path,foldername,sub_foldername,'grasp_label'), grasp_label)
# #np.save('grasp_list', imPath)
# print('count_success--- ',count_success)
# print('count_fail--- ',count_fail)


#######################################################
############shuffle to create 10 folds#################
#######################################################

#comment the above code and only run the below

'''
full_list = np.load('grasp_list.npy')
random.seed(1)
random.shuffle(full_list)
'''
'''
n=30000

full_list_t=full_list[:int(n*0.7)]


np.save('./data_path/list_30k_train1', full_list_t)

full_list_tt=full_list[int(0.7*n):n]


np.save('./data_path/list_30k_test1', full_list_tt)
'''
'''
print(len(full_list))
k=5
num_im=len(full_list)
num_per_fold = num_im/k

count=0
for i in range(0,k):
    if i<4:
        full_list_t = full_list[i*num_per_fold:(i+1)*num_per_fold]
    else:
        full_list_t = full_list[i*num_per_fold:]
    count=count+len(full_list_t)
    np.save('./data_path/klist'+str(i+1), full_list_t)

print count



n_train = int(0.7*num_im)
for i in range(1,11):
    full_list = np.load('grasp_list.npy')
    print('----full list----', full_list[:2])
    full_list_t = full_list[:] # the change of full_list1 will not affect full_list
    random.seed(i)
    random.shuffle(full_list_t)
    np.save('train_list'+str(i), full_list_t[:n_train])
    np.save('test_list'+str(i), full_list_t[n_train:])
    #print('----full list----', full_list[:2])
    print('----full list'+str(i)+'----', full_list_t[:2])

'''
####5-fold cross-validation data split####
'''
klist1=np.load('./data_path/klist1.npy')
klist2=np.load('./data_path/klist2.npy')
klist3=np.load('./data_path/klist3.npy')
klist4=np.load('./data_path/klist4.npy')
klist5=np.load('./data_path/klist5.npy')

pdb.set_trace()
np.save('./data_path/klist_train1', np.concatenate((klist2,klist3,klist4,klist5)))
np.save('./data_path/klist_train2', np.concatenate((klist1,klist3,klist4,klist5)))
np.save('./data_path/klist_train3', np.concatenate((klist2,klist1,klist4,klist5)))
np.save('./data_path/klist_train4', np.concatenate((klist2,klist3,klist1,klist5)))
np.save('./data_path/klist_train5', np.concatenate((klist2,klist3,klist4,klist1)))
'''



full_list_left = np.load('grasp_list_left.npy')
full_list_right = np.load('grasp_list_right.npy')

random.seed(1)
random.shuffle(full_list_left)

random.shuffle(full_list_right)

full_list_both = np.concatenate((full_list_left,full_list_right))
random.shuffle(full_list_both)
'''
n=30000
full_list_both = np.concatenate((full_list_left,full_list_right))
random.shuffle(full_list_both)

full_list_t=full_list_both[:int(n*0.7)]


np.save('./data_path/5folder/both_list_30k_train1', full_list_t)

full_list_tt=full_list_both[int(0.7*n):n]


np.save('./data_path/5folder/both_list_30k_test1', full_list_tt)

assert(0)
pdb.set_trace()
'''


#####first split and then shuffle
num=len(full_list_both)
both_list_1 = full_list_both[:num/2] 
both_list_2= full_list_both[num/2:]

imPath=both_list_1
root_dir = '/home/g85734wm/Desktop/grasp/grasp/data'
labels=[]
for i in range(len(imPath)):
    label_path='/'.join(imPath[i].split('/')[:-1])+'/grasp_label.npy'
    label=np.load(os.path.join(root_dir, label_path))
    labels.append(label)
print(len(labels), np.sum(labels))

pdb.set_trace() 


print(len(both_list_1))
k=5
num_im=len(both_list_1)
num_per_fold = num_im/k

count=0
for i in range(0,k):
    if i<4:
        #pdb.set_trace()
        full_list_t = both_list_1[i*num_per_fold:(i+1)*num_per_fold] 
        #t2=both_list_1[i*num_per_fold:(i+1)*num_per_fold]
        #full_list_t = np.concatenate((t1,t2))
        

    else:
        full_list_t = both_list_1[i*num_per_fold:] 
        #t2 = both_list_1[i*num_per_fold:]
        #full_list_t=np.concatenate((t1,t2))
         

    count=count+len(full_list_t)
    np.save('./data_path/5folder/both_10folder/both_10list1_'+str(i+1), full_list_t)

print count

klist1=np.load('./data_path/5folder/both_10folder/both_10list1_1.npy')
klist2=np.load('./data_path/5folder/both_10folder/both_10list1_2.npy')
klist3=np.load('./data_path/5folder/both_10folder/both_10list1_3.npy')
klist4=np.load('./data_path/5folder/both_10folder/both_10list1_4.npy')
klist5=np.load('./data_path/5folder/both_10folder/both_10list1_5.npy')

#pdb.set_trace()
np.save('./data_path/5folder/both_10folder/both_10list_train1_1', np.concatenate((klist2,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train1_2', np.concatenate((klist1,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train1_3', np.concatenate((klist2,klist1,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train1_4', np.concatenate((klist2,klist3,klist1,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train1_5', np.concatenate((klist2,klist3,klist4,klist1)))


count=0
for i in range(0,k):
    if i<4:
        #pdb.set_trace()
        full_list_t = both_list_2[i*num_per_fold:(i+1)*num_per_fold] 
        #t2=both_list_2[i*num_per_fold:(i+1)*num_per_fold]
        #full_list_t = np.concatenate((t1,t2))
        

    else:
        full_list_t = both_list_2[i*num_per_fold:] 
        #t2 = both_list_2[i*num_per_fold:]
        #full_list_t=np.concatenate((t1,t2))
         

    count=count+len(full_list_t)
    np.save('./data_path/5folder/both_10folder/both_10list2_'+str(i+1), full_list_t)

print count

klist1=np.load('./data_path/5folder/both_10folder/both_10list2_1.npy')
klist2=np.load('./data_path/5folder/both_10folder/both_10list2_2.npy')
klist3=np.load('./data_path/5folder/both_10folder/both_10list2_3.npy')
klist4=np.load('./data_path/5folder/both_10folder/both_10list2_4.npy')
klist5=np.load('./data_path/5folder/both_10folder/both_10list2_5.npy')

#pdb.set_trace()
np.save('./data_path/5folder/both_10folder/both_10list_train2_1', np.concatenate((klist2,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train2_2', np.concatenate((klist1,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train2_3', np.concatenate((klist2,klist1,klist4,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train2_4', np.concatenate((klist2,klist3,klist1,klist5)))
np.save('./data_path/5folder/both_10folder/both_10list_train2_5', np.concatenate((klist2,klist3,klist4,klist1)))
'''
#####first split and then shuffle
print(len(full_list_left))
k=5
num_im=len(full_list_left)
num_per_fold = num_im/k

count=0
for i in range(0,k):
    if i<4:
        #pdb.set_trace()
        t1 = full_list_left[i*num_per_fold:(i+1)*num_per_fold] 
        t2=full_list_right[i*num_per_fold:(i+1)*num_per_fold]
        full_list_t = np.concatenate((t1,t2))
        random.shuffle(full_list_t) 

    else:
        t1 = full_list_left[i*num_per_fold:] 
        t2 = full_list_left[i*num_per_fold:]
        full_list_t=np.concatenate((t1,t2))
        random.shuffle(full_list_t) 

    count=count+len(full_list_t)
    np.save('./data_path/5folder/both_klist'+str(i+1), full_list_t)

print count
'''


# n_train = int(0.7*num_im)
# for i in range(1,11):
#   full_list = np.load('grasp_list.npy')
#   print('----full list----', full_list[:2])
#   full_list_t = full_list[:] # the change of full_list1 will not affect full_list
#   random.seed(i)
#   random.shuffle(full_list_t)
#   np.save('./data_path/5folder/both_klist'+str(i), full_list_t[:n_train])
#   #np.save('test_list'+str(i), full_list_t[n_train:])
#   #print('----full list----', full_list[:2])
#   print('----full list'+str(i)+'----', full_list_t[:2])


####5-fold cross-validation data split####
'''
klist1=np.load('./data_path/5folder/both_klist1.npy')
klist2=np.load('./data_path/5folder/both_klist2.npy')
klist3=np.load('./data_path/5folder/both_klist3.npy')
klist4=np.load('./data_path/5folder/both_klist4.npy')
klist5=np.load('./data_path/5folder/both_klist5.npy')

#pdb.set_trace()
np.save('./data_path/5folder/both_klist_train1', np.concatenate((klist2,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_klist_train2', np.concatenate((klist1,klist3,klist4,klist5)))
np.save('./data_path/5folder/both_klist_train3', np.concatenate((klist2,klist1,klist4,klist5)))
np.save('./data_path/5folder/both_klist_train4', np.concatenate((klist2,klist3,klist1,klist5)))
np.save('./data_path/5folder/both_klist_train5', np.concatenate((klist2,klist3,klist4,klist1)))
'''