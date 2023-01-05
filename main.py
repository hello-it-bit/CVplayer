#!/usr/bin/env python
# coding: utf-8

# # 训练集说明

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
get_ipython().system('ls /home/aistudio/data')
get_ipython().system('ls /home/aistudio/data/data162979/')


# In[2]:


# 读取训练数据, 训练集包含500个模型结构，以及这些结构在cplfw，market1501，dukemtmc等8个任务上的性能排序
import json
with open('/home/aistudio/data/data162979/CCF_UFO_train.json', 'r') as f:
    train_data = json.load(f)
print(train_data['arch1'])
print('train_num:',len(train_data.keys()))


# # 处理训练数据

# In[3]:


def convert_X(arch_str):
        temp_arch = []
        total_1 = 0
        total_2 = 0
        ts = ''
        for i in range(len(arch_str)):
            if i % 3 != 0 and i != 0 and i <= 30:
                elm = arch_str[i]
                ts = ts + elm
                if elm == 'l' or elm == '1':
                    temp_arch = temp_arch + [1, 1, 0, 0]
                elif elm == 'j' or elm == '2':
                    temp_arch = temp_arch + [0, 1, 1, 0]
                elif elm == 'k' or elm == '3':
                    temp_arch = temp_arch + [0, 0, 1, 1]
                else:
                    temp_arch = temp_arch + [0, 0, 0, 0]
            
            elif i % 3 != 0 and i != 0 and i > 30:
                elm = arch_str[i]
                if elm == 'l' or elm == '1':
                    temp_arch = temp_arch + [1, 1, 0, 0, 0]
                elif elm == 'j' or elm == '2':
                    temp_arch = temp_arch + [0, 1, 1, 0, 0]
                elif elm == 'k' or elm == '3':
                    temp_arch = temp_arch + [0, 0, 1, 1, 0]
                else:
                    temp_arch = temp_arch + [0, 0, 0, 0, 1]
            
        return temp_arch

train_list = [[],[],[],[],[],[],[],[]]
arch_list_train = []
name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank', 'msmt17_rank', 'veri_rank', 'vehicleid_rank', 'veriwild_rank', 'sop_rank']
for key in train_data.keys():
    for idx, name in enumerate(name_list):
        train_list[idx].append(train_data[key][name])
    arch_list_train.append(convert_X(train_data[key]['arch']))
print(arch_list_train[0])


# # 训练各任务预测器

# In[4]:


get_ipython().system('pip install paddleslim')


# In[5]:


from paddleslim.nas import GPNAS 
import numpy as np
import scipy
import scipy.stats

gp_list = []

for i in range(len(train_list[:])):
    # 每个任务有该任务专属的gpnas预测器
    gp_list.append(GPNAS(2,2))

train_num = 400


for i in range(len(train_list[:])):
    # 划分训练及测试集
    X_all_k, Y_all_k  = np.array(arch_list_train), np.array(train_list[i])
    X_train_k, Y_train_k, X_test_k, Y_test_k = X_all_k[0:train_num:1], Y_all_k[0:train_num:1], X_all_k[train_num::1], Y_all_k[train_num::1]
    # 初始该任务的gpnas预测器参数
    gp_list[i].get_initial_mean(X_train_k[0::2],Y_train_k[0::2])
    init_cov = gp_list[i].get_initial_cov(X_train_k)
    # 更新（训练）gpnas预测器超参数
    gp_list[i].get_posterior_mean(X_train_k[1::2],Y_train_k[1::2])  
   
    # 基于测试评估预测误差   
    y_predict = gp_list[i].get_predict(X_test_k)

    #基于测试集评估预测的Kendalltau
    print('Kendalltau:',scipy.stats.stats.kendalltau( y_predict,Y_test_k))


# # 查看测试集

# In[6]:


with open('/home/aistudio/data/data162979/CCF_UFO_test.json', 'r') as f:
    test_data = json.load(f)
test_data['arch99997']


# # 处理测试集数据

# In[7]:


test_arch_list = []
for key in test_data.keys():
    test_arch =  convert_X(test_data[key]['arch'])
    test_arch_list.append(test_arch)
print(test_arch_list[99499])


# # 预测各任务上的测试集的结果

# In[8]:


rank_all = []
for task in range(len(name_list)):
    print('Predict the rank of:', name_list[task])
    rank_all.append(gp_list[task].get_predict(np.array(test_arch_list)))


# # 生成提交结果

# In[9]:


for idx,key in enumerate(test_data.keys()):
    test_data[key]['cplfw_rank'] = int(rank_all[0][idx][0])
    test_data[key]['market1501_rank'] = int(rank_all[1][idx][0])
    test_data[key]['dukemtmc_rank'] = int(rank_all[2][idx][0])
    test_data[key]['msmt17_rank'] = int(rank_all[3][idx][0])
    test_data[key]['veri_rank'] = int(rank_all[4][idx][0])
    test_data[key]['vehicleid_rank'] = int(rank_all[5][idx][0])
    test_data[key]['veriwild_rank'] = int(rank_all[6][idx][0])
    test_data[key]['sop_rank'] = int(rank_all[7][idx][0])
print('Ready to save results!')
with open('./CCF_UFO_submit_A.json', 'w') as f:
    json.dump(test_data, f)


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
