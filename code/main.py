"""
A Geometric Deep Learning Framework to Predict Brain Graph Evolution.
    

    ----------------------------------------------------------------
    This file contains the implementation of the training and testing process of our RBGM model.
        training part 

            Inputs: 

                model: constructor of our rbgm: model = model_1 = GNN_1().to(device)
                                                model_2 = GNN_1().to(device)
                
                optimizer: constructor of our model's optimizer (borrowed from PyTorch) 
                
                h_data_train_loader: (n x 35 x 35) tensor that shows connection between ROIs
                                      n_train: the total number of subjects 
                                      35: number of ROIs

                args:          parsed command line arguments, to learn more about the arguments run: 
                                       python demo.py --help

            Output:
                    for each epoch, prints out the mean training MSE loss, topological loss, and
                    total loss for each time point t1 and t2

        testing part

            Inputs:
                
                
                h_data_test_loader: (n x 35 x 35) tensor that shows connection between ROIs
                                       n_test: the total number of subjects 
                                      35: number of ROIs

                args: see train method above for model and args.
                
            Outputs:
                    saves the MSE losses for both time point tq and t1
                    creates tsne plots to compare predicted brain graph and ground-truth for each time point

    To evaluate our framework we used 3-fold cross-validation strategy.
    ---------------------------------------------------------------------
    Copyright 2021 Alpay TEKİN, Istanbul Technical University.
    All rights reserved.
"""
import sys
from pathlib import Path
from model import GNN_1,frobenious_distance
import model
from data_utils import  MRDataset, create_edge_index_attribute, swap, cross_val_indices, MRDataset2, timer
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch 
import torch.nn as nn 
import numpy as np 
import pickle 
import argparse 
from pathlib import Path
from plot import plot, visualization
#import psutil
import timeit

d = Path(__file__).resolve().parents[1]


torch.manual_seed(0)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("running on GPU")
else:
    device = torch.device('cpu')
    print("running on CPU")
    
#Parser 
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default = 200, help="Number of iteration")
parser.add_argument('--lr', type = float, default = 0.0001, help="Learninng rate")
parser.add_argument('--fold', type = int, default = 3, help = "Number of folds")
parser.add_argument('--decay', type = float, default = 0.0, help = "Weight decay")
parser.add_argument('--batch_size', type = int, default = 1, help = "Batch size")
parser.add_argument('--exp', type = int, default = 0, help = "Number of experiment")
parser.add_argument('--tp_coef', type = float, default = 10, help = "KL Loss Coefficient")
opt = parser.parse_args()

# Dataset

h_data = MRDataset2(str(d) + "/data", "lh", subs=989)

# Parameters

batch_size = opt.batch_size
lr = opt.lr
num_epochs = opt.epoch
folds = opt.fold
connectomes = 1 
train_generator = 1

#Training 

mael = torch.nn.L1Loss().to(device)
tp = torch.nn.MSELoss().to(device)
train_ind, val_ind = cross_val_indices(folds, len(h_data))
mae_loss_train_t1, mae_loss_train_t2, mae_loss_t1, mae_loss_t2 = list(), list(), list(), list()
tp_loss_train_1, tp_loss_train_2 = list(), list()
train_total_loss_1, train_total_loss_2 = list(), list()
predict_t1, predict_t2, original_t1, original_t2 = list(),list(),list(),list()
frobenious_1 = list()
frobenious_2 = list()


# Cross-validation

for fold in range(folds):
    train_set, val_set = h_data[list(train_ind[fold])], h_data[list(val_ind[fold])]
    h_data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    h_data_test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    total_step = len(h_data_train_loader)

    # Create models

    model_1 = GNN_1().to(device)
    model_2 = GNN_1().to(device)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr = lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr = lr)
    tic0 = timeit.default_timer()

    for epoch in range(num_epochs):

        # Will be used for reporting

        train_loss_t1, train_loss_t2 = 0.0, 0.0 
        mae_loss_eval_t1, mae_loss_eval_t2 = 0.0, 0.0
        tp_loss_1, tp_loss_2, tr_loss_1, tr_loss_2 = 0.0, 0.0, 0.0, 0.0
        model_1.train()
        model_2.train()
        for i, data in enumerate(h_data_train_loader):
            
            
            #Time Point 1
            data = data.to(device)
            optimizer_1.zero_grad()
            out_1 = model_1(data.x)
            
           
            # Topological Loss
            
            tp_1 = tp(out_1.sum(dim=-1), data.y.sum(dim=-1))
            tp_loss_1 += tp_1.item()
            
            #MAE Loss
            loss_1 = mael(out_1, data.y)
            train_loss_t1 += loss_1.item()
            
            total_loss_1 = loss_1 + opt.tp_coef * tp_1
            tr_loss_1 += total_loss_1.item()
            total_loss_1.backward()
            optimizer_1.step()
            
            #Time Point 2
            
            optimizer_2.zero_grad()
            out_2 = model_2(data.y)
            

            # Topological Loss
           
            tp_2 = tp(out_2.sum(dim=-1), data.y2.sum(dim=-1))
            tp_loss_2 += tp_2.item()
            
            
            #MAE Loss
            loss_2 = mael(out_2, data.y2)
            train_loss_t2 += loss_2.item()
            
            total_loss_2 = loss_2 + opt.tp_coef * tp_2
            tr_loss_2 += total_loss_2.item()
            total_loss_2.backward()
            optimizer_2.step()
            

            
            
            
            
        
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'[Train] Loss T1 : {train_loss_t1 / total_step:.5f}, Loss T2 : {train_loss_t2 / total_step:.5f} ')
        print(f'[Train] TP Loss T1: {tp_loss_1 / total_step:.5f}, TP Loss T2 : {tp_loss_2 / total_step:.5f}')
        print(f'[Train] Total Loss T1: {tr_loss_1 / total_step:.5f}, Total Loss T2 : {tr_loss_2 / total_step:.5f}' )
        mae_loss_train_t1.append(train_loss_t1 / total_step)
        mae_loss_train_t2.append(train_loss_t2 / total_step)
        tp_loss_train_1.append(tp_loss_1 / total_step)
        tp_loss_train_2.append(tp_loss_2 / total_step)
        train_total_loss_1.append(tr_loss_1 / total_step)
        train_total_loss_2.append(tr_loss_2 / total_step)
        
    
    tic1 = timeit.default_timer()
    timer(tic0,tic1)

    # Plot losses 

    plot("MAE", "MAE_Loss_T1_" + str(fold) + "_" + str(opt.exp), mae_loss_train_t1)
    plot("MAE", "MAE_Loss_T2_" + str(fold) + "_" + str(opt.exp), mae_loss_train_t2)
    plot("TP", "TP_Loss_T1_" + str(fold) + "_" + str(opt.exp), tp_loss_train_1)
    plot("TP", "TP_Loss_T2_" + str(fold) + "_" + str(opt.exp), tp_loss_train_2)
    plot("Total Loss", "Total_loss_T1_" + str(fold) + "_" + str(opt.exp), train_total_loss_1)
    plot("Total Loss", "Total_loss_T2_" + str(fold) + "_" + str(opt.exp), train_total_loss_2)
    

    # Evaluation 

    model_1.eval()
    model_2.eval()
    #model.hidden_state = torch.rand(1225,1225)
    with torch.no_grad():
        for i, data in enumerate(h_data_test_loader):
            data = data.to(device)
            out_1 = model_1(data.x)
            out_2 = model_2(out_1)
            loss_1 = mael(out_1, data.y)
            loss_2 = mael(out_2, data.y2)
            mae_loss_t1.append(loss_1.item())
            mae_loss_t2.append(loss_2.item())
            predict_t1.append(out_1.cpu().numpy())
            predict_t2.append(out_2.cpu().numpy())
            original_t1.append(data.y.cpu().numpy())
            original_t2.append(data.y2.cpu().numpy())
            frobenious_1.append(frobenious_distance(data.x, out_1))
            frobenious_2.append(frobenious_distance(data.y, out_2))
        
    
    # Save the losses 

    """
    with open(str(d) + "/results/Ground-truth_t1" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(original_t1,f)

    with open(str(d) + "/results/predicted_t1" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(predict_t1,f)

    with open(str(d) + "/results/predicted_t2" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(predict_t2,f)

     with open(str(d) + "/results/Frobenious_T1_" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(frobenious_1, f)
    
    with open(str(d) + "/results/Frobenious_T2_" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(frobenious_2, f)
    
    """

    with open(str(d) + "/results/MAE_Loss_t1_" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(mae_loss_t1, f)
    with open(str(d) + "/results/MAE_Loss_t2_" + str(fold) +  "_" + str(opt.exp), "wb" ) as f:
        pickle.dump(mae_loss_t2, f)
    
   

    # t-SNE Plots

    visualization(original_t1, predict_t1, "t-SNE_time_1_" + str(fold) + "_" + str(opt.exp))
    visualization(original_t2, predict_t2, "t-SNE_time_2_" + str(fold) + "_" + str(opt.exp))
    
    
    mae_loss_t1.clear()
    mae_loss_t2.clear()
    mae_loss_train_t1.clear()
    mae_loss_train_t2.clear()
    tp_loss_train_1.clear()
    tp_loss_train_2.clear()
    train_total_loss_1.clear()
    train_total_loss_2.clear()
    predict_t1.clear()
    predict_t2.clear()
    original_t1.clear()
    original_t2.clear()
    frobenious_1.clear()
    frobenious_2.clear()
    
    del model_1
    del model_2

print("End of the experiment")

        






