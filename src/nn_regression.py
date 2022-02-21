#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
nn_regression.py - a simple neural net rergression using PyTorch
author: Bill Thompson
license: GPL 3
copyright: 2022-02-17
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import copy

import torch
import torch.utils.data as Data

from Net import Net

def GetArgs():
    """
    GetArgs - return arguments from command line. 
    Use -h to see all arguments and their defaults.

    Returns
    -------
    argparse object
        parser object
    """    
    def ParseArgs(parser):
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write('error: %s\n' % message)
                self.print_help()
                sys.exit(2)

        parser = Parser(description='A simple neural net rergression using PyTorch')

        parser.add_argument('input_file',
                            help="""
                            Input CSV file (required).
                            Must contain columns X and Y
                            """)
        parser.add_argument('-o', '--output_path',
                            help="Output file csv file. (optional)",
                            required = False,
                            type = str)
        parser.add_argument('-b', '--batch_size',
                            required = False,
                            default = 64,
                            type=int,
                            help='Batch size (default = 64)')
        parser.add_argument('-e', '--epochs',
                            required=False,
                            type=int,
                            default = 200,
                            help='Number epochs Default=200')
        parser.add_argument('-l', '--layer1_out',
                            required=False,
                            type=int,
                            default = 200,
                            help='Number inputs for hidden layer 1 Default=200')
        parser.add_argument('-L', '--layer2_out',
                            required=False,
                            type=int,
                            default = 100,
                            help='Number inputs for hidden layer 2 Default=100')
        parser.add_argument('-r', '--learning_rate',
                            required=False,
                            type=float,
                            default = 0.01,
                            help='Learning rate Default = 0.01')
        parser.add_argument('-w', '--workers',
                            required=False,
                            type=int,
                            default = 2,
                            help='Number of processes for Dataloader Default = 2')
        parser.add_argument('-s', '--rand_seed',
                            required=False,
                            type=int,
                            help='Random mumber generator seed.')
    
        return parser.parse_args()

    parser = argparse.ArgumentParser(description='A simple neural net rergression using PyTorch')
    args = ParseArgs(parser)

    return args

def read_data(input_file):
    """
    read_data - read data for regression

    Parameters
    ----------
    input_file : str
        path to input csv file

    Returns
    -------
    tuple
        x, y - tensors containing data for regression
        X, Y - lists containing original data
    """    
    df = pd.read_csv(input_file)
    df = df.sort_values(by = 'X')

    X = df['X'].tolist()
    Y = df['Y'].tolist()

    x_scale = (X - np.mean(X)) / np.std(X)
    y_scale = (Y - np.mean(Y)) / np.std(Y)

    x = torch.Tensor(x_scale).unsqueeze(dim = 1)
    y = torch.Tensor(y_scale).unsqueeze(dim = 1)

    return x, y, X, Y

def plot_results(x, y, prediction, output_path):
    """
    plot_results - plot the data and predictions

    Parameters
    ----------
    x : Tensor
        SST data
    y : Tenso
        UK37
    output_path : str
        if not None, path to save plot
    prediction : Tensor
        Predicted results 
    """    
    fig, ax = plt.subplots()
    ax.set_title('Neural Net Regression Analysis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(x, y, s = 2, label = 'Data')
    ax.plot(x, prediction, 
            color = 'red', label = 'Prediction')
    plt.legend()
    if output_path is not None:
        fig.savefig(output_path + 'figure.png')
    plt.show()

def write_settings(input_file, output_path,
                   batch_size, epochs,
                   learning_rate, 
                   layer1_out, layer2_out,
                   workers,
                   rand_seed,
                   r2, min_loss):
    """
    write_setting - output setting to file

    Parameters
    ----------
    input_file : str
        data file
    output_path : str
        path to save data
    batch_size : int
        size of batches for regression
    epochs : int
        number of epochs
    learning_rate : float
        learing rate for NN
    layer1_out : int
        number inputs for hidden layer 1
    layer2_out : int
        number inputs for hidden layer 2
    workers : int
        number of parallel processes for regression
    rand_seed : int
        rng seed
    r2 : float
        R^2 of best model
    min_loss : float
        loss value of best model
    """                  
    if output_path is None:
        f = sys.stdout
    else:     
        out_file = output_path + 'settings.txt'
        f = open(out_file, 'w')

    print('input_file:', os.path.abspath(input_file), file = f)
    if output_path is None:
        print('output_path:', 'None', file = f)
    else:
        print('output_path:', os.path.abspath(output_path), file = f)
    print('batch_size:', batch_size, file = f)
    print('epochs:', epochs, file = f)
    print('learning_rate:', learning_rate, file = f)
    print('layer1_out:', layer1_out, file = f)
    print('layer2_out:', layer2_out, file = f)
    print('workers:', workers, file = f)
    print('rand_seed:', rand_seed, file = f)
    print('R^2:', r2, file = f)
    print('Best model loss:', min_loss, file = f)

    if output_path is not None:
        f.close()

def main():
    args = GetArgs()
    input_file = args.input_file
    output_path = args.output_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    layer1_out = args.layer1_out
    layer2_out = args.layer2_out
    workers = args.workers
    rand_seed = args.rand_seed

    if output_path is not None:
        if output_path != '/':
            output_path += '/'

    if rand_seed is None:
        rand_seed = int(time.time())
    np.random.seed(rand_seed)

    x, y, X, Y = read_data(input_file)

    # construct the model
    net = Net(layer1_out = layer1_out, layer2_out = layer2_out)

    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss_func = torch.nn.MSELoss()  # mean squared loss

    # set up data for loading
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = workers,)

    # start training
    best_model = None
    min_loss = sys.maxsize
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
            prediction = net(batch_x)

            loss = loss_func(prediction, batch_y)     
            print('epoch:', epoch, 'step:', step, 'loss:', loss.data.numpy())
            if loss.data.numpy() < min_loss:
                min_loss = loss.data.numpy()
                best_model = copy.deepcopy(net)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    # get a prediction
    prediction = best_model(x)

    r2 = r2_score(y.data.numpy(), prediction.data.numpy())
    write_settings(input_file, output_path,
                   batch_size, epochs,
                   lr, 
                   layer1_out, layer2_out,
                   workers,
                   rand_seed,
                   r2, min_loss)

    # prediction on same scale a Y
    prediction_rescale = (prediction.data.numpy() * np.std(Y) + np.mean(Y)).squeeze()
    
    plot_results(X, Y, prediction_rescale, output_path)

    if output_path is not None:
        df = pd.DataFrame({'X': x.data.numpy().squeeze(),
                           'Y': y.data.numpy().squeeze(),
                           'Prediction': prediction.data.numpy().squeeze(),
                           'X_original': X,
                           'Y_original': Y,
                           'prediction_rescale': prediction_rescale})

        df.to_csv(output_path + 'results.csv', index = False)

        torch.save(best_model.state_dict(), output_path + 'model.trc')
        
if __name__ == "__main__":
    main()
