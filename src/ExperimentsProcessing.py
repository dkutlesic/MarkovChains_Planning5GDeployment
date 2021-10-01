import os
import numpy as np
import torch

def prepare_experiments_files():
    '''
    Takes files from experiments and put them into
    two dictionaries according the dataset distribution
    '''
    chains_data = {}
    for file in os.listdir('chains_results'):
        if file[0] == '.':
            continue
        chains_data[file[17:-4]] = torch.load(os.path.join('chains_results', file))
    uni_chains_data = {}
    gaus_chains_data = {}
    for key in chains_data:
        if 'Uni' in key:
            uni_chains_data[key] = chains_data[key]
        else:
            gaus_chains_data[key] = chains_data[key]
    return uni_chains_data, gaus_chains_data

def get_experiments_data(data):
    '''
    Plots results for four base chains for different lambdas
    Parameters:
        data: dict
            Keeps all informaion about experiments with chains by their names
    '''
    x = []
    losses = []
    num_cities = []
    num_cities_in_max = []
    losses_v = []
    rej_rate = []
    rej_rate_in_max = []
    for key in data:
        lambdas = data[key]['lambdas']
        x.append(lambdas)
        loss_per_key = []
        num_cities_per_key = []
        num_cities_in_max_per_key = []
        rej_rate_in_max_per_key = []
        loss_v_per_key = []
        rej_rate_per_key = []
        for i in range(len(lambdas)):
            max_loss = np.max(data[key]['losses'][:, :, i])
            loss_per_key.append(max_loss)
            max_indices = np.where(data[key]['losses'][:, :, i] == max_loss)
            num_cities_per_key.append(np.max(data[key]['num_cities'][:, :, i]))
            num_cities_in_max_per_key.append(data[key]['num_cities'][max_indices[0][0], max_indices[1][0], i])
            loss_v_per_key.append(np.max(data[key]['losses_v'][:, :, i]))
            rej_rate_per_key.append(np.max(data[key]['rej_rate'][:, :, i]))
            rej_rate_in_max_per_key.append(data[key]['rej_rate'][max_indices[0][0], max_indices[1][0], i])

        losses.append(loss_per_key)
        num_cities.append(num_cities_per_key)
        num_cities_in_max.append(num_cities_in_max_per_key)
        losses_v.append(loss_v_per_key)
        rej_rate.append(rej_rate_per_key)
        rej_rate_in_max.append(rej_rate_in_max_per_key)    
    return losses, num_cities, num_cities_in_max, losses_v, rej_rate, rej_rate_in_max, x

def get_best_params(chains_data, lambda_num=1):
    '''
    Gets best betas and inceases from experiment files
    '''
    betas = {}
    increases = {}
    for key in chains_data:
        argmax = np.where(chains_data[key]['losses'][:, :, lambda_num] == chains_data[key]['losses'][:, :, lambda_num].max())
        betas[key] = chains_data[key]['betas'][argmax[0][0]]
        increases[key] = chains_data[key]['increases'][argmax[1][0]]
    return betas, increases