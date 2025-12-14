import os
import time
import torch
import argparse
from torch.utils.data import DataLoader

from granitewxc.utils.config import get_config
from granitewxc.datasets.eccc import EcccHrdpsGdpsDataset

import os
import json
import logging
from typing import Callable, Optional, Dict, Any

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import lru_cache

def main(args):

    config = get_config(args.config_path)
    
    # ### Computing scalers
    # We will need the dataset class with `downsample_factor=None` and dataloader `batch_size=0 mod len(dataset)` 
    train_dataset = EcccHrdpsGdpsDataset(
        json_file_path = config.data.data_training_index,
        json_static_var_path = config.data.static_data_index,
        surface_vars = config.data.input_surface_vars, 
        vertical_pres_vars = config.data.vertical_pres_vars,
        vertical_level1_vars = config.data.vertical_level1_vars,
        vertical_level2_vars = config.data.vertical_level2_vars,
        other_vars = config.data.other,
        static_vars = config.data.input_static_surface_vars,
        output_vars = config.data.output_vars,
        downsample_factor = None,
        test=True
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=1,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        **dl_kwargs,
    )

    print("len(train_dataset)", len(train_dataset))
    print("len(train_loader)", len(train_loader))
    print("device", torch.cuda.current_device())

    #--------------------
    # Mean computation obs: the mean of the means is equal to the overall mean if all the groups have the same size (this is the case)
    #-------------------

    start_time = time.time()

    x_sum = 0
    static_x_sum = 0
    y_sum = 0
    static_y_sum = 0

    x_squared_sum = 0
    static_x_squared_sum = 0
    y_squared_sum = 0
    static_y_squared_sum = 0

    num_batches = len(train_loader)
    total_elements = 0

    for i, batch in enumerate(train_loader):
        if args.gpu:
            torch.cuda.empty_cache()
            batch['x'] = batch['x'].to(torch.cuda.current_device())
            batch['y'] = batch['y'].to(torch.cuda.current_device())
            batch['static_x'] = batch['static_x'].to(torch.cuda.current_device())
            batch['static_y'] = batch['static_y'].to(torch.cuda.current_device())

        # Sum values and squared values
        x_sum += batch['x'].sum(dim=(0, 2, 3))
        static_x_sum += batch['static_x'].sum(dim=(0, 2, 3))
        y_sum += batch['y'].sum(dim=(0, 2, 3))
        static_y_sum += batch['static_y'].sum(dim=(0, 2, 3))

        x_squared_sum += (batch['x'] ** 2).sum(dim=(0, 2, 3))
        static_x_squared_sum += (batch['static_x'] ** 2).sum(dim=(0, 2, 3))
        y_squared_sum += (batch['y'] ** 2).sum(dim=(0, 2, 3))
        static_y_squared_sum += (batch['static_y'] ** 2).sum(dim=(0, 2, 3))

        total_elements += batch['x'].shape[0] * batch['x'].shape[2] * batch['x'].shape[3]

        print(f'{i}/{num_batches}', end='\r')

    # compute means
    x_mean_overall = x_sum / total_elements
    static_x_mean_overall = static_x_sum / total_elements
    y_mean_overall = y_sum / total_elements
    static_y_mean_overall = static_y_sum / total_elements

    # compute variances
    x_variance = (x_squared_sum / total_elements) - (x_mean_overall ** 2)
    static_x_variance = (static_x_squared_sum / total_elements) - (static_x_mean_overall ** 2)
    y_variance = (y_squared_sum / total_elements) - (y_mean_overall ** 2)
    static_y_variance = (static_y_squared_sum / total_elements) - (static_y_mean_overall ** 2)

    # compute standard deviations
    x_std_overall = torch.sqrt(x_variance)
    static_x_std_overall = torch.sqrt(static_x_variance)
    y_std_overall = torch.sqrt(y_variance)
    static_y_std_overall = torch.sqrt(static_y_variance)


    print(f'Time: {time.time() - start_time}')
    print('============ Computed means and standard deviations ============')

    scalers = {
        'input_mu':x_mean_overall.squeeze(0) , 
        'input_sigma': x_std_overall.squeeze(0) ,
        'input_static_mu': static_x_mean_overall.squeeze(0) , 
        'input_static_sigma':static_x_std_overall.squeeze(0),
        'target_mu': y_mean_overall.squeeze(0), 
        'target_sigma': y_std_overall.squeeze(0), 
        'target_static_mu': static_y_mean_overall.squeeze(0) , 
        'target_static_sigma':static_y_std_overall.squeeze(0),
        }

    [
        print(val.shape) for val in [
            scalers['input_mu'], 
            scalers['input_sigma'], 
            scalers['input_static_mu'],
            scalers['input_static_sigma'],
            scalers['target_mu'],
            scalers['target_sigma'], 
            scalers['target_static_mu'],
            scalers['target_static_sigma']
        ]
    ]

    os.makedirs(args.save_dir, exist_ok=True)

    for name, tensor in scalers.items():
        file_path = f'{os.path.join(args.save_dir, name)}.pt'
        torch.save(tensor, file_path)

    print("scalers['input_mu']", scalers['input_mu'])
    print("scalers['input_sigma']", scalers['input_sigma'])
    print("scalers['input_static_mu']", scalers['input_static_mu'])
    print("scalers['input_static_sigma']", scalers['input_static_sigma'])
    print("scalers['target_mu']", scalers['target_mu'])
    print("scalers['target_sigma']", scalers['target_sigma'])

    print("Tensors saved successfully!")

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mu/sigma for GDPS and HRDPS data used for ECCC downscaling")
    parser.add_argument('--batch_size', default=1, type=int, help="The size of the batch.")
    parser.add_argument('--gpu', default=True, type=bool, help="Whether to use GPU or not.")
    parser.add_argument('--config_path', required=True, type=str, help='Path to the configuration YAML file.')
    parser.add_argument('--save_dir', default='./experiments/scalers', type=str, help='Path to save the scalars.')
    args = parser.parse_args()

    print(f"Args: {args}")
  
    main(args)