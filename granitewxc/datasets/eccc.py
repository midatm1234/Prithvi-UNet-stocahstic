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


class EcccHrdpsGdpsDataset(Dataset):
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
    
    def __init__(
        self,
        json_file_path: str, # JSON with  [gdps,hrdps] data paths
        json_static_var_path :str, # JSON with static var paths
        surface_vars: list[str]= [], # list of dinamic surface vars
        vertical_pres_vars : list[str]= [], # list of vertical surface vars depending on press
        vertical_level1_vars : list[str]= [], # list of vertical surface vars depending on level1
        vertical_level2_vars : list[str]= [], # list of vertical surface vars depending on level2
        other_vars : list[str]= [], # list of other variables
        static_vars : list[str]= [], # list of static variables
        output_vars: list[str]= [], # list of output vars
        downsample_factor:int = None,
        n_random_windows:int = 1,
        crop_factor:int = None,
        type_loading: str = "normal", 
        test: bool = False
    ) -> None:

        
        # attributes
        self.surface_vars = surface_vars
        self.vertical_pres_vars = vertical_pres_vars
        self.vertical_level1_vars = vertical_level1_vars
        self.vertical_level2_vars = vertical_level2_vars
        self.other_vars = other_vars
        self.static_vars = static_vars
        self.output_vars = output_vars
        self.downsample_factor = downsample_factor
        self.n_random_windows = n_random_windows
        self.crop_factor = crop_factor
        self.type_loading = type_loading
        self.test = test

        # data loading
        self.data_index_repo = self.__load_json(json_file_path)
        self.static_data_repo = self.__load_json(json_static_var_path)
        self.log_buffer = []

        if not self.test:
            
            # get one image per epoch
            self.id_key = np.random.randint(0, len(self.data_index_repo)) 
    
            gdps_input, hrdps_input = self.__read_gdps_hrdps__(self.id_key)
    
            #random crop
    
            for k,tensor in hrdps_input.items():
                _, height, width = hrdps_input[k].shape
                break
        
            # Perform the crop
            self.hrdps_list, self.gdps_list = [], []
            for _ in range(self.n_random_windows):
                
                top = torch.randint(0, height - self.crop_factor + 1, (1,)).item()
                left = torch.randint(0, width - self.crop_factor + 1, (1,)).item()
                hrdps_input_out, gdps_input_out = {} , {}

                #print(top, left)
    
                for k,tensor in hrdps_input.items():
                    hrdps_input_out[k] = tensor[:, top:top + self.crop_factor, left:left + self.crop_factor]
    
                for k,tensor in gdps_input.items():
                    gdps_input_out[k] = tensor[:, top:top + self.crop_factor, left:left + self.crop_factor]
                    
                # downsampling gdps
                for k,tensor in gdps_input_out.items():
                    gdps_input_out[k] = self.__downsample(tensor, self.downsample_factor)
                    
                self.hrdps_list.append(hrdps_input_out)
                self.gdps_list.append(gdps_input_out)
                
        
    ##@profile
    def __load_json(self, json_file) -> Optional[Dict[str, Any]]:
        try:
            with open(json_file, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            #print(f"Error loading JSON file: {e}")
            self._log_debug("Error loading JSON file: {e}")
            return None 
            
    def _log_debug(self, message):
        """Accumulate debug messages to the buffer."""
        self.log_buffer.append(message)

    def __open_dataset(self, path: str, preferred_engine: Optional[str] = None, **kwargs) -> xr.Dataset:
        """Open a dataset trying multiple xarray engines when needed."""

        normalized_path = self.__ensure_dataset_file_ready(path)

        engines_to_try = []
        seen = set()
        for engine in (preferred_engine, None, "h5netcdf", "netcdf4", "scipy"):
            if engine in seen:
                continue
            engines_to_try.append(engine)
            seen.add(engine)

        errors = []
        for engine in engines_to_try:
            open_kwargs = dict(kwargs)
            try:
                if engine is None:
                    return xr.open_dataset(normalized_path, **open_kwargs)
                return xr.open_dataset(normalized_path, engine=engine, **open_kwargs)
            except ValueError as err:
                errors.append(f"{engine or 'auto'}: {err}")
                continue

        raise ValueError(
            f"Unable to open dataset {normalized_path} with xarray. "
            f"Attempted engines: {', '.join('auto' if e is None else e for e in engines_to_try)}. "
            f"Errors -> {' | '.join(errors)}"
        )

    def __ensure_dataset_file_ready(self, path: str) -> str:
        """Validate dataset path, detect Git LFS placeholders, and return absolute path."""

        abs_path = os.path.abspath(path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Dataset file not found: {abs_path}")

        try:
            with open(abs_path, "rb") as fh:
                prefix = fh.read(len(self.GIT_LFS_POINTER_PREFIX))
        except OSError as err:
            raise OSError(f"Unable to read dataset file {abs_path}: {err}") from err

        if prefix.startswith(self.GIT_LFS_POINTER_PREFIX):
            pointer_details = self.__read_git_lfs_pointer(abs_path)
            raise RuntimeError(
                f"{abs_path} is a Git LFS placeholder without the actual NetCDF data. "
                "Run `git lfs install` once and `git lfs pull` (or download the sample "
                "dataset) so the large files referenced in the JSON indices are available. "
                f"Pointer details: {pointer_details}"
            )

        return abs_path

    def __read_git_lfs_pointer(self, path: str) -> str:
        """Return a concise description of a Git LFS pointer file."""
        try:
            with open(path, "r", encoding="utf-8") as pointer_file:
                pointer_text = pointer_file.read().strip()
        except UnicodeDecodeError:
            return "Git LFS pointer detected"

        info_lines = []
        for line in pointer_text.splitlines():
            line = line.strip()
            if line.startswith(("oid", "size")):
                info_lines.append(line)

        return ", ".join(info_lines) if info_lines else pointer_text

    #@profile
    def __compute_temporal_features(self, idx:int, M):

        filename = os.path.basename(self.data_index_repo[str(idx)][0]) # GDPS
        
        base_name = filename[:-3]  # Remove the '.nc' extension
        date_part, hour_part = base_name.split('_')
        day = int(date_part[6:8])  # Extract DD from YYYYMMDDXX
        hour = int(hour_part)      # Extract HHH

        # only interval [1...365]
        day = day if day <= 365 else 365
        
       
        temporal_features = torch.ones((4, *M.shape[1:]), dtype=torch.float32)
        temporal_features[0] = temporal_features[0] * np.cos(day / 366 * 2 * np.pi)
        temporal_features[1] = temporal_features[1] * np.sin(day / 366 * 2 * np.pi)
        temporal_features[2] = temporal_features[2] * np.cos(hour / 24 * 2 * np.pi)
        temporal_features[3] = temporal_features[3] * np.sin(hour / 24 * 2 * np.pi)
        
        return temporal_features

    #@profile
    @lru_cache(maxsize=45)  
    def __position_signal(self, latitudes: tuple, longitudes: tuple,  dim_x:int = 1290, dim_y:int = 2540) -> torch.Tensor:

        latitudes = np.array(latitudes).reshape((dim_x, dim_y))
        longitudes = np.array(longitudes).reshape((dim_x, dim_y))
        
        longitudes = longitudes / 360 * 2.0 * np.pi
        latitudes = latitudes / 360 * 2.0 * np.pi

        position_signal_vars = np.stack([np.sin(latitudes), np.cos(longitudes), np.sin(longitudes)], axis=0)
        
        return torch.from_numpy(position_signal_vars).to(dtype=torch.float32)
        
    #@profile     
    def __get_vars(self, ds_xr: xr.Dataset,  var_names : list[str], type_vars :str = None) -> torch.Tensor:

        
        if var_names is None:
            return None
            
        else:
            # create a dict of tensors       
            dict_torch_tensors = {var: torch.tensor(ds_xr[var].values) for var in  var_names}

            # adding an extra dimension for tensors for this vars: from shape (.,.) to (.,.,.)
            if var_names == self.other_vars or var_names == self.static_vars:
                dict_torch_tensors = {key: value.unsqueeze(0) for key, value in dict_torch_tensors.items()}
               
            torch_tensor = torch.cat(list(dict_torch_tensors.values()), dim=0)
            
            if  type_vars is not None:
                print(type_vars, torch_tensor.shape)

            return torch_tensor  

    
    #@profile
    def __get_input__(self, ds_xr: xr.Dataset, idx:int, id_str:str) -> Dict[str, torch.Tensor]:

        #-----------------------------------
        # dynamic vars {surface, vertical}
        #-----------------------------------
        l_tensors = []

        if id_str =='x':
            l_vars = [self.surface_vars, self.vertical_pres_vars, self.vertical_level1_vars, self.vertical_level2_vars, self.other_vars]
            l_var_description = ["dynamic surface vars" , "dynamic vertical vars (pres)" , "dynamic vertical vars (level1)", "dynamic vertical vars (level2)", "other" ]

        else:
            l_vars = [self.output_vars]
            l_var_description = ["output vars"]


        for var, description in zip(l_vars, l_var_description):
            #l_tensors.append(self.__get_vars(ds_xr, var, description))
            data = self.__get_vars(ds_xr, var, None)
            if data is not None:
                l_tensors.append(self.__get_vars(ds_xr, var, None))

        torch_tensor_dynamic = torch.cat(l_tensors, dim=0)
 
        #--------------
        # static vars
        #--------------
        l_tensors = []

        l_tensors.append(self.__get_vars(ds_xr, self.static_vars, None ))

        #-----------------
        # positional  vars
        #-----------------
        latitudes = tuple(ds_xr.lat.values.flatten())
        longitudes = tuple(ds_xr.lon.values.flatten())
        M = self.__position_signal(latitudes , longitudes )
        #print("positional lat/lon", M.shape) 
        l_tensors.append(M)

        #------------------
        # temporal features
        #------------------
        T = self.__compute_temporal_features(idx, M)
        #print("temporal ", T.shape)
        l_tensors.append(T) 

        torch_tensor_static = torch.cat(l_tensors, dim=0)

       
        return {id_str: torch_tensor_dynamic, 'static_'+id_str:torch_tensor_static}

    #@profile
    def __downsample(self,tensor:torch.Tensor, factor:int) -> torch.Tensor:
        return  F.interpolate(tensor.unsqueeze(0), scale_factor=(1/factor, 1/factor), mode='nearest-exact').squeeze(0)

    #@profile
    def __len__(self):
        if self.test:
            return len(self.data_index_repo)
        return self.n_random_windows
            
    #@profile
    def __get_data__(self, id_key:int, static_key:str, data_type:str) -> xr.Dataset:

        index = 0 if data_type == 'gdps' else 1

        # open gdps or hrdps file
        preferred_engine = 'h5netcdf' if self.type_loading == 'lazy' else None
        open_kwargs = {} if self.type_loading == 'lazy' else dict(decode_timedelta=False)

        xr_ds = self.__open_dataset(
            self.data_index_repo[str(id_key)][index],
            preferred_engine=preferred_engine,
            **open_kwargs,
        ).isel(time=0)

        static_ds_xr = self.__open_dataset(
            self.static_data_repo[static_key],
            preferred_engine=preferred_engine,
        )
        
        # delete unnecesary variables
        if 'rotated_pole' in xr_ds.data_vars:
            xr_ds = xr_ds.drop_vars('rotated_pole')

        # interpolating static coords into dynamic var coords
        static_ds_xr = static_ds_xr.interp_like(xr_ds, method='nearest')
        xr_ds = xr_ds.assign( ME=static_ds_xr.ME,  MG=static_ds_xr.MG, Z0=static_ds_xr.Z0)

        return xr_ds


    #@profile
    @lru_cache(maxsize=45)  
    def __read_gdps_hrdps__(self,id_key:int)->list[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        gdps_xr = self.__get_data__(id_key, "static_regridded_gdps", "gdps")
        hrdps_xr = self.__get_data__(id_key, "static_hrdps", "hrdps")
        
        #----
        gdps_input = self.__get_input__(gdps_xr, id_key, 'x') 
        hrdps_input = self.__get_input__(hrdps_xr, id_key, 'y') 

        return gdps_input, hrdps_input

    #@profile
    def __getitem__(self, idx):
        if self.test:
            # clear cache to prevent multiple dawnsampling of the same file
            self.__read_gdps_hrdps__.cache_clear() 
            gdps_input, hrdps_input = self.__read_gdps_hrdps__(idx)

            # perform downsampling and
            # crops the images up to the max size multiple of 8 (downsample factor=8)
            if self.downsample_factor is not None:
                for k, tensor in gdps_input.items():
                    gdps_input[k] = self.__downsample(tensor[:, :1280,:2528], self.downsample_factor)

            for k, tensor in hrdps_input.items():
                hrdps_input[k] = tensor[:, :1280,:2528]

            return {**gdps_input, **hrdps_input}

        return {**self.hrdps_list[idx], **self.gdps_list[idx]}
