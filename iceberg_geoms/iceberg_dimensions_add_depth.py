#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:18:19 2024

@author: laserglaciers
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/iceberg_geoms/clean/'
mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/factor_4/median/2023-07-27_urel0.05_TF6.67_bergs_coeff4_v2.pkl'


file_list = [file for file in os.listdir(path) if file.endswith('gpkg')]

labels = np.arange(50,1450,50) # lengths for this example and bins
bins = np.arange(0,1450,50) # lengths for this example and bins
keel_bins = np.arange(0,520,20)
keel_labels = np.arange(20,520,20)

with open(mbergs_dict, 'rb') as src:
    # mbergs = pickle.load(src)
    mbergs = gpd.pd.read_pickle(src)

keel_dict = {}
for length in labels:
    berg = mbergs[length]
    k = berg.KEEL.sel(time=86400)
    keel_dict[length] = k.data[0]


os.chdir(path)
for file in file_list:
    
    gdf = gpd.read_file(file)
    
    gdf['envelope_box'] = gdf.envelope
    
    bounds = gdf['envelope_box'].bounds
    
    width = bounds['maxx'] - bounds['minx']
    length = bounds['maxy'] - bounds['miny']
    
    gdf['width'] = width
    gdf['length'] = length
    gdf.drop('envelope_box',axis=1, inplace=True)
    
    gdf['max_dim'] = np.maximum(gdf['length'], gdf['width'])
    gdf['binned'] = gpd.pd.cut(gdf['max_dim'],bins=bins,labels=labels)

    gdf['keel_depth'] = gdf['binned'].map(keel_dict)
    gdf['keel_binned'] = gpd.pd.cut(gdf['keel_depth'],
                                                bins=keel_bins,labels=keel_labels)


    op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/iceberg_geoms/geoms_with_depth_bin_testing/'
    
    if not os.path.exists(op):
        os.makedirs(op)
    
    
    gdf.to_file(f'{op}{file[:-5]}_dims.gpkg', driver='GPKG')