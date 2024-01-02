#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:22:05 2023

@author: sheng
"""

# Import modules
import pandas as pd
import os
import numpy as np
import time

# record the start time
start_time= time.time()

# input date of assessment
date_input = input("Enter the date of CMJ assessment with the following format yyyy_mm_dd: ")

# create folder path for CMJ data
folder_path = f"/Users/sheng/Library/Mobile Documents/com~apple~CloudDocs/Desktop/S&C/CMJ data/{date_input}"

# empty dataframe
data_frames= {}

# iterate through files
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='error', skiprows=(4),usecols=range(3))
        data_frames[filename]= df
        
# assign column header and drop useless rows in dict
for filename, df in data_frames.items():
    df.columns=['No.','Time','GRF']
    df.drop(index=0, inplace=True)
    df.drop(index=range(10001, 10004), inplace=True)
    df.reset_index(drop = True, inplace = True)

# Convert the 'Force (N)' column to numeric (integer)
for filename, df in data_frames.items():
    df['GRF'] = pd.to_numeric(df['GRF'], errors='coerce').fillna(0).astype(float)

    
# combine two force plates
for filename, df in data_frames.items():
    half_len=len(df)//2
    second_half = df.iloc[half_len:,2].values
    df.iloc[:half_len,2] =df.iloc[:half_len,2].add(second_half, fill_value=0)
    df.drop(df.index[half_len:],inplace=True)

# find out the starting point (5SD method) & mean BW
start = pd.DataFrame(columns=['filename', 'starting_tf', 'BW'])

for filename, df in data_frames.items():
    window_size = 3000
    rolling_std = df['GRF'].rolling(window=window_size).std()
    rolling_mean = df['GRF'].rolling(window=window_size).mean()
    threshold = rolling_mean - rolling_std * 5
    threshold_index = np.where(df['GRF'] < threshold)[0]
    
    if len(threshold_index) > 0:
        starting_index = threshold_index[0] - 30  # Adjust as needed
        bw_mean = df['GRF'].iloc[starting_index - window_size:starting_index].mean()
        # Replace 'NaN' with an empty string for the 'filename' column
        filename = filename if pd.notna(filename) else ''
        # Append a new row to the DataFrame
        start = start.append({'filename': filename, 'starting_tf': starting_index, 'BW': bw_mean}, ignore_index=True)
    else:
        # Replace 'NaN' with an empty string for the 'filename' column
        filename = filename if pd.notna(filename) else ''
        # Append NaN values if no data meets the criteria
        start = start.append({'filename': filename, 'starting_tf': np.nan, 'BW': np.nan}, ignore_index=True)

# normalize raw data with BW
for filename,df in data_frames.items():
    bw=start.loc[start['filename']==filename,'BW'].values[0]
    df['normalized_BW']=df['GRF']-bw

# find out the time frame of leaving the force plate
for filename, df in data_frames.items():
    starting_tf=start.loc[start['filename']==filename,'starting_tf'].values[0]
    first_0_index=(df.index>starting_tf) & (df['GRF']<=0)
    if first_0_index.any():
        start.loc[start['filename']==filename,'leaving_plate_tf']=df.index[first_0_index].min()
    else:
        print(f"No below or equal 0 value found after starting point {start[filename][0]}")

# find out the time frame of peak force
for filename, df in data_frames.items():
    starting_tf=start.loc[start['filename']==filename,'starting_tf'].values[0]
    leaving_plate_tf= start.loc[start['filename']==filename,'leaving_plate_tf'].values[0]
    subset = df[(df.index>starting_tf) & (df.index<leaving_plate_tf)]
    max_index=subset['GRF'].idxmax()
    start.loc[start['filename']==filename,'pf_tf']=df.index[max_index]
    
# find out the time frame with minimum force between starting_tf and pf
for filename, df in data_frames.items():
    starting_tf=start.loc[start['filename']==filename,'starting_tf'].values[0]
    pf_tf= start.loc[start['filename']==filename,'pf_tf'].values[0]
    subset = df[(df.index>starting_tf) & (df.index<pf_tf)]
    if not subset.empty:  # Check if there is data within the range
        min_index = subset['GRF'].idxmin()
        start.loc[start['filename'] == filename, 'min_force_tf'] = min_index
    else:
        print(f"No data points found between starting point and peak force for {filename}")
    
# find out the time frame which normalized_BW went above zero (ie the end of unweighting phase)
for filename, df in data_frames.items():
    min_force_tf = start.loc[start['filename'] == filename, 'min_force_tf'].values[0]
    leaving_plate_tf = start.loc[start['filename'] == filename, 'leaving_plate_tf'].values[0]
    subset = df[(df.index > min_force_tf) & (df.index < leaving_plate_tf)]
    back_to_0_index = np.where(subset['normalized_BW'] >= 0)[0]
    if len(back_to_0_index) > 0:
        # Find the first index where 'normalized_BW' returns back to zero
        first_back_to_zero = subset.index[back_to_0_index[0]]
        start.loc[start['filename'] == filename, 'back_to_zero'] = first_back_to_zero
    else:
        # Handle the case where 'normalized_BW' doesn't return back to zero
        start.loc[start['filename'] == filename, 'back_to_zero'] = np.nan

# calculate the normalized impulse, to find out the end of braking phase
for filename, df in data_frames.items():
    starting_tf = start.loc[start['filename'] == filename, 'starting_tf'].values[0]
    back_to_zero = start.loc[start['filename'] == filename, 'back_to_zero'].values[0]
    if not pd.isnull(starting_tf) and not pd.isnull(back_to_zero):
        starting_tf=int(starting_tf)
        back_to_zero= int(back_to_zero)
        uw_impulse= df.loc[starting_tf:back_to_zero, 'normalized_BW'].sum()
        # to find out the point of braking phase
        subset= df[df.index>back_to_zero]
        cumulative_sum= subset['normalized_BW'].cumsum()
        braking_end_index=cumulative_sum[cumulative_sum>=abs(uw_impulse)].index[0]
        braking_end_tf=df.index[braking_end_index]
        start.loc[start['filename']==filename,'end_of_braking']=braking_end_tf
    else: 
        print('starting_tf and back_to_zero is invalid')
        start.loc[start['filename']==filename,'end_of_braking']=np.nan

# calculate duration variable information
dur_var = pd.DataFrame()
dur_var['filename']= start['filename']
dur_var['Unweighting_dur']= (start['back_to_zero']-start['starting_tf'])/1000
dur_var['Braking_dur']= (start['end_of_braking']-start['back_to_zero'])/1000
dur_var['Propulsive_dur']= (start['leaving_plate_tf']-start['end_of_braking'])/1000
dur_var['total_dur'] = dur_var.apply(lambda row: row['Unweighting_dur'] + row['Braking_dur'] + row['Propulsive_dur'], axis=1)
# calculate force variables
force_var = pd.DataFrame()
force_var['filename']=start['filename']
for filename1, df in data_frames.items():
    # mark important time frames
    tf1= start.query("filename==@filename1")['starting_tf'].values[0]
    tf2= start.query("filename==@filename1")['back_to_zero'].values[0]
    tf3= start.query("filename==@filename1")['end_of_braking'].values[0]
    tf4= start.query("filename==@filename1")['leaving_plate_tf'].values[0]
   
    # calculate mean force and minimum force of unweighting phase
    if not pd.isnull(tf1) and not pd.isnull(tf2):
        tf1, tf2 = int(tf1), int(tf2)
        force_var.loc[force_var['filename']==filename1,'mean_force_unweighting']=\
            df.loc[tf1:tf2,'GRF'].mean()
        force_var.loc[force_var['filename']==filename1,'min_force_unweighting']=\
            df.loc[tf1:tf2,'GRF'].min()
    else:
        print(f'Invalid time frames for {filename1}')
   
    # calculate mean force and peak force of braking phase
    if not pd.isnull(tf2) and not pd.isnull(tf3):
        tf2, tf3 = int(tf2), int(tf3)
        force_var.loc[force_var['filename']==filename1,'mean_force_braking']=\
            df.loc[tf2:tf3,'GRF'].mean()
        force_var.loc[force_var['filename']==filename1,'peak_force_braking']=\
            df.loc[tf2:tf3,'GRF'].max()
       
    # calculate mean force and peak force of propulsive phase
    if not pd.isnull(tf3) and not pd.isnull(tf4):
        tf3,tf4= int(tf3), int(tf4)
        force_var.loc[force_var['filename']==filename1,'mean_force_propulsive']=\
            df.loc[tf3:tf4,'GRF'].mean()
        force_var.loc[force_var['filename']==filename1,'peak_force_propulsive']=\
                df.loc[tf3:tf4,'GRF'].max()

# calculate impulse variables
impulse_var = pd.DataFrame()
impulse_var['filename']=start['filename']
for filename1, df in data_frames.items():
    # mark important time frames
    tf1= start.query("filename==@filename1")['starting_tf'].values[0]
    tf2= start.query("filename==@filename1")['back_to_zero'].values[0]
    tf3= start.query("filename==@filename1")['end_of_braking'].values[0]
    tf4= start.query("filename==@filename1")['leaving_plate_tf'].values[0]
   
    # unweighting phase impulse and net impulse
    if not pd.isnull(tf1) and not pd.isnull(tf2):
        tf1,tf2 = int(tf1),int(tf2)
        impulse_var.loc[impulse_var['filename']==filename1,'unweighting_impulse']\
            =(df.loc[tf1:tf2,'GRF'].sum())/1000
        impulse_var.loc[impulse_var['filename']==filename1,'unweighting_net_impulse']\
            =(df.loc[tf1:tf2,'normalized_BW'].sum())/1000
   
    # braking phase impulse and net impulse
    if not pd.isnull(tf2) and not pd.isnull(tf3):
        tf2, tf3 = int(tf2),int(tf3)
        impulse_var.loc[impulse_var['filename']==filename1,'braking_impulse']\
            =(df.loc[tf2:tf3,'GRF'].sum())/1000
        impulse_var.loc[impulse_var['filename']==filename1,'braking_net_impulse']\
            =(df.loc[tf2:tf3,'normalized_BW'].sum())/1000
   
    # propulsive phase impulse and net impulse
    if not pd.isnull(tf3) and not pd.isnull(tf4):
        tf3, tf4 = int(tf3),int(tf4)
        impulse_var.loc[impulse_var['filename']==filename1,'propulsive_impulse']\
            =(df.loc[tf3:tf4,'GRF'].sum())/1000
        impulse_var.loc[impulse_var['filename']==filename1,'propulsive_net_impulse']\
            =(df.loc[tf3:tf4,'normalized_BW'].sum())/1000

# calculate RFD variables
RFD_var = pd.DataFrame()
RFD_var['filename']=start['filename']
for filename1, df in data_frames.items():
    # mark important time frames
    tf2= start.query("filename==@filename1")['back_to_zero'].values[0]
    tf3= start.query("filename==@filename1")['end_of_braking'].values[0]
    tf4= start.query("filename==@filename1")['leaving_plate_tf'].values[0]
  
    # calculate braking phase RFD (100ms)  
    if not pd.isnull(tf2) and not pd.isnull(tf3):
            tf2,tf3 = int(tf2), int(tf3)
            br_100_rfd = 0
            br_50_rfd = 0
            for i in range(tf2,tf3):
                start_index = i-100
                end_index = i
                force_change= df['GRF'].iloc[end_index]- df['GRF'].iloc[start_index]
                rfd= force_change/ 0.1
                
                # update braking RFD (100ms)
                if rfd> br_100_rfd:
                    br_100_rfd= rfd
            for x in range(tf2,tf3):
                start_index = x-50
                end_index = x
                force_change = df['GRF'].iloc[end_index]-df['GRF'].iloc[start_index]
                rfd= force_change/0.05
                
                #update braking RFD (50ms)
                if rfd>br_50_rfd:
                    br_50_rfd= rfd
            RFD_var.loc[RFD_var['filename']==filename1,['braking_RFD_100ms','braking_RFD_50ms']]\
                =[br_100_rfd,br_50_rfd]
    else:
        print('no available time frame')
            
    # calculate propulsive phase RFD (100ms)
    if not pd.isnull(tf3) and not pd.isnull(tf4):
        tf3,tf4= int(tf3), int(tf4)
        pr_100_rfd=0
        pr_50_rfd=0
        for i in range(tf3,tf4):
            start_index = i-100
            end_index= i
            force_change = df['GRF'].iloc[end_index]- df['GRF'].iloc[start_index]
            rfd= force_change/ 0.1
            
            # update propulsive RFD
            if rfd> pr_100_rfd:
                pr_100_rfd=rfd
        for x in range(tf3,tf4):
            start_index = x - 50
            end_index = x
            force_change=df['GRF'].iloc[end_index]- df['GRF'].iloc[start_index]
            rfd= force_change/ 0.05
            if rfd>pr_50_rfd:
                pr_50_rfd= rfd
                
        RFD_var.loc[RFD_var['filename']==filename1,['propulsive_RFD_100ms','propulsive_RFD_50ms']]=[pr_100_rfd,pr_50_rfd]
        
# calculate velocity variable
velocity_var = pd.DataFrame()
velocity_var['filename']= start['filename']
for filename1, df in data_frames.items():
    tf1= start.query("filename==@filename1")['starting_tf'].values[0]
    tf2= start.query("filename==@filename1")['back_to_zero'].values[0]
    tf3= start.query("filename==@filename1")['end_of_braking'].values[0]
    tf4= start.query("filename==@filename1")['leaving_plate_tf'].values[0]
    BW1=start.query("filename==@filename1")['BW'].values[0]
    acceleration=df['normalized_BW']/(BW1/9.8)
    TI = 0.001
   
    # calculate unweighting phase velocity
    if not pd.isnull(tf1) and not pd.isnull(tf2):
        tf1,tf2 = int(tf1), int(tf2)
        uw_velocity=0
        min_velocity = 0
        sum_velocity = 0
        for i in range(tf1,tf2):
            uw_velocity += acceleration[i] * TI
            sum_velocity += uw_velocity
            if uw_velocity< min_velocity:
                min_velocity = uw_velocity
        mean_velocity = sum_velocity/(tf2-tf1)
       
        velocity_var.loc[velocity_var['filename']==filename1,['uw_mean_v', 'uw_min_v','uw_end_v']]\
            =[mean_velocity,min_velocity,uw_velocity]
    # calculate braking phase velocity
    if not pd.isnull(tf2) and not pd.isnull(tf3):
        tf2,tf3 = int(tf2), int(tf3)
        br_velocity=uw_velocity
        peak_velocity = 0
        sum_velocity = 0
        for i in range(tf2,tf3):
            br_velocity += acceleration[i]*TI
            sum_velocity += br_velocity
            if br_velocity > peak_velocity:
                peak_velocity = br_velocity      
        mean_velocity = sum_velocity / (tf3-tf2)
        velocity_var.loc[velocity_var['filename']==filename1,['br_mean_v','br_peak_v','br_end_v']]\
            =[mean_velocity, peak_velocity,br_velocity]
     # calculate propulsive phase velocity
    if not pd.isnull(tf3) and not pd.isnull(tf4):
         tf3,tf4 = int(tf3), int(tf4)
         pr_velocity=br_velocity
         peak_velocity = 0
         sum_velocity = 0
         for i in range(tf3,tf4):
             pr_velocity += acceleration[i]*TI
             sum_velocity += pr_velocity
             if pr_velocity > peak_velocity:
                 peak_velocity = pr_velocity
         mean_velocity = sum_velocity/(tf4-tf3)
         velocity_var.loc[velocity_var['filename']==filename1,['pr_mean_v','pr_peak_v','pr_end_v']]\
             =[mean_velocity, peak_velocity,pr_velocity]

# calculate power variables (in progress!!!!!)
power_var = pd.DataFrame()
power_var['filename'] = start['filename']

for filename1, df in data_frames.items():
    tf1 = start.query("filename == @filename1")['starting_tf'].values[0]
    tf2 = start.query("filename==@filename1")['back_to_zero'].values[0]
    tf3 = start.query("filename==@filename1")['end_of_braking'].values[0]
    tf4 = start.query("filename == @filename1")['leaving_plate_tf'].values[0]
    BW1 = start.query("filename == @filename1")['BW'].values[0]
    acceleration = df['normalized_BW'] / (BW1 / 9.8)
    TI = 0.001

    if not pd.isnull(tf1) and not pd.isnull(tf2) and not pd.isnull(tf3) and not pd.isnull(tf4):
        df['velocity'] = 0.0  # Initialize the 'velocity' column with zeros
        tf1,tf2,tf3,tf4 = int(tf1), int(tf2), int(tf3), int(tf4)
        for i in range(tf1, tf4):
            velocity = 0.0  # Initialize velocity for the current time frame
            for j in range(i, tf1 - 1, -1):
                velocity += acceleration[j] * TI  # Accumulate acceleration to calculate velocity
            df.loc[i, 'velocity'] = velocity  # Store velocity in the DataFrame

        df['power'] = df['GRF'] * df['velocity']  
        unweighting_power = df.loc[tf1:tf2, 'power']  # Power during unweighting phase
        mean_unweighting_power = unweighting_power.mean()
        min_unweighting_power = unweighting_power.min()
        braking_power = df.loc[tf1:tf2,'power']
        mean_braking_power = braking_power.mean()
        peak_braking_power = braking_power.max()
        propulsive_power = df.loc[tf3:tf4,'power']
        mean_propulsive_power = propulsive_power.mean()
        peak_propulsive_power = propulsive_power.max()

        power_var.loc[power_var['filename'] == filename1, ['uw_mean_power', 'uw_min_power','br_mean_power','br_peak_power','pr_mean_power','pr_peak_power']] = \
            [mean_unweighting_power, min_unweighting_power,mean_braking_power,peak_braking_power, mean_propulsive_power, peak_propulsive_power]

# calculate other performance variables
performance_var = pd.DataFrame()
performance_var['filename']=start['filename']
for filename1, df in data_frames.items():
    peak_v = velocity_var.query("filename==@filename1")['pr_peak_v'].values[0]
    if not pd.isnull(peak_v):
        bw = start.query("filename == @filename1")['BW'].values[0]
        jump_height =  (peak_v**2)/19.6    
        performance_var.loc[performance_var['filename']==filename1,'jump_height']= jump_height    
        # calculate modRSI
        total_duration = dur_var.query("filename == @filename1")['total_dur'].values[0]
        performance_var.loc[performance_var['filename']== filename1,'modRSI'] = jump_height/ total_duration
        
# create a DataFrame that combines all the data
sum_dfs = {
    'performance_var': performance_var,
    'force_var': force_var,
    'duration_var': dur_var, 
    'impulse_var': impulse_var,
    'RFD_var': RFD_var,
    'velocity_var': velocity_var,
    'power_var':power_var}      

# merge data
merged_data = None
for df_name, df in sum_dfs.items():
    if merged_data is None:
        merged_data = df
    else:
        merged_data = pd.merge(merged_data,df,how='outer',on='filename')

merged_data['name']= merged_data['filename'].str.extract(r'([^_]+)_\d{4}')

# groupby name and calculate average
consolidated_data = merged_data.groupby('name').mean().reset_index()
athlete_data = consolidated_data[['name','jump_height','modRSI','propulsive_RFD_100ms','mean_force_propulsive','peak_force_propulsive','pr_mean_power']]
athlete_data['date']=date_input

# saving file
result_file_path = '/Users/sheng/Library/Mobile Documents/com~apple~CloudDocs/Desktop/S&C/CMJ data/Result/'
if date_input:
    athlete_data.to_csv(f"{result_file_path}CMJ_{date_input}.csv", index=False, encoding='utf-8')
    print(f"Data saved to CMJ_{date_input}.csv")
else:
    print("No file name provided. Data was not saved.")        
        
# record end time
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time}s")        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        