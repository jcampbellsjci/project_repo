import pandas as pd
import os

#### Prep ####

# Setting working directory
os.chdir("/Users/jake/Documents/project_repo/march_madness_2025")


# Loading in data files
data_file_names = sorted([i for i in os.listdir("data") if not i.startswith(".")])

# Going to create a dictionary of raw data frames
raw_data = {}
for i in data_file_names:
    df = pd.read_csv("data/" + i)
    raw_data[i.split(".")[0]] = df
