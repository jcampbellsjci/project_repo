import pandas as pd
import os

#### Prep ####

# Setting working directory
os.chdir("/Users/jake/Documents/project_repo/march_madness_2025")

# Loading in data files
data_file_names = [i for i in os.listdir("data") if not i.startswith(".")]

