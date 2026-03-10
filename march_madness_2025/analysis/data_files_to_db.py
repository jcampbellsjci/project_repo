import pandas as pd
import os
import sqlalchemy

# Setting working directory
os.chdir("/Users/jake/Documents/project_repo/march_madness_2025")

db_url = sqlalchemy.engine.URL.create(
    drivername = "postgresql+psycopg",
    username = "jake",
    password = os.getenv("DB_PASSWORD"),
    host = "localhost",
    port = 5432,
    database = "postgres"
)
engine = sqlalchemy.create_engine(db_url)

# Loading in data files
data_file_names = sorted([i for i in os.listdir("data") if not i.startswith(".")])

# Going to create a dictionary of raw data frames
raw_data = {}
for i in data_file_names:
    df = pd.read_csv("data/" + i)
    raw_data[i.split(".")[0]] = df

tables_to_upload = ["MRegularSeasonDetailedResults", "MNCAATourneySeeds", "MTeamSpellings", "MNCAATourneyDetailedResults", "MTeams", "SampleSubmissionStage2"]

for i in tables_to_upload:
    df = raw_data[i]
    df.to_sql(con = engine, name = i.lower(), schema = "march_madness", if_exists = "replace", index = False)
