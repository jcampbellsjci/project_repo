import pandas as pd
import numpy as np
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


#### Season Summaries ####

# We'll start by summarizing each team's season
# Season averages, record, etc.

# Season results are stored in a df where a single game makes up one row
# Teams are identified by whether they won or lost
# We need to transform this df so that each game has two rows (one for the winner and one for the loser)

# We'll start by renaming column references from winners and losers to team 1 and team 2
# We'll do this twice; once where the winner is called team 1 and one where the loser is called team 1
w_team1_names = [
   i.replace("W", "Team1") if i.startswith("W")
   else i.replace("L", "Team2") if i.startswith("L")
   else i
   for i in raw_data['MRegularSeasonDetailedResults'].columns
]
l_team1_names = [
   i.replace("W", "Team2") if i.startswith("W")
   else i.replace("L", "Team1") if i.startswith("L")
   else i
   for i in raw_data['MRegularSeasonDetailedResults'].columns
]

# We'll now create two versions of the season detail df and rename columns for both
w_team1 = raw_data["MRegularSeasonDetailedResults"].copy()
l_team1 = raw_data["MRegularSeasonDetailedResults"].copy()
w_team1.columns = w_team1_names
l_team1.columns = l_team1_names

# We'll now union the two dataframes to create one raw game log
raw_game_log = pd.concat([w_team1, l_team1])


# Now that we have a raw game log, let's calculate games played and records for teams each season
team_records = (
    raw_game_log
    .assign(
        Team1Win = lambda x: x["Team1Score"] > x["Team2Score"],
        Team1Loss = lambda x: x["Team1Score"] < x["Team2Score"]
    )
    .groupby(["Season", "Team1TeamID"])
    .agg(
        GamesPlayed = ("Team1TeamID", "count"),
        Wins = ("Team1Win", "sum"),
        Losses = ("Team1Loss", "sum")
    )
    .assign(WinRatio = lambda x: x["Wins"] / x["GamesPlayed"])
    .reset_index()
)
