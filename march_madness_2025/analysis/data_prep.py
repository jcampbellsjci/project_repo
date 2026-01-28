import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

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


# We're now going to summarize statistics from teams for a season
# We'll start by finding averages of each stat field for a season

# Creating a list of stat fields
stat_fields = [
    i for i in raw_game_log.columns if
    (i.startswith("Team1") or i.startswith("Team2"))
    and not (i.endswith("TeamID") or i.endswith("Loc"))
]

# Grouping by team and season and finding averages of stat fields
stat_avg = (
    raw_game_log
    .groupby(["Season", "Team1TeamID"])[stat_fields]
    .agg("mean")
    .reset_index()
)

# For some of these averages, we'll calculate ratios as input features
stat_avg = (
    stat_avg
    .assign(
        Team1FGP = lambda x: x["Team1FGM"] / x["Team1FGA"],
        Team1FGP3 = lambda x: x["Team1FGM3"] / x["Team1FGA3"],
        Team1FTP = lambda x: x["Team1FTM"] / x["Team1FTA"],
        Team2FGP = lambda x: x["Team2FGM"] / x["Team2FGA"],
        Team2FGP3 = lambda x: x["Team2FGM3"] / x["Team2FGA3"],
        Team2FTP = lambda x: x["Team2FTM"] / x["Team2FTA"]
    )
)


#### Kenpom Data ####

# Kenpom tracks advanced stats that can be a better measure of performance than general heuristic stats
# https://kenpom.com/

# We'll scrape data and put it into a df to use as model input

# Getting html from url
url = "https://kenpom.com/index.php?y=2025"
response = requests.get(url)
html = response.text

# Parsing html and getting ratings table
raw_table = BeautifulSoup(html, "html.parser").find("table", {"id": "ratings-table"})

# Getting all table rows from rating table
rows = raw_table.find_all("tr")

# Looping through rows and getting all table data and headers
headers = []
for row in rows[:]:
    cells = row.find_all(["th"])
    headers.append([cell.get_text(strip = True) for cell in cells])
headers = headers[1]

data = []
for row in rows[:]:
    cells = row.find_all(["td"])
    data.append([cell.get_text(strip = True) for cell in cells])
data = list(filter(None, data))


# We'll select specific fields from the header object and data objects
# The table structure is a bit complex, and does not translate into a df well as is
kp_finalized = [[headers[i] for i in [1] + list(range(4, 12))]]
# Now we'll loop through the data, index, and append
for i in data[:]:
    kp_finalized.append([i[j] for j in [1, 4, 5, 7, 9, 11, 13, 15, 17]])


kenpom_df = pd.DataFrame(kp_finalized[1:], columns = kp_finalized[0])

