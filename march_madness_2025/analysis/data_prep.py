import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import sqlalchemy


#### Prep ####

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

kp_dict = {}

# Looping through years and pulling and cleaning data
# I've been getting blocked from KP, so I set up a manual html csv we can pull from and clean up
im_blocked_from_kp = True

for url_year in range(2002, 2026):
    if im_blocked_from_kp:
        # Getting kp data from raw data dict
        raw_string = (
            raw_data['kp_html']
            .query("Season == @url_year")
            
        )["HTML"].tolist()

        # Formatting string into beautiful soup object
        raw_table = BeautifulSoup(raw_string[0], "html.parser")

    else:
        # Getting html from url
        url = "https://kenpom.com/index.php?y=" + str(url_year)
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
    headers = headers[0:headers.index("Luck") + 1] + ["SOSNetRtg", "SOSORtg", "SOSDRtg", "NCSOSNetRtg"]

    data = []
    for row in rows[:]:
        cells = row.find_all(["td"])
        data.append([cell.get_text(strip = True) for cell in cells])
    data = list(filter(None, data))


    # We'll select specific fields from the header object and data objects
    # The table structure is a bit complex, and does not translate into a df well as is
    kp_finalized = [[headers[i] for i in [1] + list(range(4, 13))]]
    # Now we'll loop through the data, index, and append
    for i in data[:]:
        kp_finalized.append([i[j] for j in [1, 4, 5, 7, 9, 11, 13, 15, 17, 19]])

    kenpom_df = pd.DataFrame(kp_finalized[1:], columns = kp_finalized[0])

    # KP data adds seeds and other elements to team name; we'll remove them
    kenpom_df["Team"] = kenpom_df["Team"].str.replace(r"\d+", "", regex = True)
    kenpom_df = (
        kenpom_df
        .assign(Team = lambda x: x["Team"].str.replace(r"\d+", "", regex = True))
        .assign(Team = lambda x: x["Team"].str.replace("\\*", "", regex = True))
    )

    # Statistical columns need to have plus symbols removed and treated as integers
    kenpom_df = (
        kenpom_df
            .assign(
                **{
                    i: lambda x, col = i: pd.to_numeric(
                        x[col].str.replace("\\+", "", regex = True
                    ))
                    for i in kenpom_df.columns[1:]
                }
            )
        )

    kp_dict["kp_" + str(url_year)] = kenpom_df


# Unioning kp data into one single df
kenpom_final_df = (
    pd.concat(kp_dict, names = ["Season"])
      .reset_index(level = "Season")
      .assign(Season = lambda x: pd.to_numeric(x["Season"].str.replace("kp_", "", regex = True)))
)

# Joining team name spellings to kp data so that they have a joinable team ID
# There are a chunk of teams that are spelled differently on kp; have to have a giant ifelse to assign them a team ID
kenpom_final_df = (
    kenpom_final_df
    .assign(Team = lambda x: x["Team"].str.lower())
    .merge(
        raw_data["MTeamSpellings"],
        how = "left",
        left_on = "Team",
        right_on = "TeamNameSpelling"
    )
    .assign(TeamID = lambda x: np.where(
        x["Team"] == "texas a&m corpus chris", 1394, np.where(
            x["Team"] == "illinois chicago", 1227, np.where(
                x["Team"] == "southeast missouri", 1369, np.where(
                    x["Team"] == "queens", 1474, np.where(
                        x["Team"] == "ut rio grande valley", 1410, np.where(
                            x["Team"] == "cal st. bakersfield", 1167, np.where(
                                x["Team"] == "bethune cookman", 1126, np.where(
                                    x["Team"] == "tarleton st.", 1470, np.where(
                                        x["Team"] == "tennessee martin", 1404, np.where(
                                            x["Team"] == "saint francis", 1384, np.where(
                                                x["Team"] == "louisiana monroe", 1419, np.where(
                                                    x["Team"] == "arkansas pine bluff", 1115, np.where(
                                                        x["Team"] == "mississippi valley st.", 1290, np.where(
                                                            x["Team"] == "arkansas little rock", 1114, np.where(
                                                                x["Team"] == "louisiana lafayette", 1418, np.where(
                                                                    x["Team"] == "southwest missouri st.", 1283, np.where(
                                                                        x["Team"] == "texas pan american", 1410, np.where(
                                                                            x["Team"] == "southwest texas st.", 1402, np.where(
                                                                                x["Team"] == "st. francis ny", 1383, np.where(
                                                                                    x["Team"] == "southeast missouri st.", 1369, np.where(
                                                                                        x["Team"] == "st. francis pa", 1384, np.where(
                                                                                            x["Team"] == "winston salem st.", 1445, np.where(
                                                                                                x["Team"] == "dixie st.", 1469, np.where(
                                                                                                    x["Team"] == "texas a&m commerce", 1477, x["TeamID"]
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    ))
    .assign(TeamID = lambda x: x["TeamID"].astype("int"))
    .drop(columns = ["Team", "TeamNameSpelling"])
)


#### Tournament Seeds ####

# Tournament seed data is a bit messy; want to clean up and treat as integer
tourney_seeds_clean = (
    raw_data['MNCAATourneySeeds']
    .assign(Seed = lambda x: pd.to_numeric(x["Seed"].str.replace(r"[A-Za-z]", "", regex = True)))
)


#### Combining and Finding Differences ####

# For every tourney game, we'll need to:
# 1. Join season summary, seed, and kp data to both teams
# 2. Find differences between teams

# First, we need to take our NCAA tournament game data and edit it similar to our regular season data
wtourney_team1_names = [
   i.replace("W", "Team1") if i.startswith("W")
   else i.replace("L", "Team2") if i.startswith("L")
   else i
   for i in raw_data['MNCAATourneyDetailedResults'].columns
]
ltourney_team1_names = [
   i.replace("W", "Team2") if i.startswith("W")
   else i.replace("L", "Team1") if i.startswith("L")
   else i
   for i in raw_data['MNCAATourneyDetailedResults'].columns
]

wtourney_team1 = raw_data['MNCAATourneyDetailedResults'].copy()
ltourney_team1 = raw_data['MNCAATourneyDetailedResults'].copy()
wtourney_team1.columns = wtourney_team1_names
ltourney_team1.columns = ltourney_team1_names

# Unlike regular season summaries, we only want one instance of each game
# It should be random, so outcome will be 50/50 split

# Getting random 50% of winning team
wtourney_team1_sample = wtourney_team1.sample(frac = .5)
# We'll remove games from team 2 that are in team 1 sample
ltourney_team1_sample = (
    ltourney_team1
    .merge(
        wtourney_team1_sample[["Season", "Team2TeamID", "Team1TeamID"]],
        left_on = ["Season", "Team1TeamID", "Team2TeamID"],
        right_on = ["Season", "Team2TeamID", "Team1TeamID"],
        how = "left",
        suffixes=("", "_b")
    )
    .query("Team2TeamID_b.isnull()")
    .drop(columns = ["Team2TeamID_b", "Team1TeamID_b"])
)

raw_tourney_game_log = (
    pd.concat([wtourney_team1_sample, ltourney_team1_sample])
    .assign(Outcome = lambda x: np.where(x["Team1Score"] > x["Team2Score"], 1, 0))
    [["Season", "Team1TeamID", "Team2TeamID", "Outcome"]]
)

# Joining seed info to tourney game log
final_df = (
    raw_tourney_game_log
    .merge(
         tourney_seeds_clean,
         left_on = ["Season", "Team1TeamID"],
         right_on = ["Season", "TeamID"],
         how = "left"
    )
    .drop(columns = ["TeamID"])
    .merge(
         tourney_seeds_clean,
         left_on = ["Season", "Team2TeamID"],
         right_on = ["Season", "TeamID"],
         how = "left"
    )
    .drop(columns = ["TeamID"])
    .rename(columns = {"Seed_x": "Team1Seed", "Seed_y": "Team2Seed"})
)

# Merging season stats for team 1
final_df = (
    final_df
    .merge(
         stat_avg,
         on = ["Season", "Team1TeamID"],
         how = "left"
    )
)
# We'll rename team1 to Team A
# Team 2 from stats df will become Team A Opponent
final_df.columns = [
   i.replace("Team1", "TeamA") if i.startswith("Team1")
   else i.replace("Team2", "TeamB") if i == ("Team2TeamID") or i == ("Team2Seed")
   else i.replace("Team2", "TeamAOpp") if i.startswith("Team2")
   else i
   for i in final_df.columns
]

# Merging season stats to team 2 (or as it's now known, team b)
final_df = (
    final_df
    .merge(
         stat_avg,
         left_on = ["Season", "TeamBTeamID"],
         right_on = ["Season", "Team1TeamID"],
         how = "left"
    )
    .drop(columns = ["Team1TeamID"])
)
# Renaming stat fields to team B
final_df.columns = [
   i.replace("Team1", "TeamB") if i.startswith("Team1")
   else i.replace("Team2", "TeamBOpp") if i.startswith("Team2")
   else i
   for i in final_df.columns
]


# We'll now take our final DF and put it in a postgres db
# Going to create a game ID field that combines team ID's and seasons as well as a created at ts
# Also going to join team name info
final_df = (
    final_df
    .assign(
        GameID = lambda x: x["Season"].astype("str") + "-" +
        x["TeamATeamID"].astype("str") + "-" +
        x["TeamBTeamID"].astype("str")
    )
    .assign(CreatedAt = pd.Timestamp.now(tz = "UTC"))
    .merge(
        raw_data['MTeams'][["TeamID", "TeamName"]],
        left_on = "TeamATeamID",
        right_on = "TeamID",
        how = "left"
    )
    .drop(columns = "TeamID")
    .rename(columns = {"TeamName":"TeamAName"})
    .merge(
        raw_data['MTeams'][["TeamID", "TeamName"]],
        left_on = "TeamBTeamID",
        right_on = "TeamID",
        how = "left"
    )
    .drop(columns = "TeamID")
    .rename(columns = {"TeamName":"TeamBName"})
)
# Rearranging fields so game ID and ts are first
final_df = (
    final_df[["GameID", "CreatedAt", "Season", "TeamATeamID", "TeamAName", "TeamBTeamID", "TeamBName"] +
    final_df.columns[3:-4].tolist()]
)

# Uploding to postgres
(
    final_df
    .to_sql(
        "ncaa_game_stats_raw",
        engine,
        if_exists = "replace",
        #if_exists = "append",
        index = False
    )
)


# Taking stats for team A and team b and putting them into long format
team_a_long = (
    final_df
    .melt(
        id_vars = "GameID",
        value_vars = ["TeamASeed"] + list(final_df.loc[:, "TeamAScore":"TeamAOppFTP"].columns),
        var_name = "Variable",
        value_name = "AValue"
    )
    .assign(Variable = lambda x: x["Variable"].str.replace("TeamA", ""))
)

team_b_long = (
    final_df
    .melt(
        id_vars = "GameID",
        value_vars = ["TeamBSeed"] + list(final_df.loc[:, "TeamBScore":"TeamBOppFTP"].columns),
        var_name = "Variable",
        value_name = "BValue"
    )
    .assign(Variable = lambda x: x["Variable"].str.replace("TeamB", ""))
)

# Joining long df's together and pivoting them wide
long_diff_df = (
    team_a_long
    .merge(
        team_b_long,
        on = ["GameID", "Variable"],
        how = "inner"
    )
    .assign(DiffValue = lambda x: x["AValue"] - x["BValue"])
    .pivot(
        index = "GameID",
        columns = "Variable",
        values = "DiffValue"
    )
    .reset_index()
)
long_diff_df.columns.name = None

# Joining back to ID columns from final df
final_diff_df = (
    final_df.loc[:, "GameID":"Outcome"]
    .merge(
        long_diff_df,
        on = "GameID",
        how = "inner"
    )
)

# Uploding to postgres
(
    final_diff_df
    .to_sql(
        "ncaa_game_stats_diff_raw",
        engine,
        if_exists = "replace",
        #if_exists = "append",
        index = False
    )
)
