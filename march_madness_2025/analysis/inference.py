import pandas as pd
import os
import sqlalchemy
from xgboost import XGBClassifier
import numpy as np


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

season = 2025

team_records = pd.read_sql(con = engine, sql = f'select * from march_madness.team_records where "Season" = {season}')
stat_avg = pd.read_sql(con = engine, sql = f'select * from march_madness.season_averages where "Season" = {season}')
kenpom_final_df = pd.read_sql(con = engine, sql = f'select * from march_madness.kenpom_df where "Season" = {season}')
tourney_seeds_clean = pd.read_sql(con = engine, sql = f'select * from march_madness.tourney_seeds where "Season" = {season}')
sample_submission = pd.read_sql(con = engine, sql = 'select * from march_madness.samplesubmissionstage2')
team_names = pd.read_sql(con = engine, sql = 'select * from march_madness.mteams')


# Sample submission can be used as a format for our inference data set with some slight edits
# We need to split ID into team ID's and season and remove the sample prediction
inference_df = (
    sample_submission
    .assign(IDSplit = lambda x: x["ID"].str.split("_"))
    .assign(
        Season = lambda x: pd.to_numeric(x["IDSplit"].str[0]),
        Team1TeamID = lambda x: pd.to_numeric(x["IDSplit"].str[1]),
        Team2TeamID = lambda x: pd.to_numeric(x["IDSplit"].str[2])
    )
    .drop(columns = ["ID", "Pred", "IDSplit"])
)


#### Joining Data ####

# We'll join data to sample submission file
# Real submissions require predictions for every possible matchup, but we only care about real tournament teams

# Joining team seed data
# Since the inference df has all possible matchups, we'll inner join seed info to make sure it's only tournament matchups
final_inference_df = (
    inference_df
    .merge(
         tourney_seeds_clean,
         left_on = ["Season", "Team1TeamID"],
         right_on = ["Season", "TeamID"],
         how = "inner"
    )
    .drop(columns = ["TeamID"])
    .merge(
         tourney_seeds_clean,
         left_on = ["Season", "Team2TeamID"],
         right_on = ["Season", "TeamID"],
         how = "inner"
    )
    .drop(columns = ["TeamID"])
    .rename(columns = {"Seed_x": "Team1Seed", "Seed_y": "Team2Seed"})
)

# Joining records to tourney game log
final_inference_df = (
    final_inference_df
    .merge(
         team_records,
         on = ["Season", "Team1TeamID"],
         how = "left"
    )
    .rename(columns = {
        "GamesPlayed": "Team1GamesPlayed",
        "Wins": "Team1Wins",
        "Losses": "Team1Losses",
        "WinRatio": "Team1WinRatio"
    })
    .merge(
         team_records,
         left_on = ["Season", "Team2TeamID"],
         right_on = ["Season", "Team1TeamID"],
         how = "left"
    )
    .drop(columns = "Team1TeamID_y")
    .rename(columns = {
        "Team1TeamID_x": "Team1TeamID",
        "GamesPlayed": "Team2GamesPlayed",
        "Wins": "Team2Wins",
        "Losses": "Team2Losses",
        "WinRatio": "Team2WinRatio"
    })
)

# Merging season stats for team 1
final_inference_df = (
    final_inference_df
    .merge(
         stat_avg,
         on = ["Season", "Team1TeamID"],
         how = "left"
    )
)
# We'll rename team1 to Team A
# Team 2 from stats df will become Team A Opponent
final_inference_df.columns = [
   i.replace("Team1", "TeamA") if i.startswith("Team1")
   else i.replace("Team2", "TeamB") if i in ["Team2TeamID", "Team2Seed", "Team2GamesPlayed", "Team2Wins", "Team2Losses", "Team2WinRatio"]
   else i.replace("Team2", "TeamAOpp") if i.startswith("Team2")
   else i
   for i in final_inference_df.columns
]

# Merging season stats to team 2 (or as it's now known, team b)
final_inference_df = (
    final_inference_df
    .merge(
         stat_avg,
         left_on = ["Season", "TeamBTeamID"],
         right_on = ["Season", "Team1TeamID"],
         how = "left"
    )
    .drop(columns = ["Team1TeamID"])
)
# Renaming stat fields to team B
final_inference_df.columns = [
   i.replace("Team1", "TeamB") if i.startswith("Team1")
   else i.replace("Team2", "TeamBOpp") if i.startswith("Team2")
   else i
   for i in final_inference_df.columns
]

# Merging kenpom data to final df
final_inference_df = (
    final_inference_df
    .merge(
        kenpom_final_df,
        how = "left",
        left_on = ["Season", "TeamATeamID"],
        right_on = ["Season", "TeamID"]
    )
    .drop(columns = "TeamID")
    .merge(
        kenpom_final_df,
        how = "left",
        left_on = ["Season", "TeamBTeamID"],
        right_on = ["Season", "TeamID"]
    )
    .drop(columns = "TeamID")
)
final_inference_df.columns = [
   "TeamA" + i.replace("_x", "") if i.endswith("_x")
   else "TeamB" + i.replace("_y", "") if i.endswith("_y")
   else i
   for i in final_inference_df.columns
]


# We'll now take our final DF and put it in a postgres db
# Going to create a game ID field that combines team ID's and seasons as well as a created at ts
# Also going to join team name info
final_inference_df = (
    final_inference_df
    .assign(
        GameID = lambda x: x["Season"].astype("str") + "-" +
        x["TeamATeamID"].astype("str") + "-" +
        x["TeamBTeamID"].astype("str")
    )
    .assign(CreatedAt = pd.Timestamp.now(tz = "UTC"))
    .merge(
        team_names[["TeamID", "TeamName"]],
        left_on = "TeamATeamID",
        right_on = "TeamID",
        how = "left"
    )
    .drop(columns = "TeamID")
    .rename(columns = {"TeamName":"TeamAName"})
    .merge(
        team_names[["TeamID", "TeamName"]],
        left_on = "TeamBTeamID",
        right_on = "TeamID",
        how = "left"
    )
    .drop(columns = "TeamID")
    .rename(columns = {"TeamName":"TeamBName"})
)
# Rearranging fields so game ID and ts are first
final_inference_df = (
    final_inference_df[["GameID", "CreatedAt", "Season", "TeamATeamID", "TeamAName", "TeamBTeamID", "TeamBName"] +
    final_inference_df.columns[3:-4].tolist()]
    .assign(UsageType = "inference")
)

(
    final_inference_df
    .to_sql(con = engine, name = "ncaa_game_stats_raw", schema = "march_madness",
        if_exists = "append", index = False
    )
)


# Taking stats for team A and team b and putting them into long format
team_a_long = (
    final_inference_df
    .melt(
        id_vars = "GameID",
        value_vars = ["TeamASeed"] + list(final_inference_df.loc[:, "TeamAGamesPlayed":"TeamAWinRatio"].columns) + list(final_inference_df.loc[:, "TeamAScore":"TeamAOppFTP"].columns) + list(final_inference_df.loc[:, "TeamANetRtg":"TeamANCSOSNetRtg"].columns),
        var_name = "Variable",
        value_name = "AValue"
    )
    .assign(Variable = lambda x: x["Variable"].str.replace("TeamA", ""))
)

team_b_long = (
    final_inference_df
    .melt(
        id_vars = "GameID",
        value_vars = ["TeamBSeed"] + list(final_inference_df.loc[:, "TeamBGamesPlayed":"TeamBWinRatio"].columns) + list(final_inference_df.loc[:, "TeamBScore":"TeamBOppFTP"].columns) + list(final_inference_df.loc[:, "TeamBNetRtg":"TeamBNCSOSNetRtg"].columns),
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
final_inference_diff_df = (
    final_inference_df.loc[:, "GameID":"TeamBName"]
    .merge(
        long_diff_df,
        on = "GameID",
        how = "inner"
    )
    .assign(UsageType = "inference")
)

# Uploding to postgres
(
    final_inference_diff_df
    .to_sql(con = engine, name = "ncaa_game_stats_diff_raw", schema = "march_madness",
        if_exists = "append", index = False
    )
)


#### Making Predictions ####

# Loading model
xgb_model = XGBClassifier()
xgb_model.load_model("models/xgb_model.json")

# Prepping inference data
predictor_exclusions = ["Wins", "Losses", "FTM", "OppFTM", "FGM", "OppFGM", "FGM3", "OppFGM3", "NetRtg", "SOSDRtg", "SOSORtg","FTA", "OppFTA", "OppTO", "OppStl"]
id_fields = ["GameID", "CreatedAt", "Season", "TeamATeamID", "TeamAName", "TeamBTeamID", "TeamBName", "UsageType"]
inference_x = final_inference_diff_df.drop(columns = predictor_exclusions + id_fields)

inference_predictions = xgb_model.predict_proba(inference_x, )[:, 1]

inference_pred = (
    final_inference_diff_df[["GameID"]]
    .assign(
        PredType = "inference",
        Outcome = np.nan,
        PredProb = inference_predictions
    )
    [["PredType", "GameID", "Outcome", "PredProb"]]
)
inference_pred.to_sql(
    con = engine, name = "predictions", schema = "march_madness",
    index = False, if_exists = "append"
)
