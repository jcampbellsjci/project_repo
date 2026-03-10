from sklearn.metrics import brier_score_loss
import os
import pandas as pd
import sqlalchemy
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import optuna

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

diff_df = pd.read_sql(
    con = engine, sql = """select * from march_madness.ncaa_game_stats_diff_raw where "UsageType" = 'training'"""
)


#### Preprocessing ####

# Want to split df into training, validation, and testing set
# We'll use most recent year (with outcomes) as a testing set
season_filter = 2024
test_df = (
    diff_df
    .query("Season == @season_filter")
)

# We'll split the seasons preceeding this into a training and validation set
train_val_df = (
    diff_df
    .query("Season < @season_filter")
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size = .2, stratify = train_val_df["Outcome"]
)


#### Collinear Variables ####

# We're going to be trying to interpret variable importance, so we'll remove extremely-collinear variables
corr_matrix = train_df.loc[:, "AdjT":"Wins"].corr()
corr_df = (
    corr_matrix
    .reset_index()
    .melt(
        id_vars = "index",
        var_name = "Variable",
        value_name = "Value"
    )
    .query("index != Variable")
    .sort_values(by = "Value", ascending = False, key = abs)
)

# We'll create a list of columns to exclude due to their extreme correlation w/ others
predictor_exclusions = [
    # Removing fields that are represented by ratios
    "Wins", "Losses", "FTM", "OppFTM", "FGM", "OppFGM", "FGM3", "OppFGM3",
    # Removing fields that are just combos of other fields
    "NetRtg", "SOSDRtg", "SOSORtg",
    # Removing cause and effect fields
    "FTA", "OppFTA", "OppTO", "OppStl" 
]
corr_df.query("index not in @predictor_exclusions and Variable not in @predictor_exclusions").head(20)

# Getting id fields to exclude from training
id_fields = ["GameID", "CreatedAt", "Season", "TeamATeamID", "TeamAName", "TeamBTeamID", "TeamBName", "UsageType"]

train_x = train_df.drop(columns = ["Outcome"] + predictor_exclusions + id_fields)
train_y = train_df["Outcome"]
val_x = val_df.drop(columns = ["Outcome"] + predictor_exclusions + id_fields)
val_y = val_df["Outcome"]
test_x = test_df.drop(columns = ["Outcome"] + predictor_exclusions + id_fields)
test_y = test_df["Outcome"]


#### Parameter Estimation ####

# Defining objective function for optuna
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0)
    }

    model = XGBClassifier(**params)

    cv_scores = cross_val_score(
        model, train_x, train_y,
        cv = RepeatedKFold(n_splits = 5, n_repeats = 3),
        scoring = "neg_brier_score"
    )

    return cv_scores.mean()


study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 50, show_progress_bar = True)


#### Model Training and Predictions ####

# Training model on best params from optuna
xgb_model = XGBClassifier(**study.best_params)
xgb_model.fit(train_x, train_y)
xgb_model.save_model("models/xgb_model.json")

# Creating predictions and calculating brier score
train_predictions = xgb_model.predict_proba(train_x, )[:, 1]
brier_score_loss(y_true = train_y, y_proba = train_predictions)
val_predictions = xgb_model.predict_proba(val_x, )[:, 1]
brier_score_loss(y_true = val_y, y_proba = val_predictions)
test_predictions = xgb_model.predict_proba(test_x, )[:, 1]
brier_score_loss(y_true = test_y, y_proba = test_predictions)


# We'll combine predictions from each set together and store in db
pred_list = {
    "training": train_df[["GameID", "Outcome"]].assign(PredProb = train_predictions),
    "validation": val_df[["GameID", "Outcome"]].assign(PredProb = val_predictions),
    "testing": test_df[["GameID", "Outcome"]].assign(PredProb = test_predictions)
}
pred_df = (
    pd.concat(pred_list)
    .reset_index(level = 0, names = "PredType")
    .reset_index(drop = True)
)
pred_df.to_sql(
    con = engine, name = "predictions", schema = "march_madness",
    index = False, if_exists = "replace"
)
