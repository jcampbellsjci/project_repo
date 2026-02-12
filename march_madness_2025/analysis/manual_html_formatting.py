import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import os

os.chdir("/Users/jake/Documents/project_repo/march_madness_2025")

# If kp html data exists, load it; if not, create empty df
if Path("data/kp_html.csv").exists():
    kp_html_df = pd.read_csv("data/kp_html.csv")
else:
    kp_html_df = pd.DataFrame(
        {
            "Season": [],
            "HTML": []
        }
    )

# Specify season and html
season = 2025
html = """"""

# Parsing html for ratings table
html_ratings_table = (
    BeautifulSoup(html, "html.parser")
    .find("table", {"id": "ratings-table"})
)

# Concatenating new row to og df and writing to csv
kp_html_df_new = pd.concat(
    kp_html_df,
    {
        "Season": season,
        "HTML": html_ratings_table
    }
)

kp_html_df_new.to_csv("data/kp_html.csv", index = False)
