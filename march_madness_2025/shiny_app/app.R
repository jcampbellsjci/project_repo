library(shiny)
library(DBI)
library(RPostgres)
library(DT)
library(tidyverse)

con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5432,
  dbname = "postgres",
  user = "jake",
  password = Sys.getenv("DB_PASSWORD")
)

raw_stat_df <- dbGetQuery(
  conn = con,
  statement = "select * from march_madness.ncaa_game_stats_raw"
)
diff_stat_df <- dbGetQuery(
  conn = con,
  statement = "select * from march_madness.ncaa_game_stats_diff_raw"
)
prediction_df <- dbGetQuery(
  conn = con,
  statement = "select * from march_madness.predictions"
)
shap_df <- dbGetQuery(
  conn = con,
  statement = "select * from march_madness.shap_values"
)

raw_stat_df <- raw_stat_df %>%
  bind_rows(
    raw_stat_df %>%
      rename_with(~ gsub("^TeamA", "TEMP", .)) %>%
      rename_with(~ gsub("^TeamB", "TeamA", .)) %>%
      rename_with(~ gsub("^TEMP", "TeamB", .)) %>%
      mutate(
        Outcome = as.integer(ifelse(Outcome == 1, 0, 1)),
        GameID = str_c(Season, TeamATeamID, TeamBTeamID, sep = "-")
      )
  )
diff_stat_df <- diff_stat_df %>%
  bind_rows(
    diff_stat_df %>%
      rename_with(~ gsub("^TeamA", "TEMP", .)) %>%
      rename_with(~ gsub("^TeamB", "TeamA", .)) %>%
      rename_with(~ gsub("^TEMP", "TeamB", .)) %>%
      mutate(
        Outcome = as.integer(ifelse(Outcome == 1, 0, 1)),
        GameID = str_c(Season, TeamATeamID, TeamBTeamID, sep = "-")
      ) %>%
      mutate(across(.cols = c(AdjT:Wins), .fns = ~ . * -1))
  )
prediction_df <- prediction_df %>%
  bind_rows(
    prediction_df %>%
      mutate(
        Outcome = as.integer(ifelse(Outcome == 1, 0, 1)),
        GameID = str_c(
          str_split_fixed(GameID, pattern = "-", n = 3)[1],
          str_split_fixed(GameID, pattern = "-", n = 3)[3],
          str_split_fixed(GameID, pattern = "-", n = 3)[2]
        ),
        PredProb = 1 - PredProb
      )
  )
shap_df <- shap_df %>%
  bind_rows(
    shap_df %>%
      mutate(
        GameID = str_c(
          str_split_fixed(GameID, pattern = "-", n = 3)[1],
          str_split_fixed(GameID, pattern = "-", n = 3)[3],
          str_split_fixed(GameID, pattern = "-", n = 3)[2]
        ),
        Value = Value * -1
      )
  )

# Define UI for app that draws a histogram ----
ui <- fluidPage(
  titlePanel("NCAA Tournament Analysis"),

  fluidRow(
    column(
      width = 2,
      selectInput(
        inputId = "season_select",
        label = "Season",
        choices = sort(unique(raw_stat_df$Season))
      ),
      selectInput(
        inputId = "team_a_select",
        label = "Team",
        choices = sort(unique(raw_stat_df$TeamAName))
      ),
      selectInput(
        inputId = "team_b_select",
        label = "Opponent",
        choices = sort(unique(raw_stat_df$TeamBName))
      )
    ),
    column(
      width = 2,
      style = "padding-left: 40px",
      tags$div(
        tags$label("Outcome"),
        textOutput(outputId = "outcome"),
        style = "margin-bottom: 12px"
      ),
      tags$div(
        tags$label("Predicted Probability"),
        textOutput(outputId = "pred_prob"),
        style = "margin-bottom: 12px"
      ),
      tags$div(
        tags$label("Overall Percentile"),
        textOutput(outputId = "pred_perc_overall"),
        style = "margin-bottom: 12px"
      ),
      tags$div(
        tags$label("Seed Percentile"),
        textOutput(outputId = "pred_perc_seed")
      )
    ),
    column(
      width = 6,
      DTOutput(outputId = "stat_table", width = "100%")
    )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output, session) {
  observe({
    valid_teams <- raw_stat_df %>%
      mutate(TeamANameSeed = str_c(TeamAName, TeamASeed, sep = " - ")) %>%
      filter(Season == input$season_select) %>%
      pull(TeamANameSeed)

    updateSelectInput(
      session = session,
      inputId = "team_a_select",
      choices = valid_teams
    )
  })

  observe({
    valid_opponents <- raw_stat_df %>%
      mutate(TeamANameSeed = str_c(TeamAName, TeamASeed, sep = " - ")) %>%
      mutate(TeamBNameSeed = str_c(TeamBName, TeamBSeed, sep = " - ")) %>%
      filter(
        Season == input$season_select & TeamANameSeed == input$team_a_select
      ) %>%
      pull(TeamBNameSeed)

    updateSelectInput(
      session = session,
      inputId = "team_b_select",
      choices = valid_opponents
    )
  })

  transformed_df <- raw_stat_df %>%
    select(GameID, TeamAName, TeamBName, TeamASeed, TeamBSeed, Season) %>%
    mutate(TeamANameSeed = str_c(TeamAName, TeamASeed, sep = " - ")) %>%
    mutate(TeamBNameSeed = str_c(TeamBName, TeamBSeed, sep = " - ")) %>%
    inner_join(
      diff_stat_df %>%
        select(GameID, AdjT:Wins),
      by = "GameID"
    ) %>%
    pivot_longer(
      cols = AdjT:Wins,
      names_to = "Variable",
      values_to = "DiffValue"
    ) %>%
    inner_join(
      shap_df %>%
        select(GameID, Variable, ShapValue = Value),
      by = c("GameID", "Variable")
    ) %>%
    inner_join(
      prediction_df %>%
        select(GameID, Outcome, PredProb),
      by = c("GameID")
    ) %>%
    group_by(Variable) %>%
    mutate(
      DiffPercOverall = percent_rank(DiffValue),
      ShapPercOverall = percent_rank(ShapValue),
      PredProbPercOverall = percent_rank(PredProb)
    ) %>%
    group_by(Variable, TeamASeed, TeamBSeed) %>%
    mutate(
      DiffPercSeed = percent_rank(DiffValue),
      ShapPercSeed = percent_rank(ShapValue),
      PredProbPercSeed = percent_rank(PredProb)
    ) %>%
    ungroup()

  output$outcome <- renderText({
    transformed_df %>%
      filter(
        Season == input$season_select &
          TeamANameSeed == input$team_a_select &
          TeamBNameSeed == input$team_b_select
      ) %>%
      distinct(Outcome) %>%
      mutate(
        Outcome = ifelse(
          Outcome == 1,
          "Win",
          ifelse(Outcome == 0, "Loss", "No Decision")
        )
      ) %>%
      pull(Outcome)
  })

  output$pred_prob <- renderText({
    transformed_df %>%
      filter(
        Season == input$season_select &
          TeamANameSeed == input$team_a_select &
          TeamBNameSeed == input$team_b_select
      ) %>%
      distinct(PredProb) %>%
      mutate(PredProb = str_c(round(100 * PredProb, 2), "%")) %>%
      pull(PredProb)
  })

  output$pred_perc_overall <- renderText({
    transformed_df %>%
      filter(
        Season == input$season_select &
          TeamANameSeed == input$team_a_select &
          TeamBNameSeed == input$team_b_select
      ) %>%
      distinct(PredProbPercOverall) %>%
      mutate(
        PredProbPercOverall = str_c(round(100 * PredProbPercOverall, 2), "%")
      ) %>%
      pull(PredProbPercOverall)
  })

  output$pred_perc_seed <- renderText({
    transformed_df %>%
      filter(
        Season == input$season_select &
          TeamANameSeed == input$team_a_select &
          TeamBNameSeed == input$team_b_select
      ) %>%
      distinct(PredProbPercSeed) %>%
      mutate(
        PredProbPercSeed = str_c(round(100 * PredProbPercSeed, 2), "%")
      ) %>%
      pull(PredProbPercSeed)
  })

  output$stat_table <- renderDataTable(
    {
      datatable(
        transformed_df %>%
          filter(
            Season == input$season_select &
              TeamANameSeed == input$team_a_select &
              TeamBNameSeed == input$team_b_select
          ) %>%
          arrange(desc(abs(ShapValue))) %>%
          select(
            Variable,
            DiffValue,
            DiffPercOverall,
            DiffPercSeed,
            ShapValue,
            ShapPercOverall,
            ShapPercSeed
          ),
        options = list(dom = "t")
      ) %>%
        formatRound(
          columns = c(
            "DiffValue",
            "DiffPercOverall",
            "DiffPercSeed",
            "ShapValue",
            "ShapPercOverall",
            "ShapPercSeed"
          ),
          digits = 4
        )
    },
  )
}

shinyApp(ui = ui, server = server)
