library(shiny)
library(DBI)
library(RPostgres)
library(DT)
library(plotly)
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
      separate_wider_delim(
        cols = GameID,
        delim = "-",
        names = c("Season", "IDA", "IDB")
      ) %>%
      mutate(
        Outcome = as.integer(ifelse(Outcome == 1, 0, 1)),
        GameID = str_c(Season, IDB, IDA, sep = "-"),
        PredProb = 1 - PredProb
      ) %>%
      select(-c(Season, IDA, IDB))
  )
shap_df <- shap_df %>%
  bind_rows(
    shap_df %>%
      separate_wider_delim(
        cols = GameID,
        delim = "-",
        names = c("Season", "IDA", "IDB")
      ) %>%
      mutate(
        GameID = str_c(Season, IDB, IDA, sep = "-"),
        Value = Value * -1
      ) %>%
      select(-c(Season, IDA, IDB))
  )


ui <- fluidPage(
  titlePanel("NCAA Tournament Analysis"),

  fluidRow(
    column(
      width = 2,
      style = "margin-top: 12.5px",
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
      style = "padding-left: 40px; margin-top: 12.5px",
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
  ),

  fluidRow(
    column(
      width = 4,
      plotlyOutput(outputId = "accuracy_plot")
    ),
    column(
      width = 4,
      plotlyOutput(outputId = "pred_plot")
    ),
    column(
      width = 4,
      plotlyOutput(outputId = "shap_plot")
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
          mutate(
            DiffPercOverall = str_c(round(100 * DiffPercOverall, 2), "%"),
            DiffPercSeed = str_c(round(100 * DiffPercSeed, 2), "%"),
            ShapValue = str_c(round(100 * ShapValue, 2), "%"),
            ShapPercOverall = str_c(round(100 * ShapPercOverall, 2), "%"),
            ShapPercSeed = str_c(round(100 * ShapPercSeed, 2), "%")
          ) %>%
          select(
            Variable,
            DiffValue,
            DiffPercOverall,
            DiffPercSeed,
            ShapValue,
            ShapPercOverall,
            ShapPercSeed
          ),
        selection = "single",
        options = list(
          dom = "tp",
          columnDefs = list(list(className = "dt-left", targets = "_all")),
          pageLength = 5
        )
      ) %>%
        formatRound(
          columns = c(
            "DiffValue"
          ),
          digits = 2
        )
    },
  )

  observeEvent(
    {
      input$season_select
      input$team_a_select
      input$team_b_select
    },
    {
      selectRows(dataTableProxy("stat_table"), 1)
    }
  )

  selected_variable <- reactive({
    req(input$stat_table_rows_selected)

    transformed_df %>%
      filter(
        Season == input$season_select,
        TeamANameSeed == input$team_a_select,
        TeamBNameSeed == input$team_b_select
      ) %>%
      arrange(desc(abs(ShapValue))) %>%
      slice(input$stat_table_rows_selected) %>%
      pull(Variable)
  })

  output$accuracy_plot <- renderPlotly(
    {
      p <- transformed_df %>%
        filter(
          TeamASeed == as.integer(str_extract(input$team_a_select, "\\d+")) &
            TeamBSeed == as.integer(str_extract(input$team_b_select, "\\d+"))
        ) %>%
        distinct(GameID, .keep_all = T) %>%
        filter(!is.na(Outcome)) %>%
        mutate(ProbBin = ntile(PredProb, 5)) %>%
        group_by(ProbBin, Outcome) %>%
        summarise(n = n()) %>%
        group_by(ProbBin) %>%
        mutate(
          total = sum(n),
          pct = n / sum(n)
        ) %>%
        ggplot(
          aes(
            x = ProbBin,
            y = n,
            fill = factor(Outcome),
            text = str_c(
              "Outcome: ",
              Outcome,
              "<br>Count: ",
              n,
              "<br>Bin total: ",
              total,
              "<br>Win rate: ",
              str_c(round(100 * pct, 2), "%")
            )
          )
        ) +
        geom_col() +
        geom_text(
          data = ~ filter(.x, Outcome == 1),
          aes(label = str_c(round(100 * pct, 2), "%")),
          position = position_stack(vjust = 0.5),
          size = 3
        ) +
        scale_fill_manual(values = c("0" = "#ff6b6b", "1" = "#4ecdc4")) +
        labs(
          title = "Win rate by prediction bucket",
          fill = "Outcome"
        )

      plt <- ggplotly(p, tooltip = c("text"))

      for (i in seq_along(plt$x$data)) {
        type <- plt$x$data[[i]]$type

        if (type == "bar") {
          plt$x$data[[i]]$hovertemplate <- "%{text}<extra></extra>"
        } else {
          plt$x$data[[i]]$hoverinfo <- "none"
        }
      }

      plt %>%
        layout(
          hoverlabel = list(align = "left"),
          legend = list(
            orientation = "h",
            x = 0.5,
            y = -0.2,
            xanchor = "center"
          )
        )
    }
  )

  output$pred_plot <- renderPlotly(
    {
      p <- transformed_df %>%
        filter(
          TeamASeed == as.integer(str_extract(input$team_a_select, "\\d+")) &
            TeamBSeed == as.integer(str_extract(input$team_b_select, "\\d+"))
        ) %>%
        distinct(GameID, .keep_all = T) %>%
        ggplot(aes(x = PredProb)) +
        geom_density() +
        geom_jitter(
          aes(
            x = PredProb,
            y = 0,
            fill = factor(Outcome),
            text = str_c(
              "Season: ",
              Season,
              "<br>Team: ",
              TeamANameSeed,
              "<br>Opponent: ",
              TeamBNameSeed,
              "<br>Predicted probability: ",
              str_c(round(100 * PredProb, 2), "%")
            )
          ),
          size = 2.5,
          alpha = .7
        ) +
        labs(
          title = "Prediction distribution",
          fill = "Outcome"
        )

      ggplotly(p, tooltip = c("text")) %>%
        style(hoverinfo = "none", traces = 1) %>%
        layout(
          hoverlabel = list(align = "left"),
          legend = list(
            orientation = "h",
            x = 0.5,
            y = -0.2,
            xanchor = "center"
          )
        )
    }
  )

  output$shap_plot <- renderPlotly(
    {
      req(selected_variable())

      p <- transformed_df %>%
        filter(
          TeamASeed == as.integer(str_extract(input$team_a_select, "\\d+")) &
            TeamBSeed == as.integer(str_extract(input$team_b_select, "\\d+"))
        ) %>%
        filter(Variable == selected_variable()) %>%
        ggplot(
          aes(
            x = ShapValue,
            y = DiffValue,
            fill = factor(Outcome),
            text = str_c(
              "Season: ",
              Season,
              "<br>Team: ",
              TeamANameSeed,
              "<br>Opponent: ",
              TeamBNameSeed,
              "<br>Predicted probability: ",
              str_c(round(100 * PredProb, 2), "%"),
              "<br>Shap value: ",
              str_c(round(100 * ShapValue, 2), "%")
            )
          )
        ) +
        geom_point(pch = 21, alpha = .7, size = 2.5) +
        labs(
          title = str_c("Shap value distribution: ", selected_variable()),
          fill = "Outcome"
        )

      ggplotly(p, tooltip = c("text")) %>%
        layout(
          hoverlabel = list(align = "left"),
          legend = list(
            orientation = "h",
            x = 0.5,
            y = -0.2,
            xanchor = "center"
          )
        )
    }
  )
}

shinyApp(ui = ui, server = server)
