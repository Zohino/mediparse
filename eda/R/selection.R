FUNNEL_STEPS <- c(
  "Všechny propouštěcí zprávy",
  "S kódem ICD",
  sprintf("Aspoň %d znaků", SHORT_LIMIT),
  "Bez přesných duplicit"
)

INPUT_CORPUS_TABLE <- "input_corpus"

input_corpus_table <- function(con) {
  if (INPUT_CORPUS_TABLE %in% DBI::dbListTables(con)) return(INPUT_CORPUS_TABLE)
  path <- project_path("src", "mediparse", "infrastructure", "input_corpus.sql")
  # DuckDB váže tabulky při definici makra, proto musí pohledy už existovat.
  invisible(DBI::dbExecute(con, paste(readLines(path, encoding = "UTF-8"), collapse = "\n")))
  invisible(DBI::dbExecute(con, sprintf(
    "CREATE TEMP TABLE %s AS SELECT * FROM input_corpus_selection(?)",
    DBI::dbQuoteIdentifier(con, INPUT_CORPUS_TABLE)
  ), params = list(SHORT_LIMIT)))
  INPUT_CORPUS_TABLE
}

input_corpus_sql <- function(con) {
  sprintf(
    "SELECT note_id, subject_id, hadm_id FROM %s WHERE copy = 1",
    DBI::dbQuoteIdentifier(con, input_corpus_table(con))
  )
}

corpus_funnel <- function(con) {
  table <- DBI::dbQuoteIdentifier(con, input_corpus_table(con))
  rows <- DBI::dbGetQuery(con, sprintf("
    WITH steps AS (
      SELECT 1 AS step, subject_id FROM %s
      UNION ALL SELECT 2, subject_id FROM %s WHERE has_icd
      UNION ALL SELECT 3, subject_id FROM %s WHERE copy IS NOT NULL
      UNION ALL SELECT 4, subject_id FROM %s WHERE copy = 1
    ),
    per_patient AS (
      SELECT step, subject_id, count(*) AS notes FROM steps GROUP BY step, subject_id
    )
    SELECT
      step,
      sum(notes) AS zprav,
      count(*) AS pacientu,
      count(*) FILTER (WHERE notes > 1) AS vice_zprav
    FROM per_patient
    GROUP BY step
  ", table, table, table, table))
  index <- match(seq_along(FUNNEL_STEPS), rows$step)
  counts <- function(column) {
    out <- as.numeric(rows[[column]][index])
    out[is.na(out)] <- 0
    out
  }
  data.frame(
    krok = FUNNEL_STEPS,
    zprav = counts("zprav"),
    pacientu = counts("pacientu"),
    vice_zprav = counts("vice_zprav"),
    stringsAsFactors = FALSE
  )
}

COMPARISON_ROWS <- c(
  "Zpráv",
  "Pacientů",
  "Bez hadm_id (%)",
  "S kódem ICD (%)",
  "Medián délky (znaky)",
  "95. percentil délky (znaky)",
  sprintf("Prázdné nebo kratší než %d znaků", SHORT_LIMIT),
  "Přesné duplicity"
)

row_of <- function(df, table) df[df$tabulka == table, , drop = FALSE]

comparison_column <- function(table, summaries, coverage, quality, shown) {
  covered <- row_of(coverage, table)
  checked <- row_of(quality, table)
  patients <- summaries[[table]]$patients
  total <- if (nrow(covered) == 0L) NA else covered$zprav
  share <- function(column) {
    if (nrow(covered) == 0L) "–" else percent_text(covered[[column]], covered$zprav)
  }
  count <- function(value) if (length(value) == 0L) "–" else number_text(value)
  c(
    count(total),
    count(patients),
    share("bez_hadm"),
    share("s_icd"),
    percentile_cell(shown, table, "znaky", "p50"),
    percentile_cell(shown, table, "znaky", "p95"),
    count(checked$prazdne + checked$pod_100),
    count(checked$duplicity)
  )
}

corpus_comparison <- function(summaries, coverage, quality, shown,
                              tables = c("discharge", "radiology")) {
  out <- data.frame(ukazatel = COMPARISON_ROWS, stringsAsFactors = FALSE)
  for (table in tables) {
    out[[table]] <- comparison_column(table, summaries, coverage, quality, shown)
  }
  out
}

corpus_size <- function(con) {
  as.numeric(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM (%s)", input_corpus_sql(con)))$n)
}
