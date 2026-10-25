FUNNEL_STEPS <- c(
  "Všechny propouštěcí zprávy",
  "S kódem ICD",
  sprintf("Aspoň %d znaků", SHORT_LIMIT),
  "Bez přesných duplicit"
)

eligible_cte <- function(con) {
  metrics <- DBI::dbQuoteIdentifier(con, metrics_for(con, "discharge"))
  sprintf("
    WITH base AS (
      SELECT
        d.note_id,
        d.subject_id,
        d.hadm_id,
        d.hadm_id IS NOT NULL AND EXISTS (
          SELECT 1 FROM diagnoses_icd i WHERE i.hadm_id = d.hadm_id
        ) AS has_icd,
        coalesce(m.chars, 0) >= %d AS long_enough,
        m.fingerprint
      FROM discharge d
      JOIN %s m ON m.note_id = d.note_id
    ),
    eligible AS (
      SELECT *, row_number() OVER (PARTITION BY fingerprint ORDER BY note_id) AS copy
      FROM base
      WHERE has_icd AND long_enough
    )
  ", SHORT_LIMIT, metrics)
}

input_corpus_sql <- function(con) {
  paste(
    eligible_cte(con),
    "SELECT note_id, subject_id, hadm_id FROM eligible WHERE copy = 1"
  )
}

# Texty vstupního korpusu: tatáž definice pro každý průchod sešitu 9, který text posílá mimo DuckDB.
input_texts_sql <- function(con) {
  sprintf(
    "SELECT c.note_id, d.text FROM (%s) c JOIN discharge d ON d.note_id = c.note_id",
    input_corpus_sql(con)
  )
}

corpus_funnel <- function(con) {
  rows <- DBI::dbGetQuery(con, paste0(eligible_cte(con), ",
    steps AS (
      SELECT 1 AS step, subject_id FROM base
      UNION ALL SELECT 2, subject_id FROM base WHERE has_icd
      UNION ALL SELECT 3, subject_id FROM eligible
      UNION ALL SELECT 4, subject_id FROM eligible WHERE copy = 1
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
  "))
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
