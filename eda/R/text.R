PERCENTILES <- c(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
PERCENTILE_COLUMNS <- sprintf("p%02d", as.integer(round(100 * PERCENTILES)))
LENGTH_METRICS <- c(znaky = "chars", slova = "words", znacky = "markers")

metrics_table_name <- function(table) paste0("note_metrics_", table)

note_metrics <- function(con, table) {
  name <- metrics_table_name(table)
  DBI::dbExecute(con, sprintf("
    CREATE OR REPLACE TEMP TABLE %s AS
    SELECT
      note_id,
      length(text) AS chars,
      len(regexp_extract_all(text, '\\S+')) AS words,
      len(list_filter(regexp_extract_all(text, '_+'), x -> length(x) = 3)) AS markers,
      len(list_filter(regexp_extract_all(text, '_+'), x -> length(x) > 3)) AS longer_runs
    FROM (SELECT note_id, coalesce(text, '') AS text FROM %s)
  ", DBI::dbQuoteIdentifier(con, name), DBI::dbQuoteIdentifier(con, table)))
  name
}

metrics_for <- function(con, table) {
  name <- metrics_table_name(table)
  if (name %in% DBI::dbListTables(con)) {
    return(name)
  }
  note_metrics(con, table)
}

per_table <- function(con, tables, query) {
  rows <- lapply(tables, function(table) {
    metrics <- DBI::dbQuoteIdentifier(con, metrics_for(con, table))
    out <- DBI::dbGetQuery(con, query(metrics))
    cbind(tabulka = rep(table, nrow(out)), out, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

as_numeric_columns <- function(df, columns) {
  df[columns] <- lapply(df[columns], as.numeric)
  df
}

percentile_sql <- function(metric, column) {
  quantiles <- sprintf("quantile_disc(%s, %s) AS %s", column, PERCENTILES, PERCENTILE_COLUMNS)
  sprintf(
    "SELECT '%s' AS metrika, count(%s) AS n, avg(%s) AS prumer, %s FROM %%s",
    metric, column, column, paste(quantiles, collapse = ", ")
  )
}

length_percentiles <- function(con, tables) {
  selects <- mapply(percentile_sql, names(LENGTH_METRICS), LENGTH_METRICS, USE.NAMES = FALSE)
  out <- per_table(con, tables, function(metrics) {
    paste(sprintf(selects, metrics), collapse = " UNION ALL ")
  })
  as_numeric_columns(out, c("n", "prumer", PERCENTILE_COLUMNS))
}

format_number <- function(x) {
  format(x, big.mark = NBSP, scientific = FALSE, trim = TRUE)
}

format_percentiles <- function(percentiles) {
  out <- percentiles
  for (index in seq_along(PERCENTILES)) {
    column <- PERCENTILE_COLUMNS[index]
    shown <- dua_quantile_ok(percentiles$n, PERCENTILES[index]) & !is.na(percentiles[[column]])
    out[[column]] <- ifelse(shown, format_number(percentiles[[column]]), "–")
  }
  out$prumer <- ifelse(
    percentiles$n >= DUA_THRESHOLD & !is.na(percentiles$prumer),
    dua_decimal(percentiles$prumer), "–"
  )
  out$n <- dua_count(percentiles$n)
  out
}

length_histogram <- function(con, tables) {
  out <- per_table(con, tables, function(metrics) sprintf("
    SELECT floor(log10(chars) * 10) / 10 AS kos, count(*) AS n
    FROM %s
    WHERE chars > 0
    GROUP BY kos
    ORDER BY kos
  ", metrics))
  as_numeric_columns(out, c("kos", "n"))
}

marker_summary <- function(con, tables) {
  out <- per_table(con, tables, function(metrics) sprintf("
    SELECT
      count(*) AS zprav,
      coalesce(sum(markers), 0) AS znacek,
      count(*) FILTER (WHERE markers > 0) AS se_znackou,
      coalesce(sum(longer_runs), 0) AS delsi_useky
    FROM %s
  ", metrics))
  as_numeric_columns(out, c("zprav", "znacek", "se_znackou", "delsi_useky"))
}
