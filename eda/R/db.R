# Pravidla přístupu k datům (adresář, stráž, paměť, spill, pohledy) mají jedinou kopii
# v Pythonu; R podle nich jen otevře vlastní spojení, dokud se nepřepíše poslední R sešit.
eda_database <- function() reticulate::import("mediparse.entrypoints.eda_database")

# Prostředí se předává explicitně, protože `os.environ` pod reticulate je snímek ze startu
# interpretu a pozdější `Sys.setenv` (i `withr::local_envvar`) by nezachytil.
eda_environ <- function() as.list(Sys.getenv())

eda_parquet_dir <- function() eda_database()$parquet_dir(eda_environ())

eda_connect <- function() {
  rules <- eda_database()$connection_settings(eda_environ())
  con <- DBI::dbConnect(duckdb::duckdb())
  invisible(DBI::dbExecute(
    con, paste("SET temp_directory =", DBI::dbQuoteString(con, rules$temp_directory))
  ))
  invisible(DBI::dbExecute(
    con, paste("SET memory_limit =", DBI::dbQuoteString(con, rules$memory_limit))
  ))
  for (name in names(rules$views)) {
    invisible(DBI::dbExecute(con, sprintf(
      "CREATE VIEW %s AS SELECT * FROM read_parquet(%s)",
      DBI::dbQuoteIdentifier(con, name), DBI::dbQuoteString(con, rules$views[[name]])
    )))
  }
  con
}

eda_check_python <- function() {
  prefix <- normalizePath(Sys.getenv("CONDA_PREFIX"), mustWork = FALSE)
  python <- normalizePath(reticulate::py_config()$python, mustWork = FALSE)
  if (prefix == "" || !startsWith(python, paste0(prefix, "/"))) {
    stop(sprintf("Python %s leží mimo prostředí pixi %s.", python, prefix))
  }
  invisible(python)
}

python_path <- function(path) reticulate::import("pathlib")$Path(path)

config_path <- function(name) {
  root <- Sys.getenv("PIXI_PROJECT_ROOT", file.path("..", "..", ".."))
  file.path(root, "config", name)
}

# Dávky jdou po `batch` zprávách, protože texty se předávají mimo DuckDB (tokenizér,
# parser sekcí přes reticulate) a celý korpus se do paměti R nevejde.
batched_table <- function(con, name, columns, rows_sql, fn, batch = 2000L) {
  if (name %in% DBI::dbListTables(con)) {
    return(name)
  }
  rows <- paste0(name, "_rows")
  quote <- function(identifier) DBI::dbQuoteIdentifier(con, identifier)
  definition <- paste(quote(names(columns)), columns, collapse = ", ")
  DBI::dbExecute(con, sprintf("CREATE TEMP TABLE %s (%s)", quote(name), definition))
  done <- FALSE
  on.exit({
    DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", quote(rows)))
    if (!done) DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", quote(name)))
  })
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    SELECT row_number() OVER (ORDER BY note_id) AS rn, note_id, text FROM (%s)
  ", quote(rows), rows_sql))
  total <- as.numeric(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", quote(rows)))$n)
  for (index in seq_len(ceiling(total / batch))) {
    chunk <- DBI::dbGetQuery(con, sprintf(
      "SELECT note_id, text FROM %s WHERE rn BETWEEN %.0f AND %.0f ORDER BY rn",
      quote(rows), (index - 1) * batch + 1, index * batch
    ))
    DBI::dbAppendTable(con, name, fn(chunk))
  }
  done <- TRUE
  name
}
