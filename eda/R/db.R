# Pravidla přístupu k datům (adresář, stráž, paměť, spill, pohledy) mají jedinou kopii
# v Pythonu; R podle nich jen otevře vlastní spojení, dokud se nepřepíše poslední R sešit.
eda_database <- function() reticulate::import("mediparse.entrypoints.eda_database")

# Prostředí se předává explicitně, protože `os.environ` pod reticulate je snímek ze startu
# interpretu a pozdější `Sys.setenv` (i `withr::local_envvar`) by nezachytil.
eda_environ <- function() as.list(Sys.getenv())

eda_parquet_dir <- function() eda_database()$parquet_dir(eda_environ())

eda_connect <- function(database = eda_database()) {
  rules <- database$connection_settings(eda_environ())
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
