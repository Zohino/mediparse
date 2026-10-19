DEFAULT_MEMORY <- "8GB"
BLOCKING_VARIABLES <- c(
  CI = "Veřejné CI nesmí číst data MIMIC.",
  CLAUDECODE = "Relace Claude Code nesmí číst data MIMIC: výstup by odešel hostovanému modelu."
)

eda_parquet_dir <- function() {
  Sys.getenv(
    "MEDIPARSE_PARQUET",
    file.path(Sys.getenv("PIXI_PROJECT_ROOT"), "resources", "mimic", "parquet")
  )
}

eda_connect <- function() {
  parquet <- eda_parquet_dir()
  if (!dir.exists(parquet)) {
    stop(sprintf("Adresář s parquetem %s neexistuje.", parquet))
  }
  files <- list.files(parquet, pattern = "\\.parquet$", full.names = TRUE)
  if (length(files) == 0) {
    stop(sprintf("V adresáři %s není žádný parquet.", parquet))
  }
  if (!file.exists(file.path(parquet, "synthetic.json"))) {
    for (name in names(BLOCKING_VARIABLES)) {
      if (nzchar(Sys.getenv(name))) {
        stop(sprintf(
          "Adresář %s neobsahuje synthetic.json, takže jde o skutečná data. %s",
          parquet, BLOCKING_VARIABLES[[name]]
        ))
      }
    }
  }
  memory <- Sys.getenv("MEDIPARSE_DUCKDB_MEMORY", DEFAULT_MEMORY)
  if (memory == "") memory <- DEFAULT_MEMORY
  temp <- file.path(dirname(normalizePath(parquet)), "duckdb-tmp")
  con <- DBI::dbConnect(duckdb::duckdb())
  invisible(DBI::dbExecute(
    con, paste("SET temp_directory =", DBI::dbQuoteString(con, temp))
  ))
  invisible(DBI::dbExecute(
    con, paste("SET memory_limit =", DBI::dbQuoteString(con, memory))
  ))
  for (file in files) {
    name <- tools::file_path_sans_ext(basename(file))
    invisible(DBI::dbExecute(con, sprintf(
      "CREATE VIEW %s AS SELECT * FROM read_parquet(%s)",
      DBI::dbQuoteIdentifier(con, name), DBI::dbQuoteString(con, file)
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
