DEFAULT_MEMORY <- "8GB"
BLOCKING_VARIABLES <- c(
  CI = "Veřejné CI nesmí číst data MIMIC.",
  CLAUDECODE = "Relace Claude Code nesmí číst data MIMIC: výstup by odešel hostovanému modelu."
)

eda_connect <- function() {
  parquet <- Sys.getenv(
    "MEDIPARSE_PARQUET",
    file.path(Sys.getenv("PIXI_PROJECT_ROOT"), "resources", "mimic", "parquet")
  )
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
