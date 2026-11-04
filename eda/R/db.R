project_path <- function(...) {
  root <- Sys.getenv("PIXI_PROJECT_ROOT", file.path("..", "..", ".."))
  file.path(root, ...)
}

config_path <- function(name) project_path("config", name)

scalar_text <- function(value) {
  is.character(value) && length(value) == 1L && !is.na(value) && nzchar(value)
}

object_keys <- function(value, keys) {
  is.list(value) && length(value) == length(keys) && setequal(names(value), keys)
}

input_corpus_rule <- function(path = config_path("input_corpus.json")) {
  if (!file.exists(path)) stop("Soubor input_corpus.json neexistuje: ", path, call. = FALSE)
  rule <- tryCatch(jsonlite::read_json(path), error = function(error) {
    stop("Soubor input_corpus.json nelze přečíst: ", path, call. = FALSE)
  })
  valid <- object_keys(rule, "min_chars") && is.integer(rule$min_chars) &&
    length(rule$min_chars) == 1L && !is.na(rule$min_chars) && rule$min_chars >= 1L
  if (!valid) stop("Neplatný config input_corpus.json: ", path, call. = FALSE)
  rule
}

data_access_rules <- function(path = project_path("src", "mediparse", "domain", "data_access.json")) {
  if (!file.exists(path)) stop("Soubor pravidel data_access.json neexistuje: ", path, call. = FALSE)
  rules <- tryCatch(jsonlite::read_json(path), error = function(error) {
    stop("Soubor pravidel data_access.json nelze přečíst: ", path, call. = FALSE)
  })
  valid <- object_keys(rules, "blocking") && is.list(rules$blocking) &&
    is.null(names(rules$blocking)) && length(rules$blocking) > 0L &&
    all(vapply(rules$blocking, function(variable) {
      object_keys(variable, c("name", "reason")) &&
        scalar_text(variable$name) && scalar_text(variable$reason)
    }, logical(1)))
  if (!valid) stop("Neplatná pravidla data_access.json: ", path, call. = FALSE)
  names <- vapply(rules$blocking, `[[`, character(1), "name")
  if (anyDuplicated(names)) stop("Duplicitní jména v data_access.json.", call. = FALSE)
  rules$blocking
}

eda_parquet_dir <- function() {
  given <- Sys.getenv("MEDIPARSE_PARQUET")
  if (nzchar(given)) given else project_path("resources", "mimic", "parquet")
}

eda_synthetic_marker <- function(directory) {
  path <- file.path(directory, "synthetic.json")
  if (!file.exists(path)) return(FALSE)
  marker <- tryCatch(jsonlite::read_json(path), error = function(error) {
    stop("Neplatná značka synthetic.json.", call. = FALSE)
  })
  # R smí být přísnější: typ integer odmítne i velké JSON floaty, které Python nepřevede na int.
  valid <- object_keys(marker, c("prefix", "rows")) && scalar_text(marker$prefix) &&
    is.integer(marker$rows) && length(marker$rows) == 1L && !is.na(marker$rows)
  if (!valid) stop("Neplatná značka synthetic.json.", call. = FALSE)
  TRUE
}

eda_connect <- function() {
  directory <- eda_parquet_dir()
  if (!dir.exists(directory)) stop("Adresář s parquetem neexistuje: ", directory, call. = FALSE)
  directory <- normalizePath(directory, mustWork = TRUE)
  files <- sort(list.files(directory, pattern = "\\.parquet$", full.names = TRUE))
  if (length(files) == 0L) stop("Adresář neobsahuje žádný parquet: ", directory, call. = FALSE)
  synthetic <- eda_synthetic_marker(directory)
  rules <- data_access_rules()
  for (variable in rules) {
    if (!synthetic && nzchar(Sys.getenv(variable$name))) {
      stop("Adresář neobsahuje synthetic.json, takže jde o skutečná data. ", variable$reason, call. = FALSE)
    }
  }
  memory <- Sys.getenv("MEDIPARSE_DUCKDB_MEMORY")
  if (!nzchar(memory)) memory <- "8GB"
  temp <- file.path(dirname(directory), "duckdb-tmp")
  con <- DBI::dbConnect(duckdb::duckdb())
  complete <- FALSE
  on.exit(if (!complete) DBI::dbDisconnect(con, shutdown = TRUE))
  invisible(DBI::dbExecute(con, paste("SET temp_directory =", DBI::dbQuoteString(con, temp))))
  invisible(DBI::dbExecute(con, paste("SET memory_limit =", DBI::dbQuoteString(con, memory))))
  for (path in files) {
    name <- tools::file_path_sans_ext(basename(path))
    invisible(DBI::dbExecute(con, sprintf(
      "CREATE VIEW %s AS SELECT * FROM read_parquet(%s)",
      DBI::dbQuoteIdentifier(con, name), DBI::dbQuoteString(con, path)
    )))
  }
  complete <- TRUE
  con
}
