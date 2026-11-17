feature_source <- function(con, file, columns, directory = Sys.getenv("MEDIPARSE_EDA_CACHE"), command = "mediparse-eda-note-features") {
  if (directory == "") stop("Chybí MEDIPARSE_EDA_CACHE; spusťte render skriptem EDA.")
  if (!file.exists(file.path(directory, "cache.json"))) {
    stop(sprintf("cache není publikovaná; spusťte %s", command))
  }
  path <- file.path(directory, file)
  if (!file.exists(path)) stop(sprintf("Soubor %s neexistuje; spusťte nástroj %s.", path, command))
  source <- sprintf("read_parquet(%s)", DBI::dbQuoteString(con, path))
  check_contract(con, source, columns)
  source
}

feature_table <- function(con, name, file, columns, key, complete) {
  if (name %in% DBI::dbListTables(con)) return(name)
  source <- feature_source(con, file, columns)
  # Materializace zabrání opakování výběru korpusu (okno přes otisky discharge)
  # v každém z mnoha agregačních dotazů sešitu.
  DBI::dbExecute(con, sprintf(
    "CREATE TEMP TABLE %s AS SELECT f.* FROM %s f JOIN (%s) c USING (note_id)",
    DBI::dbQuoteIdentifier(con, name), source, input_corpus_sql(con)
  ))
  table <- DBI::dbQuoteIdentifier(con, name)
  duplicates <- DBI::dbGetQuery(con, sprintf(
    "SELECT count(*) AS n FROM (SELECT 1 FROM %s GROUP BY %s HAVING count(*) > 1)",
    table, paste(DBI::dbQuoteIdentifier(con, key), collapse = ", ")
  ))$n
  if (duplicates > 0) {
    stop(sprintf("Soubor %s má %s duplicitních klíčů; spusťte mediparse-eda-note-features.", file, duplicates))
  }
  if (complete) {
    missing <- DBI::dbGetQuery(con, sprintf(
      "SELECT count(*) AS n FROM (%s) c WHERE NOT EXISTS (SELECT 1 FROM %s f WHERE f.note_id = c.note_id)",
      input_corpus_sql(con), table
    ))$n
    if (missing > 0) {
      stop(sprintf("V souboru %s chybí %s zpráv; spusťte mediparse-eda-note-features.", file, missing))
    }
  }
  name
}
