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

feature_table <- function(con, name, file, columns) {
  if (name %in% DBI::dbListTables(con)) return(name)
  source <- feature_source(con, file, columns)
  # Materializace zabrání opakování výběru korpusu (okno přes otisky discharge)
  # v každém z mnoha agregačních dotazů sešitu.
  DBI::dbExecute(con, sprintf(
    "CREATE TEMP TABLE %s AS SELECT f.* FROM %s f JOIN (%s) c USING (note_id)",
    DBI::dbQuoteIdentifier(con, name), source, input_corpus_sql(con)
  ))
  name
}
