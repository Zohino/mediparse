feature_source <- function(con, file, columns) {
  directory <- Sys.getenv("MEDIPARSE_EDA_CACHE")
  if (directory == "") stop("Chybí MEDIPARSE_EDA_CACHE; spusťte render skriptem EDA.")
  path <- file.path(directory, file)
  if (!file.exists(path)) stop(sprintf("Soubor %s neexistuje; spusťte nástroj mediparse-eda-note-features.", path))
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
