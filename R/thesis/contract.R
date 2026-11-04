check_contract <- function(con, source, columns) {
  schema <- DBI::dbGetQuery(con, paste("DESCRIBE SELECT * FROM", source))
  for (name in names(columns)) {
    actual <- schema$column_type[match(name, schema$column_name)]
    if (is.na(actual)) {
      stop(sprintf(
        "Zdroj %s: sloupec %s, očekávaný typ %s, skutečný typ chybí.",
        source, name, columns[[name]]
      ))
    }
    if (actual != columns[[name]]) {
      stop(sprintf(
        "Zdroj %s: sloupec %s, očekávaný typ %s, skutečný typ %s.",
        source, name, columns[[name]], actual
      ))
    }
  }
  identifiers <- DBI::dbQuoteIdentifier(con, names(columns))
  null_counts <- DBI::dbGetQuery(con, sprintf(
    "SELECT %s FROM %s",
    paste(sprintf("count(*) - count(%s)", identifiers), collapse = ", "), source
  ))
  for (i in seq_along(columns)) {
    if (null_counts[[i]] > 0) {
      stop(sprintf(
        "Zdroj %s: sloupec %s, očekávaný typ %s bez NULL, skutečný typ %s s NULL.",
        source, names(columns)[i], columns[[i]], columns[[i]]
      ))
    }
  }
  invisible(TRUE)
}

read_contract_table <- function(path, columns) {
  if (!file.exists(path)) {
    stop(sprintf("Soubor %s neexistuje.", path))
  }
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  source <- sprintf("read_parquet(%s)", DBI::dbQuoteString(con, path))
  check_contract(con, source, columns)
  DBI::dbGetQuery(con, sprintf(
    "SELECT %s FROM %s", paste(DBI::dbQuoteIdentifier(con, names(columns)), collapse = ", "), source
  ))
}
