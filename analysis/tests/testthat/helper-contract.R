METRICS_FIXTURE <- "../../../tests/fixtures/contracts/metrics.parquet"

write_contract_parquet <- function(query, path = tempfile(fileext = ".parquet")) {
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  DBI::dbExecute(con, sprintf(
    "COPY (%s) TO %s (FORMAT parquet)", query, DBI::dbQuoteString(con, path)
  ))
  path
}
