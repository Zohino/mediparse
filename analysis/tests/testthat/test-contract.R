write_contract_parquet <- function(query, path = tempfile(fileext = ".parquet")) {
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  DBI::dbExecute(con, sprintf(
    "COPY (%s) TO %s (FORMAT parquet)", query, DBI::dbQuoteString(con, path)
  ))
  path
}

test_that("kontrakt odmítne INTEGER místo BIGINT s kontextem chyby", {
  path <- write_contract_parquet("SELECT 1::BIGINT AS ok, 1::INTEGER AS n")
  withr::defer(unlink(path))

  expect_error(
    read_contract_table(path, c(ok = "BIGINT", n = "BIGINT")),
    paste0(basename(path), ".*sloupec n.*BIGINT.*INTEGER")
  )
})

test_that("kontrakt odmítne chybějící deklarovaný sloupec", {
  path <- write_contract_parquet("SELECT 1::BIGINT AS other")
  withr::defer(unlink(path))

  expect_error(
    read_contract_table(path, c(missing_n = "BIGINT")),
    paste0(basename(path), ".*sloupec missing_n.*BIGINT.*chybí")
  )
})

test_that("kontrakt odmítne NULL v deklarovaném sloupci", {
  path <- write_contract_parquet("SELECT 1::BIGINT AS ok, NULL::BIGINT AS n")
  withr::defer(unlink(path))

  expect_error(
    read_contract_table(path, c(ok = "BIGINT", n = "BIGINT")),
    paste0(basename(path), ".*sloupec n.*BIGINT.*NULL")
  )
})

test_that("kontrakt odmítne neexistující soubor se jménem souboru", {
  path <- tempfile(pattern = "missing-contract-", fileext = ".parquet")

  expect_error(
    read_contract_table(path, c(n = "BIGINT")),
    paste0("Soubor .*", basename(path), ".*neexistuje")
  )
})

test_that("kontrakt vrátí jen deklarované sloupce v deklarovaném pořadí", {
  path <- write_contract_parquet(
    'SELECT 7::BIGINT AS n, 0.5::DOUBLE AS "a""b", NULL::INTEGER AS ignored',
    tempfile(pattern = "contract's-", fileext = ".parquet")
  )
  withr::defer(unlink(path))

  actual <- read_contract_table(path, c('a"b' = "DOUBLE", n = "BIGINT"))

  expected <- data.frame('a"b' = 0.5, n = 7, check.names = FALSE)
  expect_identical(actual, expected)
})
