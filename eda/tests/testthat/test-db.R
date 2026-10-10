write_parquet_tables <- function(dir, marker = TRUE) {
  dir.create(dir, recursive = TRUE)
  if (marker) writeLines('{"prefix":"t","rows":1}', file.path(dir, "synthetic.json"))
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  for (name in c("discharge", "admissions")) {
    DBI::dbExecute(con, sprintf(
      "COPY (SELECT 'x' AS a, 1 AS b) TO '%s' (FORMAT parquet)",
      file.path(dir, paste0(name, ".parquet"))
    ))
  }
}

setting <- function(con, name) {
  DBI::dbGetQuery(con, sprintf("SELECT current_setting('%s') AS v", name))$v
}

test_that("eda_connect vytvoří view pro každý parquet", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_setequal(DBI::dbListTables(con), c("discharge", "admissions"))
  expect_equal(DBI::dbGetQuery(con, "SELECT count(*) AS n FROM discharge")$n, 1)
})

test_that("eda_connect nastaví temp_directory vedle parquetu", {
  base <- withr::local_tempdir()
  dir <- file.path(base, "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(
    normalizePath(setting(con, "temp_directory"), mustWork = FALSE),
    normalizePath(file.path(base, "duckdb-tmp"), mustWork = FALSE)
  )
})

test_that("eda_connect nastaví memory_limit z proměnné prostředí", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, MEDIPARSE_DUCKDB_MEMORY = "2GiB")
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(setting(con, "memory_limit"), "2.0 GiB")
})

test_that("eda_connect má výchozí memory_limit 8GB", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, MEDIPARSE_DUCKDB_MEMORY = NA)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(setting(con, "memory_limit"), "7.4 GiB")
})

test_that("eda_connect odmítne chybějící adresář", {
  withr::local_envvar(MEDIPARSE_PARQUET = file.path(withr::local_tempdir(), "none"))

  expect_error(eda_connect(), "neexistuje")
})

test_that("eda_connect odmítne adresář bez parquetu", {
  dir <- withr::local_tempdir()
  withr::local_envvar(MEDIPARSE_PARQUET = dir)

  expect_error(eda_connect(), "žádný parquet")
})

test_that("eda_connect odmítne skutečná data pod CLAUDECODE", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = "1", CI = NA)

  expect_error(eda_connect(), "Claude Code")
})

test_that("eda_connect odmítne skutečná data v CI", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = NA, CI = "true")

  expect_error(eda_connect(), "CI")
})

test_that("eda_connect pod CLAUDECODE projde se syntetickou značkou", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = "1", CI = "true")
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_true("discharge" %in% DBI::dbListTables(con))
})

test_that("eda_connect bez blokujících proměnných projde i bez značky", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = NA, CI = NA)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_true("discharge" %in% DBI::dbListTables(con))
})

test_that("eda_check_python projde s Pythonem z prostředí", {
  prefix <- dirname(dirname(Sys.getenv("RETICULATE_PYTHON")))
  withr::local_envvar(CONDA_PREFIX = prefix)

  expect_no_error(eda_check_python())
})

test_that("eda_check_python selže, když Python neleží v prostředí", {
  withr::local_envvar(CONDA_PREFIX = file.path(withr::local_tempdir(), "jine"))

  expect_error(eda_check_python(), "mimo prostředí")
})

test_that("eda_parquet_dir má výchozí adresář pod kořenem projektu", {
  withr::local_envvar(MEDIPARSE_PARQUET = NA, PIXI_PROJECT_ROOT = "/koren")

  expect_equal(eda_parquet_dir(), "/koren/resources/mimic/parquet")
})
