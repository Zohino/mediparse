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

test_that("input_corpus_rule splňuje sdílené případy Pythonu a R", {
  cases <- jsonlite::read_json(project_path("tests", "fixtures", "input_corpus_rule_cases.json"))
  path <- file.path(withr::local_tempdir(), "rule.json")
  for (case in cases) {
    writeLines(case$text, path)
    if (case$valid) {
      expect_no_error(input_corpus_rule(path))
    } else {
      expect_error(input_corpus_rule(path), "input_corpus.json", info = case$name)
    }
  }
})

test_that("input_corpus_rule odmítne chybějící i nečitelný soubor", {
  directory <- withr::local_tempdir()
  expect_error(input_corpus_rule(file.path(directory, "missing.json")), "input_corpus.json")
  expect_error(suppressWarnings(input_corpus_rule(directory)), "input_corpus.json")
})

test_that("SHORT_LIMIT odpovídá schválenému prahu sto znaků", {
  expect_identical(SHORT_LIMIT, 100L)
})

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

test_that("eda_connect odmítne neplatnou značku synthetic.json bez blokujících proměnných", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  writeLines("{}", file.path(dir, "synthetic.json"))
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = NA, CI = NA)

  expect_error(eda_connect(), "Neplatná značka")
})

test_that("eda_connect čte prostředí při každém volání, ne snímek ze startu", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, MEDIPARSE_DUCKDB_MEMORY = NA)
  first <- eda_connect()
  DBI::dbDisconnect(first, shutdown = TRUE)
  withr::local_envvar(MEDIPARSE_DUCKDB_MEMORY = "3GiB")
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(setting(con, "memory_limit"), "3.0 GiB")
})

test_that("eda_connect založí pohled se jménem s pomlčkou", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  file.rename(file.path(dir, "discharge.parquet"), file.path(dir, "mimic-iv-bhc.parquet"))
  withr::local_envvar(MEDIPARSE_PARQUET = dir)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(DBI::dbGetQuery(con, 'SELECT count(*) AS n FROM "mimic-iv-bhc"')$n, 1)
})

test_that("eda_parquet_dir má výchozí adresář pod kořenem projektu", {
  withr::local_envvar(MEDIPARSE_PARQUET = NA, PIXI_PROJECT_ROOT = "/koren")

  expect_equal(eda_parquet_dir(), "/koren/resources/mimic/parquet")
})

test_that("eda_parquet_dir s prázdnou proměnnou použije výchozí adresář", {
  withr::local_envvar(MEDIPARSE_PARQUET = "", PIXI_PROJECT_ROOT = "/koren")
  expect_equal(eda_parquet_dir(), "/koren/resources/mimic/parquet")
})

test_that("eda_connect čte blokující proměnnou a důvod ze souboru", {
  root <- withr::local_tempdir()
  path <- file.path(root, "src", "mediparse", "domain", "data_access.json")
  dir.create(dirname(path), recursive = TRUE)
  writeLines('{"blocking":[{"name":"MEDIPARSE_TEST_BLOCK","reason":"testovací důvod"}]}', path)
  dir <- file.path(root, "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(PIXI_PROJECT_ROOT = root, MEDIPARSE_PARQUET = dir,
                    MEDIPARSE_TEST_BLOCK = "1", CI = NA, CLAUDECODE = NA)
  expect_equal(data_access_rules()[[1]]$name, "MEDIPARSE_TEST_BLOCK")
  expect_error(eda_connect(), "testovací důvod")
})

test_that("data_access_rules odmítne chybějící soubor", {
  path <- file.path(withr::local_tempdir(), "rules.json")
  expect_error(data_access_rules(path), "data_access.json")
})

test_that("data_access_rules splňuje sdílené případy Pythonu a R", {
  cases <- jsonlite::read_json(project_path("tests", "fixtures", "data_access_rules_cases.json"))
  path <- file.path(withr::local_tempdir(), "rules.json")
  for (case in cases) {
    writeLines(case$text, path)
    if (case$valid) {
      expect_no_error(data_access_rules(path))
    } else {
      expect_error(data_access_rules(path), "data_access.json", info = case$name)
    }
  }
})

test_that("eda_connect hlásí důvod první blokující proměnné ze souboru", {
  root <- withr::local_tempdir()
  path <- file.path(root, "src", "mediparse", "domain", "data_access.json")
  dir.create(dirname(path), recursive = TRUE)
  writeLines('{"blocking":[{"name":"MEDIPARSE_TEST_FIRST","reason":"první zákaz"},{"name":"MEDIPARSE_TEST_SECOND","reason":"druhý zákaz"}]}', path)
  dir <- file.path(root, "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(PIXI_PROJECT_ROOT = root, MEDIPARSE_PARQUET = dir,
                    MEDIPARSE_TEST_FIRST = "1", MEDIPARSE_TEST_SECOND = "1",
                    CI = NA, CLAUDECODE = NA)
  expect_error(eda_connect(), "první zákaz")
})

test_that("eda_connect odmítne i syntetiku s chybějícími pravidly", {
  root <- withr::local_tempdir()
  dir <- file.path(root, "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(PIXI_PROJECT_ROOT = root, MEDIPARSE_PARQUET = dir)
  expect_error(eda_connect(), "data_access.json")
})

test_that("synthetic.json splňuje sdílené případy Pythonu a R", {
  cases <- jsonlite::read_json(project_path("tests", "fixtures", "synthetic_marker_cases.json"))
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = NA, CI = NA)
  for (case in cases) {
    writeLines(case$text, file.path(dir, "synthetic.json"))
    if (case$valid) {
      con <- eda_connect()
      expect_true("discharge" %in% DBI::dbListTables(con), info = case$name)
      DBI::dbDisconnect(con, shutdown = TRUE)
    } else {
      expect_error(eda_connect(), "Neplatná značka", info = case$name)
    }
  }
})
