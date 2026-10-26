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

test_that("eda_connect odmítne neplatnou značku synthetic.json pod CLAUDECODE", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir, marker = FALSE)
  writeLines("{}", file.path(dir, "synthetic.json"))
  withr::local_envvar(MEDIPARSE_PARQUET = dir, CLAUDECODE = "1", CI = NA)

  expect_error(eda_connect(), "synthetic.json")
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

test_that("eda_connect založí pohledy podle nastavení z Pythonu", {
  dir <- file.path(withr::local_tempdir(), "parquet")
  write_parquet_tables(dir)
  withr::local_envvar(MEDIPARSE_PARQUET = dir)
  fake <- list(connection_settings = function(environ) {
    list(
      memory_limit = "1GiB",
      temp_directory = file.path(dirname(dir), "duckdb-tmp"),
      views = list(jiny = file.path(dir, "discharge.parquet"))
    )
  })
  con <- eda_connect(fake)
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_equal(DBI::dbListTables(con), "jiny")
  expect_equal(setting(con, "memory_limit"), "1.0 GiB")
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

length_rows <- function(chunk) {
  data.frame(note_id = chunk$note_id, n = nchar(chunk$text), stringsAsFactors = FALSE)
}

LENGTH_COLUMNS <- c(note_id = "VARCHAR", n = "INTEGER")
NOTES_SQL <- "SELECT note_id, text FROM discharge"

test_that("batched_table zapíše výsledky přes víc dávek v pořadí note_id", {
  notes <- text_notes(c("c", "aa", "bbb", "dddd", "e"))
  notes$note_id <- c("n3", "n1", "n2", "n5", "n4")
  con <- mimic_con(discharge = notes)
  sizes <- integer()
  fn <- function(chunk) {
    sizes <<- c(sizes, nrow(chunk))
    length_rows(chunk)
  }
  name <- batched_table(con, "lengths", LENGTH_COLUMNS, NOTES_SQL, fn, batch = 2L)
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", name))

  expect_equal(name, "lengths")
  expect_equal(sizes, c(2L, 2L, 1L))
  expect_equal(out$note_id, paste0("n", 1:5))
  expect_equal(out$n, c(2, 3, 1, 1, 4))
  expect_false("lengths_rows" %in% DBI::dbListTables(con))
})

test_that("batched_table dá pro prázdný zdroj typovanou prázdnou tabulku", {
  con <- mimic_con()
  name <- batched_table(con, "lengths", LENGTH_COLUMNS, NOTES_SQL, function(chunk) stop("nemá se volat"))
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", name))

  expect_equal(names(out), c("note_id", "n"))
  expect_equal(nrow(out), 0)
  expect_type(out$n, "integer")
})

test_that("batched_table při chybě smaže výslednou i pomocnou tabulku", {
  con <- mimic_con(discharge = text_notes(c("a", "b", "c")))

  expect_error(
    batched_table(con, "lengths", LENGTH_COLUMNS, NOTES_SQL, function(chunk) stop("selhání"), batch = 2L),
    "selhání"
  )
  expect_false(any(grepl("^lengths", DBI::dbListTables(con))))
})

test_that("batched_table vrátí existující tabulku, aniž by volal fn", {
  con <- mimic_con(discharge = text_notes(c("a", "b")))
  batched_table(con, "lengths", LENGTH_COLUMNS, NOTES_SQL, length_rows)
  name <- batched_table(con, "lengths", LENGTH_COLUMNS, NOTES_SQL, function(chunk) stop("nemá se volat"))

  expect_equal(name, "lengths")
})
