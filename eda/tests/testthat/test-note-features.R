test_that("fixtury Pythonu splňují nezávislé kontrakty EDA včetně seznamů tokenů", {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))
  withr::local_envvar(MEDIPARSE_EDA_CACHE = CONTRACT_FIXTURES)
  for (item in list(
    list("note_sections.parquet", SECTION_COLUMNS),
    list("note_preprocessing.parquet", PREPROCESSING_COLUMNS),
    list("note_acronyms.parquet", ACRONYM_COLUMNS)
  )) {
    expect_type(feature_source(con, item[[1]], item[[2]]), "character")
  }
  sections <- read_contract_table(file.path(CONTRACT_FIXTURES, "note_sections.parquet"), SECTION_COLUMNS)
  expect_true(OTHER_SECTION %in% sections$section)
  expect_identical(OTHER_SECTION, "other")
  expect_true(any(sections$body_masked))
  expect_identical(names(sections), names(SECTION_COLUMNS))
  expect_type(sections$body_start, "integer")
  expect_type(sections$body_masked, "logical")
  expect_identical(corpus_acronyms(con), c("AF", "CKD", "COPD", "DM", "DM2", "ESRD"))
})

test_that("čtení cache vyžaduje umístění a existující soubor", {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))
  withr::local_envvar(MEDIPARSE_EDA_CACHE = "")
  expect_error(feature_source(con, "note_sections.parquet", SECTION_COLUMNS), "Chybí MEDIPARSE_EDA_CACHE")
  Sys.setenv(MEDIPARSE_EDA_CACHE = withr::local_tempdir())
  expect_error(feature_source(con, "note_sections.parquet", SECTION_COLUMNS), "neexistuje")
})

test_that("opakované čtení sekcí ponechá materializovanou tabulku", {
  con <- corpus_con("Allergies:\nnone\n")
  fixture_segments(con)
  DBI::dbExecute(con, sprintf("DELETE FROM %s", SECTIONS_TABLE))
  expect_identical(segment_corpus(con), SECTIONS_TABLE)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", SECTIONS_TABLE))$n, 0)
})
