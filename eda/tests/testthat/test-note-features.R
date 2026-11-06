test_that("fixtury Pythonu splňují nezávislé kontrakty EDA včetně seznamů tokenů", {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))
  directory <- withr::local_tempdir()
  file.copy(list.files(CONTRACT_FIXTURES, pattern = "^note_.*[.]parquet$", full.names = TRUE), directory)
  writeLines("{}", file.path(directory, "cache.json"))
  withr::local_envvar(MEDIPARSE_EDA_CACHE = directory)
  for (item in list(
    list("note_sections.parquet", SECTION_COLUMNS),
    list("note_preprocessing.parquet", PREPROCESSING_COLUMNS),
    list("note_acronyms.parquet", ACRONYM_COLUMNS),
    list("note_variants.parquet", VARIANT_COLUMNS)
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
  directory <- withr::local_tempdir()
  writeLines("{}", file.path(directory, "cache.json"))
  Sys.setenv(MEDIPARSE_EDA_CACHE = directory)
  expect_error(feature_source(con, "note_sections.parquet", SECTION_COLUMNS), "neexistuje")
})

test_that("opakované čtení sekcí ponechá materializovanou tabulku", {
  con <- corpus_con("Allergies:\nnone\n")
  fixture_segments(con)
  DBI::dbExecute(con, sprintf("DELETE FROM %s", SECTIONS_TABLE))
  expect_identical(segment_corpus(con), SECTIONS_TABLE)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", SECTIONS_TABLE))$n, 0)
})

test_that("segment_corpus odmítne nesprávný typ z publikované cache", {
  con <- mimic_con()
  directory <- withr::local_tempdir()
  withr::local_envvar(MEDIPARSE_EDA_CACHE = directory)
  path <- file.path(directory, "note_sections.parquet")
  fixture <- file.path(CONTRACT_FIXTURES, "note_sections.parquet")
  DBI::dbExecute(con, sprintf("
    COPY (
      SELECT note_id, position::BIGINT AS position, section, span_chars, body_chars,
        body_masked, body_start, body_end FROM read_parquet(%s)
    ) TO %s (FORMAT parquet)", DBI::dbQuoteString(con, fixture), DBI::dbQuoteString(con, path)
  ))
  writeLines("{}", file.path(directory, "cache.json"))

  expect_error(segment_corpus(con), "sloupec position")
})

test_that("čtení cache odmítne hotový parquet bez publikačního markeru", {
  con <- mimic_con()
  directory <- withr::local_tempdir()
  withr::local_envvar(MEDIPARSE_EDA_CACHE = directory)
  file.copy(file.path(CONTRACT_FIXTURES, "note_sections.parquet"), directory)

  expect_error(segment_corpus(con), "cache není publikovaná; spusťte mediparse-eda-note-features", fixed = TRUE)
})

test_that("fixtura cache obsahuje jen zprávy discharge a pro prázdnou tabulku je prázdná", {
  con <- mimic_con()
  directory <- use_feature_fixtures(con)
  for (file in c("note_sections.parquet", "note_preprocessing.parquet", "note_acronyms.parquet", "note_variants.parquet")) {
    count <- DBI::dbGetQuery(con, sprintf(
      "SELECT count(*) AS n FROM read_parquet(%s)", DBI::dbQuoteString(con, file.path(directory, file))
    ))$n
    expect_equal(count, 0)
  }
  expect_identical(corpus_acronyms(con), character())
})


test_that("varianty odmítnou chybějící sloupec, jiný typ a NULL", {
  con <- mimic_con()
  directory <- withr::local_tempdir()
  withr::local_envvar(MEDIPARSE_EDA_CACHE = directory)
  path <- file.path(directory, "note_variants.parquet")
  fixture <- file.path(CONTRACT_FIXTURES, "note_variants.parquet")
  writeLines("{}", file.path(directory, "cache.json"))
  for (columns in c(
    "note_id, variant, occurrences",
    "note_id, variant, diagnosis, occurrences::BIGINT AS occurrences",
    "note_id, variant, NULL::VARCHAR AS diagnosis, occurrences"
  )) {
    if (file.exists(path)) unlink(path)
    DBI::dbExecute(con, sprintf(
      "COPY (SELECT %s FROM read_parquet(%s)) TO %s (FORMAT parquet)",
      columns, DBI::dbQuoteString(con, fixture), DBI::dbQuoteString(con, path)
    ))
    expect_error(note_variants(con), "sloupec (diagnosis|occurrences)")
  }
})
