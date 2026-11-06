FEATURE_FIXTURES <- "../../../tests/fixtures/note_features"
CONTRACT_FIXTURES <- "../../../tests/fixtures/contracts"

use_feature_fixtures <- function(con, env = parent.frame()) {
  directory <- withr::local_tempdir(.local_envir = env)
  withr::local_envvar(MEDIPARSE_EDA_CACHE = directory, .local_envir = env)
  lookup <- sprintf("read_parquet(%s)", DBI::dbQuoteString(con, file.path(FEATURE_FIXTURES, "notes.parquet")))
  missing <- DBI::dbGetQuery(con, sprintf(
    "SELECT count(*) AS n FROM discharge d WHERE NOT EXISTS (SELECT 1 FROM %s l WHERE l.text = d.text)", lookup
  ))$n
  if (missing > 0) stop("Zkonstruovaná zpráva chybí ve fixtuře měření; doplňte fixturu zapisovačem Pythonu.")
  for (file in c("note_sections.parquet", "note_preprocessing.parquet", "note_acronyms.parquet", "note_variants.parquet")) {
    DBI::dbExecute(con, sprintf("
      COPY (
        SELECT d.note_id, f.* EXCLUDE (note_id)
        FROM read_parquet(%s) f JOIN %s l USING (note_id)
        JOIN discharge d ON d.text = l.text
      ) TO %s (FORMAT parquet)",
      DBI::dbQuoteString(con, file.path(FEATURE_FIXTURES, file)), lookup,
      DBI::dbQuoteString(con, file.path(directory, file))
    ))
  }
  writeLines("{}", file.path(directory, "cache.json"))
  invisible(directory)
}

fixture_segments <- function(con, env = parent.frame()) {
  use_feature_fixtures(con, env)
  segment_corpus(con)
}

fixture_preprocessing <- function(con, env = parent.frame()) {
  use_feature_fixtures(con, env)
  note_preprocessing(con)
}
