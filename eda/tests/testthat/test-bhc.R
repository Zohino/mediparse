test_that("mimic_con zná tabulku mimic-iv-bhc, i prázdnou", {
  empty <- mimic_con()
  expect_true("mimic-iv-bhc" %in% DBI::dbListTables(empty))
  expect_equal(DBI::dbGetQuery(empty, 'SELECT count(*) AS n FROM "mimic-iv-bhc"')$n, 0)

  rows <- data.frame(note_id = "n1", target = "text", stringsAsFactors = FALSE)
  filled <- mimic_con(`mimic-iv-bhc` = rows)
  expect_equal(DBI::dbGetQuery(filled, 'SELECT target FROM "mimic-iv-bhc"')$target, "text")
})

# Očekávání jsou řetězce vypočtené skriptem clin-bhc-summ (utils/mimic_iv_bhc_preprocessing.py)
# v Pythonu, ne výstup SQL.
CLEANING_CASES <- list(
  list("a   b  c", "a b c"),
  list("a\t\tb", "a b"),
  list("a\r\nb\r\n\r\nc", "a b c"),
  list("a b  c", "a b c"),
  list("a b", "a b"),
  list("a\u000bb", "a b"),
  list("a　b", "a b"),
  list("Brief Hospital Course:\nPt  was ok.", "Pt was ok."),
  list("BRIEF HOSPITAL COURSE:\nPt ok", "Pt ok"),
  list("BRIEF HOSPITAL COURSE\nPt ok", " Pt ok"),
  list("Brief Hospital Course\nPt ok", " Pt ok"),
  list("Brief hospital course:\nPt ok", "Brief hospital course: Pt ok"),
  list("Brief Hospital Course: text here", ": text here"),
  list(
    "Brief Hospital Course:\nPt ok. See Brief Hospital Course above.",
    "Pt ok. See above."
  ),
  list("Brief Hospital Course:\nBrief Hospital\nCourse: stable", "Brief Hospital Course: stable"),
  list("Brief  Hospital Course x", " x")
)

clean_in_sql <- function(inputs) {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))
  DBI::dbWriteTable(con, "c", data.frame(i = seq_along(inputs), x = inputs, stringsAsFactors = FALSE))
  DBI::dbGetQuery(con, sprintf("SELECT %s AS y FROM c ORDER BY i", bhc_clean_sql("x")))$y
}

test_that("bhc_clean_sql dává totéž co skript clin-bhc-summ", {
  inputs <- vapply(CLEANING_CASES, `[[`, "", 1L)
  expected <- vapply(CLEANING_CASES, `[[`, "", 2L)

  expect_equal(clean_in_sql(inputs), expected)
})

test_that("bhc_clean_sql zachová NULL jako NULL", {
  expect_true(is.na(clean_in_sql(NA_character_)))
})
