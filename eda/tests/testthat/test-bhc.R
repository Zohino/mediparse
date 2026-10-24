test_that("mimic_con zná tabulku mimic-iv-bhc, i prázdnou", {
  empty <- mimic_con()
  expect_true("mimic-iv-bhc" %in% DBI::dbListTables(empty))
  expect_equal(DBI::dbGetQuery(empty, 'SELECT count(*) AS n FROM "mimic-iv-bhc"')$n, 0)

  rows <- data.frame(note_id = "n1", target = "text", stringsAsFactors = FALSE)
  filled <- mimic_con(`mimic-iv-bhc` = rows)
  expect_equal(DBI::dbGetQuery(filled, 'SELECT target FROM "mimic-iv-bhc"')$target, "text")
})
