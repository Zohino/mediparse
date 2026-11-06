spelling_con <- function(tokens = character(), df = numeric(), env = parent.frame()) {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE), envir = env)
  DBI::dbWriteTable(con, "token_frequency", data.frame(token = tokens, df = df))
  con
}

manual_spelling_con <- function(env = parent.frame()) {
  spelling_con(
    c("pneumonia", "pnuemonia", "pnemonia", "pneumonias", "diabetes", "diabtes", "atrial", "trial", "fibrillation", "fibrilation", "zzzzz", "dm2", "abcd"),
    c(100000, 300, 9, 5000, 80000, 120, 90000, 40000, 60000, 1, 50, 3, 2), env
  )
}

MANUAL_TARGETS <- c("pneumonia", "diabetes", "atrial", "fibrillation")

test_that("blízké tvary dodrží ruční vzdálenost, poměr DF a práh popisků", {
  con <- manual_spelling_con()
  forms <- near_miss_forms(con, MANUAL_TARGETS)
  expect_identical(forms$tvar, c("pneumonias", "pnuemonia", "diabtes"))
  expect_equal(forms$df, c(5000, 300, 120))
  expect_identical(forms$cil, c("pneumonia", "pneumonia", "diabetes"))
  expect_identical(near_miss_table(forms)$zpravy, c("5\u00a0000", "300", "120"))
  expect_match(near_miss_text(forms), "3\u00a0tvary", fixed = TRUE)
  DBI::dbExecute(con, "UPDATE token_frequency SET df = 10 WHERE token = 'pnemonia'")
  expect_true("pnemonia" %in% near_miss_forms(con, MANUAL_TARGETS)$tvar)
})

test_that("odhad překlepů odpovídá ručně spočtenému čitateli a jmenovateli", {
  estimate <- typo_estimate(manual_spelling_con())
  expect_equal(estimate$slovnik, 11)
  expect_equal(estimate$kandidati, 4)
  expect_equal(estimate$vyskyty_kandidatu, 430)
  expect_equal(estimate$vyskyty, 375480)
  table <- typo_table(estimate)
  expect_identical(table$podil_slovniku, "36,4")
  expect_identical(table$podil_vyskytu, "0,1")
  text <- typo_text(estimate)
  expect_match(text, "popisný odhad podle definice", fixed = TRUE)
  expect_match(text, "legitimní slova", fixed = TRUE)
  expect_match(text, "vynechává překlepy", fixed = TRUE)
})

test_that("token se dvěma častými sousedy je jediný kandidát", {
  estimate <- typo_estimate(spelling_con(c("bbbbb", "bbbby", "bbbbx"), c(500, 500, 1)))
  expect_equal(estimate$kandidati, 1)
  expect_equal(estimate$slovnik, 3)
  expect_equal(estimate$vyskyty_kandidatu, 1)
})

test_that("odhad zahrne přesný poměr sto a vyřadí menší poměr", {
  con <- spelling_con(c("aaaaa", "aaaab"), c(1000, 10))
  expect_equal(typo_estimate(con)$kandidati, 1)
  DBI::dbExecute(con, "UPDATE token_frequency SET df = 11 WHERE token = 'aaaab'")
  expect_equal(typo_estimate(con)$kandidati, 0)
})

test_that("poměr DF musí dosáhnout sta i u jediného výskytu", {
  expect_equal(typo_estimate(spelling_con(c("aaaaa", "aaaab"), c(99, 1)))$kandidati, 0)
  expect_equal(typo_estimate(spelling_con(c("aaaaa", "aaaab"), c(100, 1)))$kandidati, 1)
})

test_that("prázdné souhrny vrátí nerozhodnutelný text a prázdné tabulky", {
  con <- spelling_con()
  forms <- near_miss_forms(con, MANUAL_TARGETS)
  expect_match(near_miss_text(forms), UNDECIDABLE_TEXT, fixed = TRUE)
  expect_match(typo_text(typo_estimate(con)), UNDECIDABLE_TEXT, fixed = TRUE)
  expect_match(variant_text(data.frame(vyskyty = numeric())), UNDECIDABLE_TEXT, fixed = TRUE)
  expect_equal(nrow(near_miss_table(forms)), 0L)
  expect_equal(nrow(near_miss_forms(con, character())), 0L)
})

test_that("cíle jsou jedinečná dost dlouhá slova jen vybraných diagnóz", {
  candidates <- list(
    first = list(keywords_en = c("chronic pulmonary disease", "pulmonary")),
    second = list(keywords_en = c("chronic kidney disease", "end-stage renal disease")),
    ignored = list(keywords_en = "pulmonary extraneous")
  )
  expect_identical(spelling_targets(candidates, c("first", "second")), c("kidney", "pulmonary"))
  expect_identical(spelling_targets(candidates, character()), character())
})

test_that("DF počítá zprávy s tokenem a materializuje se pouze jednou", {
  con <- corpus_con(c("alpha AF", "alpha gamma"))
  fixture_preprocessing(con)
  expect_identical(token_frequency(con), "token_frequency")
  freq <- DBI::dbGetQuery(con, "SELECT * FROM token_frequency WHERE token IN ('alpha', 'gamma') ORDER BY token")
  expect_identical(freq$token, c("alpha", "gamma"))
  expect_equal(as.numeric(freq$df), c(2, 1))
  DBI::dbExecute(con, "DELETE FROM token_frequency")
  token_frequency(con)
  expect_equal(DBI::dbGetQuery(con, "SELECT count(*) AS n FROM token_frequency")$n, 0)
})

test_that("varianty z konfigurace zveřejní sedm výskytů a tři z pěti zpráv bez kódu", {
  con <- corpus_con(rep("Pt stable", 100), codes = c("I48", "I48", "Z99", "Z99", "Z99", "I48", rep("Z99", 94)))
  use_feature_fixtures(con)
  note_variants(con)
  # Zkonstruované dlouhé řádky drží počty výskytů nezávislé na počtu zpráv s kódem.
  DBI::dbExecute(con, sprintf("DELETE FROM %s", VARIANTS_TABLE))
  ids <- DBI::dbGetQuery(con, "SELECT note_id FROM discharge ORDER BY CAST(substr(note_id, 2) AS INTEGER)")$note_id
  DBI::dbAppendTable(con, VARIANTS_TABLE, data.frame(
    note_id = ids, variant = "afib", diagnosis = "atrial_fibrillation",
    occurrences = c(3L, 1L, 1L, 1L, 1L, rep(0L, 95))
  ))
  summary <- variant_summary(con)
  expect_equal(summary$vyskyty, 7)
  expect_equal(summary$podil, 0.6)
  expect_identical(summary$diagnosis, "atrial_fibrillation")
  expect_identical(summary$label_cs, "Fibrilace síní")
  expect_identical(variant_table(summary)$bez_kodu, "60,0")
  expect_identical(variant_table(summary)$vyskyty, "7")
  expect_match(variant_text(summary), "1\u00a0varianta", fixed = TRUE)
})

test_that("nulový výskyt varianty nemá jmenovatel a prázdný korpus nespadne", {
  con <- corpus_con("Pt stable")
  use_feature_fixtures(con)
  summary <- variant_summary(con)
  expect_true(all(summary$vyskyty == 0))
  expect_true(all(is.na(summary$podil)))
  expect_true(all(variant_table(summary)$bez_kodu == "–"))
  expect_match(variant_text(summary), UNDECIDABLE_TEXT, fixed = TRUE)
  empty <- mimic_con()
  use_feature_fixtures(empty)
  expect_equal(nrow(variant_summary(empty)), 0L)
  expect_equal(nrow(variant_table(variant_summary(empty))), 0L)
})
