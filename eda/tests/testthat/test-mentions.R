test_that("read_candidates načte kapitolu, fráze a zkratky", {
  pool <- read_candidates(candidates_path())

  expect_equal(pool$diabetes$kapitola, "E")
  expect_equal(pool$diabetes$acronyms_en, c("DM", "DM2"))
  expect_equal(pool$pneumonia$acronyms_en, character())
  expect_true("afib" %in% pool$atrial_fibrillation$keywords_en)
})

test_that("každý kandidát má klíčová slova a kapitolu podle první předpony ICD-10", {
  pool <- read_candidates(candidates_path())

  for (entry in pool) {
    expect_true(length(entry$keywords_en) > 0L)
    expect_equal(entry$kapitola, substr(entry$icd10_prefixes[[1]], 1L, 1L))
  }
})

test_that("zkratky jsou velká písmena a číslice a fráze neobsahují zkratku velkými písmeny", {
  pool <- read_candidates(candidates_path())

  for (entry in pool) {
    expect_true(all(grepl("^[A-Z0-9]+$", entry$acronyms_en)))
    expect_equal(entry$keywords_en, tolower(entry$keywords_en))
  }
})

test_that("read_text_mentions načte negační výrazy, okno a lexikon příznaků", {
  mentions <- read_text_mentions(text_mentions_path())

  expect_length(mentions$negation$cues, 11L)
  expect_equal(mentions$negation$window_chars, 40L)
  expect_length(mentions$symptoms$keywords_en, 20L)
  expect_true("chest pain" %in% mentions$symptoms$keywords_en)
})
