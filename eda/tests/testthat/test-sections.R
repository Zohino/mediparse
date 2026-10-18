test_that("lexikon sekcí má 17 sekcí šablony MIMIC s popiskem a synonymem", {
  lexicon <- read_section_headers(section_headers_path())
  keys <- unique(lexicon$sections$sekce)

  expect_length(keys, 17L)
  expect_true(all(c("discharge_diagnosis", "discharge_condition") %in% keys))
  expect_false(anyNA(lexicon$sections$label_cs))
  expect_true(all(nzchar(lexicon$sections$label_cs)))
  expect_true(all(nzchar(lexicon$sections$synonymum)))
  expect_length(lexicon$preamble, 8L)
})

test_that("normalizované synonymum patří jediné sekci a není v preamble", {
  lexicon <- read_section_headers(section_headers_path())
  normalize <- function(x) sub("[ .]+$", "", gsub("\\s+", " ", tolower(x)))
  synonyms <- normalize(lexicon$sections$synonymum)

  expect_false(anyDuplicated(synonyms) > 0L)
  expect_length(intersect(synonyms, normalize(lexicon$preamble)), 0L)
})
