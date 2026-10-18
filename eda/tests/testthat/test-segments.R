LEXICON_PATH <- function() section_headers_path()
TEXT_WITH_SECTIONS <- "Name: ___\n\nAllergies:\nnone\n\nChief Complaint:\npain\n"

test_that("section_rows přijme dvě shodné zprávy pod různými note_id", {
  module <- sections_module()
  chunk <- list(
    note_id = c(k = "n1", k = "n2"),
    text = c(k = TEXT_WITH_SECTIONS, k = TEXT_WITH_SECTIONS)
  )
  out <- section_rows(module, LEXICON_PATH(), chunk)

  expect_equal(unique(out$note_id), c("n1", "n2"))
  expect_equal(nrow(out), 6L)
  expect_equal(out$section[1:3], c(module$OTHER, "allergies", "chief_complaint"))
})

test_that("section_rows zvládne dávku o jedné zprávě", {
  chunk <- list(note_id = "jediny", text = "Allergies:\nnone\n")
  out <- section_rows(sections_module(), LEXICON_PATH(), chunk)

  expect_equal(out$note_id, "jediny")
  expect_equal(out$section, "allergies")
  expect_equal(out$position, 0L)
})

test_that("section_rows vrátí pro prázdnou dávku typovaný data.frame s pěti sloupci", {
  chunk <- list(note_id = character(), text = character())
  out <- section_rows(sections_module(), LEXICON_PATH(), chunk)

  expect_equal(names(out), names(SECTION_COLUMNS))
  expect_equal(nrow(out), 0L)
  expect_type(out$note_id, "character")
  expect_type(out$position, "integer")
  expect_type(out$span_chars, "integer")
  expect_type(out$body_chars, "integer")
})

test_that("segment_corpus naplní tabulku jen zprávami vstupního korpusu", {
  texts <- c("Allergies:\nnone\n", "Chief Complaint:\npain\n")
  notes <- text_notes(paste0(strrep("x", SHORT_LIMIT), "\n", texts))
  notes$hadm_id <- c("h1", "h2")
  icd <- data.frame(hadm_id = "h1", icd_code = "Z99", icd_version = "10", stringsAsFactors = FALSE)
  short <- text_notes("Allergies:\nnone\n", prefix = "krátká")
  short$hadm_id <- "h1"
  con <- mimic_con(discharge = rbind(notes, short), diagnoses_icd = icd)
  name <- segment_corpus(con, sections_module(), LEXICON_PATH())
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s ORDER BY note_id, position", name))

  expect_equal(name, SECTIONS_TABLE)
  expect_equal(unique(out$note_id), "n1")
  expect_equal(out$section, c("other", "allergies"))
  expect_equal(sum(out$span_chars), nchar(notes$text[[1]]))
})

test_that("segment_corpus nad prázdným korpusem vytvoří prázdnou typovanou tabulku", {
  con <- mimic_con()
  name <- segment_corpus(con, sections_module(), LEXICON_PATH())
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", name))

  expect_equal(names(out), names(SECTION_COLUMNS))
  expect_equal(nrow(out), 0L)
})
