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

LEXICON <- list(
  sections = data.frame(
    sekce = c("allergies", "chief_complaint", "chief_complaint"),
    label_cs = c("Alergie", "Hlavní obtíže", "Hlavní obtíže"),
    synonymum = c("Allergies", "Chief Complaint", "Presenting Complaint"),
    stringsAsFactors = FALSE
  ),
  preamble = "Service"
)

header_con <- function(texts, env = parent.frame()) {
  hadm <- paste0("h", seq_along(texts))
  padding <- paste0(strrep("x", SHORT_LIMIT), " ", hadm, "\n")
  notes <- text_notes(paste0(padding, texts))
  notes$hadm_id <- hadm
  rows <- data.frame(hadm_id = hadm, icd_code = "Z99", icd_version = "10", stringsAsFactors = FALSE)
  mimic_con(discharge = notes, diagnoses_icd = rows, env = env)
}

found_headers <- function(texts) {
  con <- header_con(texts)
  table <- DBI::dbQuoteIdentifier(con, note_headers(con))
  out <- DBI::dbGetQuery(con, sprintf("SELECT note_id, header, norm FROM %s ORDER BY note_id, header", table))
  split(out$header, factor(out$note_id, levels = paste0("n", seq_along(texts))))
}

test_that("note_headers najde hlavičku na začátku řádku včetně vloženého pole a odsazení", {
  found <- found_headers(c(
    "Service: MEDICINE",
    "  Chief Complaint:\n\tAllergies: none",
    "Allergies:\r\nPenicillin\r\nSocial History:\r\nnone",
    "text\nFamily History:"
  ))

  expect_equal(unname(found), list(
    "Service", c("Allergies", "Chief Complaint"), c("Allergies", "Social History"), "Family History"
  ))
})

test_that("note_headers nebere krátký řetězec, čas, značku ani text uprostřed řádku", {
  found <- found_headers(c("Ab: x", "Time 10:30", "# CHF: yes", "see Chief Complaint: x"))

  expect_equal(unname(found), list(character(), character(), character(), character()))
})

test_that("normalizace sloučí velikost písmen, mezery a koncové tečky", {
  con <- header_con(c("Chief Complaint:", "CHIEF  COMPLAINT :", "Chief Complaint.:"))
  table <- DBI::dbQuoteIdentifier(con, note_headers(con))

  out <- DBI::dbGetQuery(con, sprintf("SELECT DISTINCT norm FROM %s", table))

  expect_equal(out$norm, "chief complaint")
})

test_that("note_headers vrací tutéž tabulku podruhé a jeden řádek na výskyt", {
  con <- header_con("Service: A\nService: B")

  first <- note_headers(con)

  expect_equal(note_headers(con), first)
  expect_equal(first, "note_headers_input")
  expect_equal(DBI::dbGetQuery(con, "SELECT count(*) AS n FROM note_headers_input")$n, 2)
})

repeated <- function(text, times) rep(text, times)

test_that("publishable_floor je 1 % korpusu a nejméně DUA_THRESHOLD", {
  expect_equal(publishable_floor(c(0, 500, 100000)), c(10, 10, 1000))
})

test_that("header_ranking zobrazí hlavičku jednou za zprávu a vrátí nejčastější zápis", {
  texts <- c(repeated("Service: A\nService: B", 4), repeated("SERVICE: A", 4), repeated("Service: A", 2))
  con <- header_con(texts)

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, "service")
  expect_equal(out$n, 10)
  expect_equal(out$celkem, 10)
  expect_equal(out$hlavicka, "Service")
})

test_that("při shodě zápisu rozhoduje abecední pořadí", {
  con <- header_con(c(repeated("SERVICE: A", 6), repeated("Service: A", 6)))

  expect_equal(header_ranking(con, LEXICON)$hlavicka, "SERVICE")
})

test_that("header_ranking skryje hlavičku pod podlahou, i když má aspoň deset zpráv", {
  texts <- c(
    repeated("Allergies: none\nPatient states that: x", 1),
    repeated("Allergies: none", 14),
    repeated("Allergies: none\nPatient states that: x", 11),
    repeated("Foo: x", 1474)
  )
  con <- header_con(texts)

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, c("foo", "allergies"))
  expect_equal(out$n, c(1474, 26))
  expect_equal(publishable_floor(out$celkem[[1]]), 15)
})

test_that("header_ranking zařadí hlavičky do sekce, preamble a mimo lexikon", {
  con <- header_con(repeated("Presenting Complaint: a\nService: b\nFoo: c", 10))

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, c("foo", "presenting complaint", "service"))
  expect_equal(out$zarazeni, c("mimo lexikon", "Hlavní obtíže", "preamble"))
})

test_that("header_ranking vrátí nejvýš RANKING_SIZE řádků", {
  codes <- paste0(rep(LETTERS, each = 26), rep(LETTERS, 26))[seq_len(RANKING_SIZE + 5L)]
  con <- header_con(repeated(paste(sprintf("Head %s: x", codes), collapse = "\n"), 12))

  expect_equal(nrow(header_ranking(con, LEXICON)), RANKING_SIZE)
})

test_that("header_ranking řadí podle n sestupně a pak podle norm", {
  con <- header_con(c(repeated("Beta: x\nAlpha: x", 11), repeated("Gamma: x", 12)))

  expect_equal(header_ranking(con, LEXICON)$norm, c("gamma", "alpha", "beta"))
})

test_that("header_totals spočítá zprávy bez hlavičky, se sekcí a unikátní hlavičky", {
  con <- header_con(c("Allergies: none\nFoo Bar: x", "Foo Bar: y", "no header here"))

  out <- header_totals(con, LEXICON)

  expect_equal(out, list(celkem = 3, unikatni = 2, s_sekci = 1, bez_hlavicky = 1))
})

test_that("prázdný vstupní korpus vrátí prázdné výsledky bez chyby", {
  con <- mimic_con()

  expect_equal(nrow(header_ranking(con, LEXICON)), 0L)
  expect_equal(header_totals(con, LEXICON), list(celkem = 0, unikatni = 0, s_sekci = 0, bez_hlavicky = 0))
})
