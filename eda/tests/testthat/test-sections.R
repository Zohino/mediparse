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

found_headers <- function(texts) {
  con <- corpus_con(texts)
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
  con <- corpus_con(c("Chief Complaint:", "CHIEF  COMPLAINT :", "Chief Complaint.:"))
  table <- DBI::dbQuoteIdentifier(con, note_headers(con))

  out <- DBI::dbGetQuery(con, sprintf("SELECT DISTINCT norm FROM %s", table))

  expect_equal(out$norm, "chief complaint")
})

test_that("note_headers vrací tutéž tabulku podruhé a jeden řádek na výskyt", {
  con <- corpus_con("Service: A\nService: B")

  first <- note_headers(con)

  expect_equal(note_headers(con), first)
  expect_equal(first, "note_headers_input")
  expect_equal(DBI::dbGetQuery(con, "SELECT count(*) AS n FROM note_headers_input")$n, 2)
})

test_that("publishable_floor je 1 % korpusu a nejméně DUA_THRESHOLD", {
  expect_equal(publishable_floor(c(0, 500, 100000)), c(10, 10, 1000))
})

test_that("header_ranking zobrazí hlavičku jednou za zprávu a vrátí nejčastější zápis", {
  texts <- c(rep("Service: A\nService: B", 4), rep("SERVICE: A", 4), rep("Service: A", 2))
  con <- corpus_con(texts)

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, "service")
  expect_equal(out$n, 10)
  expect_equal(out$celkem, 10)
  expect_equal(out$hlavicka, "Service")
})

test_that("při shodě zápisu rozhoduje abecední pořadí", {
  con <- corpus_con(c(rep("SERVICE: A", 6), rep("Service: A", 6)))

  expect_equal(header_ranking(con, LEXICON)$hlavicka, "SERVICE")
})

test_that("header_ranking skryje hlavičku pod podlahou, i když má aspoň deset zpráv", {
  texts <- c(
    "Allergies: none\nPatient states that: x",
    rep("Allergies: none", 14),
    rep("Allergies: none\nPatient states that: x", 11),
    rep("Foo: x", 1474)
  )
  con <- corpus_con(texts)

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, c("foo", "allergies"))
  expect_equal(out$n, c(1474, 26))
  expect_equal(publishable_floor(out$celkem[[1]]), 15)
})

test_that("header_ranking zařadí hlavičky do sekce, preamble a mimo lexikon", {
  con <- corpus_con(rep("Presenting Complaint: a\nService: b\nFoo: c", 10))

  out <- header_ranking(con, LEXICON)

  expect_equal(out$norm, c("foo", "presenting complaint", "service"))
  expect_equal(out$zarazeni, c("mimo lexikon", "Hlavní obtíže", "preamble"))
})

test_that("header_ranking vrátí nejvýš RANKING_SIZE řádků", {
  codes <- paste0(rep(LETTERS, each = 26), rep(LETTERS, 26))[seq_len(RANKING_SIZE + 5L)]
  con <- corpus_con(rep(paste(sprintf("Head %s: x", codes), collapse = "\n"), 12))

  expect_equal(nrow(header_ranking(con, LEXICON)), RANKING_SIZE)
})

test_that("header_ranking řadí podle n sestupně a pak podle norm", {
  con <- corpus_con(c(rep("Beta: x\nAlpha: x", 11), rep("Gamma: x", 12)))

  expect_equal(header_ranking(con, LEXICON)$norm, c("gamma", "alpha", "beta"))
})

test_that("header_totals spočítá zprávy bez hlavičky, se sekcí a unikátní hlavičky", {
  con <- corpus_con(c("Allergies: none\nFoo Bar: x", "Foo Bar: y", "no header here"))

  out <- header_totals(con, LEXICON)

  expect_equal(out, list(celkem = 3, unikatni = 2, s_sekci = 1, bez_hlavicky = 1))
})

test_that("prázdný vstupní korpus vrátí prázdné výsledky bez chyby", {
  con <- mimic_con()

  expect_equal(nrow(header_ranking(con, LEXICON)), 0L)
  expect_equal(header_totals(con, LEXICON), list(celkem = 0, unikatni = 0, s_sekci = 0, bez_hlavicky = 0))
})

test_that("sekce s více synonymy se počítá jednou na zprávu a opakování je zpráva s aspoň dvěma výskyty", {
  con <- corpus_con(c(
    "Chief Complaint: a\nPresenting Complaint: b", "Presenting Complaint: a",
    "Allergies: a\nAllergies: b", "Allergies: a", "nothing"
  ))

  out <- section_coverage(con, LEXICON)

  expect_equal(out$sekce, c("allergies", "chief_complaint"))
  expect_equal(out$label_cs, c("Alergie", "Hlavní obtíže"))
  expect_equal(out$n, c(2, 2))
  expect_equal(out$opakovani, c(1, 1))
  expect_equal(out$celkem, c(5, 5))
})

test_that("synonym_coverage nechá mrtvé synonymum s nulou a počítá dokumentovou frekvenci", {
  con <- corpus_con(c("Chief Complaint: a\nChief Complaint: b", "CHIEF COMPLAINT: a", "Allergies: x"))

  out <- synonym_coverage(con, LEXICON)

  expect_equal(out$synonymum, c("Allergies", "Chief Complaint", "Presenting Complaint"))
  expect_equal(out$sekce, c("allergies", "chief_complaint", "chief_complaint"))
  expect_equal(out$n, c(1, 2, 0))
  expect_equal(out$opakovani, c(0, 1, 0))
  expect_equal(out$celkem, c(3, 3, 3))
})

test_that("pokrytí sekcí a synonym nad prázdným korpusem má nuly a všechny řádky lexikonu", {
  con <- mimic_con()

  sections <- section_coverage(con, LEXICON)
  synonyms <- synonym_coverage(con, LEXICON)

  expect_equal(sections$n, c(0, 0))
  expect_equal(sections$celkem, c(0, 0))
  expect_equal(synonyms$n, c(0, 0, 0))
  expect_equal(synonyms$opakovani, c(0, 0, 0))
})

ranking_fixture <- function() {
  data.frame(
    norm = c("chief complaint", "service", "imaging", "exam"),
    hlavicka = c("Chief Complaint", "Service", "IMAGING", "Exam"),
    n = c(1000, 990, 600, 200),
    celkem = 1000,
    zarazeni = c("Hlavní obtíže", PREAMBLE_CLASS, OUTSIDE_CLASS, OUTSIDE_CLASS),
    stringsAsFactors = FALSE
  )
}

totals_fixture <- function(celkem = 1000) {
  list(celkem = celkem, unikatni = 123, s_sekci = 900, bez_hlavicky = 50)
}

test_that("header_table uvede hlavičku, zařazení, počet a podíl", {
  out <- header_table(ranking_fixture())

  expect_equal(out$hlavicka, c("Chief Complaint", "Service", "IMAGING", "Exam"))
  expect_equal(out$zarazeni, c("Hlavní obtíže", "preamble", "mimo lexikon", "mimo lexikon"))
  expect_equal(out$n, c("1 000", "990", "600", "200"))
  expect_equal(out$podil, c("100,0", "99,0", "60,0", "20,0"))
})

test_that("ranking_bars obarví podle druhu zařazení a seřadí nejčastější nahoru", {
  out <- ranking_bars(ranking_fixture())

  expect_equal(as.character(out$druh), c("sekce lexikonu", "preamble", "mimo lexikon", "mimo lexikon"))
  expect_equal(levels(out$hlavicka), rev(c("Chief Complaint", "Service", "IMAGING", "Exam")))
  expect_equal(out$podil, c(1, 0.99, 0.6, 0.2))
})

test_that("section_table a synonym_table ukážou pokrytí a opakování v procentech", {
  sections <- data.frame(
    sekce = "allergies", label_cs = "Alergie", n = 995, opakovani = 100, celkem = 1000,
    stringsAsFactors = FALSE
  )
  synonyms <- data.frame(
    sekce = c("allergies", "allergies"), synonymum = c("Allergies", "Allergy"),
    n = c(995, 0), opakovani = c(100, 0), celkem = 1000, stringsAsFactors = FALSE
  )

  expect_equal(section_table(sections)$pokryti, "99,5")
  expect_equal(section_table(sections)$opakovani, "10,0")
  expect_equal(synonym_table(synonyms)$pokryti, c("99,5", "0,0"))
  expect_equal(synonym_table(synonyms)$synonymum, c("Allergies", "Allergy"))
})

test_that("tabulky nad prázdným korpusem mají pomlčky nebo nula řádků", {
  empty_synonyms <- data.frame(
    sekce = "allergies", synonymum = "Allergies", n = 0, opakovani = 0, celkem = 0,
    stringsAsFactors = FALSE
  )

  expect_equal(nrow(header_table(ranking_fixture()[0, ])), 0L)
  expect_equal(synonym_table(empty_synonyms)$pokryti, "–")
})

test_that("candidates_text vyjmenuje hlavičky mimo lexikon s pokrytím aspoň CANDIDATE_COVERAGE", {
  text <- candidates_text(ranking_fixture(), totals_fixture())

  expect_match(text, "`IMAGING`", fixed = TRUE)
  expect_no_match(text, "Exam", fixed = TRUE)
  expect_match(text, "60,0 %", fixed = TRUE)
  expect_match(candidates_text(ranking_fixture()[1:2, ], totals_fixture()), "žádný", fixed = TRUE)
  expect_match(candidates_text(ranking_fixture()[0, ], totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

synonyms_fixture <- function(celkem = 1000) {
  data.frame(
    sekce = c("a", "a", "b", "c"), synonymum = c("Alpha", "Beta", "Gamma", "Delta"),
    n = c(900, 0, 500, 0), opakovani = c(49, 0, 50, 0), celkem = celkem,
    stringsAsFactors = FALSE
  )
}

test_that("dead_synonyms_text vyjmenuje synonyma s nulou a skloňuje jejich počet", {
  expect_match(dead_synonyms_text(synonyms_fixture(), totals_fixture()), "2 synonyma", fixed = TRUE)
  expect_match(dead_synonyms_text(synonyms_fixture(), totals_fixture()), "`Beta`", fixed = TRUE)
  expect_match(dead_synonyms_text(synonyms_fixture()[c(1, 3), ], totals_fixture()), "žádné", fixed = TRUE)
  expect_match(dead_synonyms_text(synonyms_fixture(0), totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

test_that("repeated_synonyms_text bere opakování od REPEAT_SHARE včetně hranice", {
  text <- repeated_synonyms_text(synonyms_fixture(), totals_fixture())

  expect_match(text, "`Gamma`", fixed = TRUE)
  expect_no_match(text, "Alpha", fixed = TRUE)
  expect_match(text, "1 synonymum", fixed = TRUE)
  expect_match(repeated_synonyms_text(synonyms_fixture()[1, ], totals_fixture()), "Žádné", fixed = TRUE)
  expect_match(repeated_synonyms_text(synonyms_fixture(0), totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

test_that("coverage_text uvede pokrytí sekcí, zprávy bez hlavičky a počet unikátních hlaviček", {
  text <- coverage_text(totals_fixture())

  expect_match(text, "90,0 %", fixed = TRUE)
  expect_match(text, "5,0 %", fixed = TRUE)
  expect_match(text, "123 unikátních hlaviček", fixed = TRUE)
  expect_match(coverage_text(totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})
