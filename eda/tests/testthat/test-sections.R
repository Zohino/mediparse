test_that("lexikon načte všechny sekce konfigurace, každou s popiskem a synonymem", {
  raw <- jsonlite::fromJSON(section_headers_path(), simplifyVector = FALSE)
  lexicon <- read_section_headers(section_headers_path())
  keys <- unique(lexicon$sections$sekce)

  expect_equal(keys, names(raw$sections))
  expect_true(all(c("discharge_diagnosis", "discharge_condition") %in% keys))
  expect_false(anyNA(lexicon$sections$label_cs))
  expect_true(all(nzchar(lexicon$sections$label_cs)))
  expect_true(all(nzchar(lexicon$sections$synonymum)))
  expect_equal(lexicon$preamble, as.character(unlist(raw$preamble)))
})

test_that("lexikon nemá dvě sekce ani sekci a preamble se stejným normalizovaným synonymem", {
  con <- mimic_con()

  expect_identical(header_lexicon(con, read_section_headers(section_headers_path())), LEXICON_TABLE)
})

test_that("read_section_headers odmítne sekci bez synonym nebo bez popisku", {
  write_config <- function(sections) {
    path <- withr::local_tempfile(fileext = ".json", .local_envir = parent.frame())
    jsonlite::write_json(list(sections = sections, preamble = list("Name")), path, auto_unbox = TRUE)
    path
  }

  expect_error(
    read_section_headers(write_config(list(a = list(label_cs = "A", synonyms = list())))),
    "a", fixed = TRUE
  )
  expect_error(
    read_section_headers(write_config(list(b = list(synonyms = list("B"))))),
    "b", fixed = TRUE
  )
})

test_that("header_lexicon odmítne kolizi normalizovaných synonym mezi sekcemi i s preamble", {
  con <- mimic_con()
  sections <- data.frame(
    sekce = c("a", "b"), label_cs = c("A", "B"), synonymum = c("Allergies", "ALLERGIES ."),
    stringsAsFactors = FALSE
  )

  expect_error(header_lexicon(con, list(sections = sections, preamble = "Name")), "Allergies", fixed = TRUE)
  expect_error(
    header_lexicon(con, list(sections = sections[1, ], preamble = "allergies")),
    "Allergies", fixed = TRUE
  )
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

test_that("note_headers nebere řetězec s normalizovaným tvarem pod MIN_HEADER_CHARS, čas, značku ani text uprostřed řádku", {
  found <- found_headers(c(
    "Ab: x", "Ab.: x", "Ab : x", "Time 10:30", "# CHF: yes", "see Chief Complaint: x"
  ))

  expect_equal(unname(found), rep(list(character()), 6L))
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
  expect_equal(out$zarazeni, c("mimo lexikon", "Hlavní obtíže", "administrativní pole"))
  expect_equal(out$druh, c(KIND_OUTSIDE, KIND_SECTION, KIND_PREAMBLE))
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
    druh = c(KIND_SECTION, KIND_PREAMBLE, KIND_OUTSIDE, KIND_OUTSIDE),
    zarazeni = c("Hlavní obtíže", "administrativní pole", "mimo lexikon", "mimo lexikon"),
    stringsAsFactors = FALSE
  )
}

totals_fixture <- function(celkem = 1000) {
  list(celkem = celkem, unikatni = 123, s_sekci = 900, bez_hlavicky = 50)
}

test_that("header_table uvede hlavičku, zařazení, počet a podíl", {
  out <- header_table(ranking_fixture())

  expect_equal(out$hlavicka, c("Chief Complaint", "Service", "IMAGING", "Exam"))
  expect_equal(out$zarazeni, c("Hlavní obtíže", "administrativní pole", "mimo lexikon", "mimo lexikon"))
  expect_equal(out$n, c("1 000", "990", "600", "200"))
  expect_equal(out$podil, c("100,0", "99,0", "60,0", "20,0"))
})

test_that("ranking_bars obarví podle druhu zařazení a seřadí nejčastější nahoru", {
  out <- ranking_bars(ranking_fixture())

  expect_equal(as.character(out$druh), c("sekce lexikonu", "administrativní pole", "mimo lexikon", "mimo lexikon"))
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

test_that("header_candidates vrátí všechny hlavičky mimo lexikon nad CANDIDATE_COVERAGE i za RANKING_SIZE", {
  codes <- paste0(rep(LETTERS, each = 26), rep(LETTERS, 26))[seq_len(RANKING_SIZE + 5L)]
  texts <- c(
    rep(paste(c(sprintf("Head %s: x", codes), "Allergies: x"), collapse = "\n"), 12),
    rep("Few: x", 4)
  )
  con <- corpus_con(texts)

  out <- header_candidates(con, LEXICON)

  expect_equal(nrow(header_ranking(con, LEXICON)), RANKING_SIZE)
  expect_equal(nrow(out), RANKING_SIZE + 5L)
  expect_true(tolower(sprintf("Head %s", codes[[RANKING_SIZE + 5L]])) %in% out$norm)
  expect_false(any(c("allergies", "few") %in% out$norm))
  expect_true(all(out$n / out$celkem >= CANDIDATE_COVERAGE))
})

test_that("header_candidates nad prázdným korpusem nevrátí nic", {
  expect_equal(nrow(header_candidates(mimic_con(), LEXICON)), 0L)
})

test_that("candidates_text vyjmenuje kandidáty, nebo napíše žádný", {
  candidates <- ranking_fixture()[3, ]

  expect_equal(
    candidates_text(candidates, totals_fixture()),
    "Kandidáti na doplnění lexikonu (hlavičky mimo lexikon s pokrytím aspoň 50\u00a0%): `IMAGING` (60,0\u00a0%)."
  )
  expect_match(candidates_text(candidates[0, ], totals_fixture()), "žádný", fixed = TRUE)
  expect_match(candidates_text(candidates[0, ], totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

synonyms_fixture <- function(celkem = 1000, dead = 2, repeated = 2) {
  count <- max(dead, repeated, 1L)
  data.frame(
    sekce = rep("a", count), synonymum = paste0("S", seq_len(count)),
    n = ifelse(seq_len(count) <= dead, 0, 500),
    opakovani = ifelse(seq_len(count) <= repeated, 50, 49),
    celkem = celkem, stringsAsFactors = FALSE
  )
}

test_that("dead_synonyms_text skloňuje sloveso i podstatné jméno pro 1, 2 a 5 synonym", {
  sentence <- function(dead) {
    dead_synonyms_text(synonyms_fixture(dead = dead, repeated = 0)[seq_len(max(dead, 1)), ], totals_fixture())
  }

  expect_equal(sentence(1), "Bez jediné zprávy je 1\u00a0synonymum: `S1`.")
  expect_equal(sentence(2), "Bez jediné zprávy jsou 2\u00a0synonyma: `S1` a `S2`.")
  expect_equal(sentence(5), "Bez jediné zprávy je 5\u00a0synonym: `S1`, `S2`, `S3`, `S4` a `S5`.")
  expect_match(dead_synonyms_text(synonyms_fixture(dead = 0, repeated = 0), totals_fixture()), "žádné")
  expect_match(dead_synonyms_text(synonyms_fixture(0), totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

test_that("repeated_synonyms_text skloňuje sloveso i podstatné jméno a bere hranici včetně", {
  sentence <- function(repeated) {
    repeated_synonyms_text(synonyms_fixture(dead = 0, repeated = repeated), totals_fixture())
  }
  share <- "5,0\u00a0%"

  expect_equal(sentence(1), sprintf("V aspoň 5\u00a0%% zpráv se opakuje 1\u00a0synonymum: `S1` (%s).", share))
  expect_match(sentence(2), "se opakují 2\u00a0synonyma: `S1` (5,0\u00a0%) a `S2`", fixed = TRUE)
  expect_match(sentence(5), "se opakuje 5\u00a0synonym: ", fixed = TRUE)
  expect_match(sentence(0), "Žádné", fixed = TRUE)
  expect_match(repeated_synonyms_text(synonyms_fixture(0), totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})

test_that("coverage_text uvede pokrytí sekcí, zprávy bez hlavičky a počet unikátních hlaviček", {
  text <- coverage_text(totals_fixture())

  expect_match(text, "90,0 %", fixed = TRUE)
  expect_match(text, "5,0 %", fixed = TRUE)
  expect_match(text, "123 unikátních hlaviček", fixed = TRUE)
  expect_match(coverage_text(totals_fixture(0)), "nelze posoudit", fixed = TRUE)
})
