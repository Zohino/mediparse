MODULE <- preprocessing_module()
PATH <- candidates_path()
# corpus_con předřadí každé zprávě řádek `xxx… h<n>`, což jsou dva tokeny v obou variantách.
PADDING_TOKENS <- 2L

chunk_of <- function(texts) {
  list(note_id = paste0("n", seq_along(texts)), text = texts)
}

test_that("preprocessing_rows spočítá tokeny a zkratky jedné zprávy a typy sedí", {
  acronyms <- corpus_acronyms(MODULE, PATH)
  out <- preprocessing_rows(MODULE, PATH, acronyms, chunk_of("Pt 5mg ___ AF"))

  expect_equal(names(out), c("note_id", "tokens_raw", "tokens_preprocessed", acronyms))
  expect_equal(out$tokens_raw, 4L)
  expect_equal(out$tokens_preprocessed, 6L)
  expect_equal(out$AF, 1L)
  expect_equal(out$DM2, 0L)
  expect_type(out$tokens_raw, "integer")
  expect_type(out$COPD, "integer")
})

test_that("preprocessing_rows a vocabulary_rows zvládnou prázdnou dávku s typovanými sloupci", {
  acronyms <- corpus_acronyms(MODULE, PATH)
  empty <- list(note_id = character(), text = character())
  rows <- preprocessing_rows(MODULE, PATH, acronyms, empty)
  vocabulary <- vocabulary_rows(MODULE, PATH, empty)

  expect_equal(nrow(rows), 0L)
  expect_equal(names(rows), c("note_id", "tokens_raw", "tokens_preprocessed", acronyms))
  expect_type(rows$tokens_raw, "integer")
  expect_type(rows$AF, "integer")
  expect_equal(nrow(vocabulary), 0L)
  expect_equal(names(vocabulary), names(VOCABULARY_COLUMNS))
  expect_type(vocabulary$token, "character")
})

test_that("klíče PREPROCESSING_VARIANTS odpovídají variantám mostu", {
  expect_equal(names(PREPROCESSING_VARIANTS), c(MODULE$RAW, MODULE$PREPROCESSED))
})

test_that("note_preprocessing dá řádek na zprávu vstupního korpusu i přes víc dávek a podruhé nepřepočítá", {
  con <- corpus_con(c("Pt AF", "Pt DM2 and DM", "Pt"))
  note_preprocessing(con, MODULE, PATH, batch = 2L)
  table <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s ORDER BY note_id", PREPROCESSING_TABLE))

  expect_equal(nrow(table), 3L)
  expect_equal(table$AF, c(1L, 0L, 0L))
  expect_equal(table$DM2, c(0L, 1L, 0L))
  expect_equal(table$DM, c(0L, 1L, 0L))
  expect_equal(table$tokens_raw, PADDING_TOKENS + c(2L, 4L, 1L))
  expect_equal(table$tokens_preprocessed, PADDING_TOKENS + c(3L, 4L, 1L))

  DBI::dbExecute(con, sprintf("DELETE FROM %s", PREPROCESSING_TABLE))
  note_preprocessing(con, MODULE, PATH, batch = 2L)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", PREPROCESSING_TABLE))$n, 0)
})

test_that("slovník je sjednocení přes dávky: sdílený token se započte jednou", {
  con <- corpus_con(c("alpha AF", "alpha gamma"))
  note_preprocessing(con, MODULE, PATH)
  note_vocabulary(con, MODULE, PATH, batch = 1L)
  summary <- preprocessing_summary(con)

  # xxx…, h1, h2, alpha, af, gamma a v druhé variantě atrial, fibrillation místo af.
  expect_equal(summary$slovnik, c(6, 7))
  expect_equal(summary$varianta, c("raw", "preprocessed"))
})

test_that("preprocessing_summary uvádí medián, p95 a průměr tokenů na zprávu", {
  con <- corpus_con(c("AF", "AF AF", "alpha"))
  note_preprocessing(con, MODULE, PATH)
  note_vocabulary(con, MODULE, PATH)
  summary <- preprocessing_summary(con)

  expect_equal(summary$p50, PADDING_TOKENS + c(1, 2))
  expect_equal(summary$p95, PADDING_TOKENS + c(2, 4))
  expect_equal(summary$n, c(3, 3))
  expect_equal(summary$prumer, PADDING_TOKENS + c(4 / 3, 7 / 3))
  table <- preprocessing_table(summary)
  expect_equal(table$varianta, unname(PREPROCESSING_LABELS))
  expect_equal(table$median, c("3", "4"))
  expect_equal(table$p95, c("4", "6"))
  expect_equal(table$prumer, c("3,3", "4,3"))
})

test_that("prázdný korpus dá slovník 0, percentily NA a tabulku s pomlčkami", {
  con <- mimic_con()
  note_preprocessing(con, MODULE, PATH)
  note_vocabulary(con, MODULE, PATH)
  summary <- preprocessing_summary(con)
  table <- preprocessing_table(summary)

  expect_equal(summary$slovnik, c(0, 0))
  expect_true(all(is.na(summary$p50)))
  expect_equal(table$slovnik, c("0", "0"))
  expect_equal(table$median, c("–", "–"))
  expect_equal(table$prumer, c("–", "–"))
  expect_match(preprocessing_text(summary), "nelze posoudit")
})

# Zprávy čtyř skupin buněk 2×2 (zkratka AF × kód fibrilace); `repeats` je počet výskytů AF v každé
# zprávě se zkratkou, jednotlivě nebo pro všechny, aby šly nastavit výskyty nezávisle na zprávách.
af_groups <- function(oba = 10L, jen_zkratka = 10L, jen_kod = 10L, zadne = 10L, repeats = 60L) {
  with_acronym <- oba + jen_zkratka
  repeats <- rep_len(repeats, with_acronym)
  list(
    texts = c(
      vapply(repeats, function(n) paste("Pt", paste(rep("AF", n), collapse = " ")), character(1)),
      rep("Pt stable", jen_kod + zadne)
    ),
    codes = c(
      rep("I48", oba), rep("Z99", jen_zkratka), rep("I48", jen_kod), rep("Z99", zadne)
    )
  )
}

af_summary <- function(groups) {
  con <- if (length(groups$texts) == 0L) mimic_con() else corpus_con(groups$texts, codes = groups$codes)
  note_preprocessing(con, MODULE, PATH)
  acronym_summary(con, corpus_acronyms(MODULE, PATH), read_candidates(PATH))
}

af_row <- function(summary) summary[summary$zkratka == "AF", ]

test_that("acronym_summary zveřejní sedm výskytů a podíl tří z pěti zpráv", {
  summary <- af_summary(af_groups(oba = 2L, jen_zkratka = 3L, jen_kod = 1L, zadne = 94L, repeats = c(3L, 1L, 1L, 1L, 1L)))
  af <- af_row(summary)
  table <- acronym_table(af)

  expect_equal(af$vyskyty, 7)
  expect_equal(af$podil, 0.6)
  expect_equal(af$kandidat, "atrial_fibrillation")
  expect_equal(af$label_cs, "Fibrilace síní")
  expect_equal(table$vyskyty, "7")
  expect_equal(table$bez_kodu, "60,0")
})

test_that("podíl se uvádí i pro jedinou zprávu se zkratkou", {
  summary <- af_summary(af_groups(oba = 0L, jen_zkratka = 1L, jen_kod = 0L, zadne = 0L, repeats = 1L))

  expect_equal(af_row(summary)$podil, 1)
  expect_equal(acronym_table(af_row(summary))$bez_kodu, "100,0")
})

test_that("výskyty zkratek se v tabulce zobrazí přesně", {
  summary <- data.frame(
    zkratka = c("A", "B", "C", "D", "E"), label_cs = "x", vyskyty = c(0, 5, 9, 10, 1500),
    podil = NA_real_, stringsAsFactors = FALSE
  )

  expect_equal(acronym_table(summary)$vyskyty, c("0", "5", "9", "10", "1\u00a0500"))
})

test_that("souhrn a tabulka zkratek zachovají požadované sloupce a pořadí", {
  summary <- af_summary(af_groups(jen_kod = 9L))

  expect_equal(names(summary), c("zkratka", "kandidat", "label_cs", "vyskyty", "podil"))
  expect_equal(names(acronym_table(summary)), c("zkratka", "label_cs", "vyskyty", "bez_kodu"))
  expect_equal(acronym_table(summary)$zkratka, sort(acronym_table(summary)$zkratka))
})

test_that("nulové výskyty a jmenovatel dají nedefinovaný podíl a vlastní větu", {
  summary <- af_summary(af_groups(oba = 0L, jen_zkratka = 0L, jen_kod = 1L, zadne = 99L))
  af <- af_row(summary)

  expect_equal(af$vyskyty, 0)
  expect_true(is.na(af$podil))
  expect_equal(acronym_table(af)$bez_kodu, "–")
  expect_match(acronym_text(summary), "zkratka nemá výskyt", fixed = TRUE)
})

test_that("acronym_summary nad prázdným korpusem nespadne", {
  summary <- af_summary(list(texts = character(), codes = character()))

  expect_equal(nrow(summary), length(corpus_acronyms(MODULE, PATH)))
  expect_true(all(is.na(summary$podil)))
  expect_equal(acronym_table(summary)$vyskyty, rep("0", nrow(summary)))
  expect_match(acronym_text(summary), "zkratka nemá výskyt", fixed = TRUE)
})

test_that("acronym_summary bez zkratek vrátí prázdný souhrn a tabulku", {
  con <- mimic_con()
  summary <- acronym_summary(con, character(), read_candidates(PATH))

  expect_equal(names(summary), c("zkratka", "kandidat", "label_cs", "vyskyty", "podil"))
  expect_equal(nrow(acronym_table(summary)), 0L)
  expect_match(acronym_text(summary), "nelze posoudit", fixed = TRUE)
})

test_that("acronym_text srovnává AF s mediánem ostatních zkratek", {
  base <- data.frame(
    zkratka = c("AF", "COPD", "DM", "CKD"), vyskyty = 10,
    podil = c(0.5, 0.1, 0.3, 0.2), stringsAsFactors = FALSE
  )
  expect_match(acronym_text(base), paste0("AF je bez kódu diagnózy 50,0", NBSP, "%, u ostatních zkratek v mediánu 20,0", NBSP, "%"))

  absent <- base
  absent$podil[1] <- NA_real_
  absent$vyskyty[1] <- 0
  expect_match(acronym_text(absent), "zkratka nemá výskyt", fixed = TRUE)

  alone <- base
  alone$podil[-1] <- NA_real_
  alone$vyskyty[-1] <- 0
  expect_match(acronym_text(alone), "ostatní zkratky nemají výskyt", fixed = TRUE)
})

test_that("preprocessing_text skloňuje počty tokenů a uvádí změnu slovníku", {
  summary <- function(raw_vocabulary, prepared_vocabulary, median_raw = 5, median_prepared = 2) {
    data.frame(
      varianta = c("raw", "preprocessed"), slovnik = c(raw_vocabulary, prepared_vocabulary),
      n = 3, p50 = c(median_raw, median_prepared), p95 = c(9, 8), prumer = c(5, 4),
      stringsAsFactors = FALSE
    )
  }

  expect_match(preprocessing_text(summary(1, 5)), paste0("1", NBSP, "token,"))
  expect_match(preprocessing_text(summary(1, 5)), paste0("5", NBSP, "tokenů"))
  expect_match(preprocessing_text(summary(2, 3, 1, 4)), paste0("2", NBSP, "tokeny"))
  expect_match(preprocessing_text(summary(2, 3, 1, 4)), paste0("1", NBSP, "token"))
  expect_match(preprocessing_text(summary(100, 80)), paste0("změna o -20,0", NBSP, "%"))
  expect_match(preprocessing_text(summary(0, 0)), paste0("0", NBSP, "tokenů"))
  expect_no_match(preprocessing_text(summary(0, 0)), "změna")
})

test_that("tabulky projdou export_table s rozvržením a funkce zvládnou jedinečné řetězce", {
  texts <- paste("jedinecny-retezec-1 AF", c("jedinecny-retezec-2", "jedinecny-retezec-3"))
  con <- corpus_con(texts, codes = c("I48", "Z99"))
  note_preprocessing(con, MODULE, PATH)
  note_vocabulary(con, MODULE, PATH)
  dir <- withr::local_tempdir()
  summary <- preprocessing_summary(con)
  acronyms <- acronym_summary(con, corpus_acronyms(MODULE, PATH), read_candidates(PATH))

  for (spec in list(
    list("tbl-predzpracovani", preprocessing_table(summary)),
    list("tbl-zkratky", acronym_table(acronyms))
  )) {
    layout <- PREPROCESSING_LAYOUTS[[spec[[1]]]]
    export_table(
      spec[[2]], col.names = layout$col.names, align = layout$align,
      widths = layout$widths, label = spec[[1]], dir = dir
    )
  }

  expect_true(all(file.exists(file.path(dir, paste0(names(PREPROCESSING_LAYOUTS), ".tex")))))
  expect_no_error(preprocessing_text(summary))
  expect_no_error(acronym_text(acronyms))
})
