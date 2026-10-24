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

AGREEMENT_TEXTS <- c(
  n1 = "Brief Hospital Course:\nPt  was   ok\nand well.\n\nDischarge Medications:\nasa\n",
  n2 = "Brief Hospital Course:\nfirst part.\n\nPast Medical History:\nhx\n",
  n3 = "Brief Hospital Course:\nalpha beta gamma delta\n",
  n4 = "Brief Hospital Course:\nalpha beta gamma delta\n",
  n5 = "Brief Hospital Course:\nalpha beta gamma delta\n",
  n6 = "Brief Hospital Course: earlier admission\nreal story\n\nBrief Hospital Course:\ntruth text\n",
  n7 = "Brief Hospital Course:\nmiddle part\n",
  n8 = "Brief Hospital Course:\nsomething\n",
  n9 = "Allergies:\nnone\n",
  n10 = "Brief Hospital Course:\nonly in corpus\n",
  n11 = "Brief Hospital Course: old case\n\nPast Medical History:\nx\n\nBrief Hospital Course:\nnew case\n"
)
AGREEMENT_TARGETS <- c(
  n1 = "Pt was ok and well.",
  n2 = "first part. Past Medical History: hx more",
  n3 = "alpha beta gamma",
  n4 = "alpha beta",
  n5 = "alpha",
  n6 = "truth text",
  n7 = "start middle part",
  n8 = "totally different words",
  n9 = "whatever",
  n11 = "new case",
  mimo = "not in the corpus"
)

agreement_con <- function(texts = AGREEMENT_TEXTS, targets = AGREEMENT_TARGETS, env = parent.frame()) {
  bhc <- data.frame(note_id = names(targets), target = unname(targets), stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), env = env, `mimic-iv-bhc` = bhc)
  # corpus_con pojmenovává zprávy n1…, shodně s klíči AGREEMENT_TEXTS
  segment_corpus(con, sections_module(), section_headers_path())
  con
}

category_row <- function(agreement, key) {
  agreement$kategorie[agreement$kategorie$kategorie == key, ]
}

test_that("bhc_agreement zařadí zprávy do všech sedmi kategorií se správným n", {
  out <- bhc_agreement(agreement_con())

  expect_equal(out$kategorie$kategorie, names(BHC_CATEGORIES))
  expect_equal(
    setNames(out$kategorie$n, out$kategorie$kategorie),
    c(
      nenalezena = 1, shoda = 1, konec_driv = 1, konec_pozdeji = 3,
      zacatek_driv = 1, zacatek_pozdeji = 1, jina = 2
    )
  )
  expect_equal(out$sparovano, 10)
  expect_equal(out$jen_bhc, 1)
  expect_equal(out$jen_korpus, 1)
  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement uvádí medián rozdílu délky jen u kategorií 3 až 6", {
  out <- bhc_agreement(agreement_con())
  median_of <- function(key) category_row(out, key)$median_rozdilu

  expect_equal(median_of("konec_driv"), nchar(AGREEMENT_TARGETS[["n2"]]) - nchar("first part."))
  expect_equal(median_of("konec_pozdeji"), 12)
  expect_equal(median_of("zacatek_pozdeji"), nchar("start middle part") - nchar("middle part"))
  expect_false(is.na(median_of("zacatek_driv")))
  expect_true(all(is.na(c(median_of("nenalezena"), median_of("shoda"), median_of("jina")))))
})

test_that("bhc_agreement: úsek čištěný jednou je shodou s už vyčištěným targetem", {
  texts <- c(n1 = "Brief Hospital Course:\nBrief Hospital\nCourse: stable\n")
  targets <- c(n1 = "Brief Hospital Course: stable")
  out <- bhc_agreement(agreement_con(texts, targets))

  expect_equal(category_row(out, "shoda")$n, 1)
  # Slitá fráze není pevným bodem čištění, takže ji kontrola targetu počítá; je to vzácný
  # falešný poplach, který druhé čištění targetu nahrazovat nesmí.
  expect_equal(out$necisty_target, 1)
})

test_that("bhc_agreement počítá targety, které nejsou pevným bodem čištění", {
  texts <- rep("Brief Hospital Course:\nalpha beta\n", 3)
  targets <- c(n1 = "alpha beta", n2 = "alpha\nbeta", n3 = "alpha  beta")
  out <- bhc_agreement(agreement_con(setNames(texts, names(targets)), targets))

  expect_equal(out$necisty_target, 2)
})

test_that("bhc_agreement: falešný první výskyt s jinou hlavičkou lexikonu je jiná neshoda", {
  texts <- c(n1 = AGREEMENT_TEXTS[["n11"]])
  out <- bhc_agreement(agreement_con(texts, c(n1 = "new case")))

  expect_equal(category_row(out, "jina")$n, 1)
  expect_equal(category_row(out, "zacatek_driv")$n, 0)
})

test_that("bhc_agreement: prázdný průnik dá nuly ve všech kategoriích", {
  out <- bhc_agreement(agreement_con(c(n1 = "Allergies:\nx\n"), c(jiny = "text")))

  expect_equal(out$sparovano, 0)
  expect_equal(out$jen_bhc, 1)
  expect_equal(out$jen_korpus, 1)
  expect_equal(out$kategorie$n, rep(0, length(BHC_CATEGORIES)))
  expect_true(all(is.na(out$kategorie$median_rozdilu)))
})

test_that("bhc_agreement zastaví render na duplicitním note_id nebo NULL v targetu", {
  texts <- c(n1 = "Brief Hospital Course:\nalpha\n")
  duplicated <- data.frame(note_id = c("n1", "n1"), target = c("a", "b"), stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), `mimic-iv-bhc` = duplicated)
  segment_corpus(con, sections_module(), section_headers_path())
  expect_error(bhc_agreement(con), "duplicit", fixed = TRUE)

  missing <- data.frame(note_id = "n1", target = NA_character_, stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), `mimic-iv-bhc` = missing)
  segment_corpus(con, sections_module(), section_headers_path())
  expect_error(bhc_agreement(con), "NULL", fixed = TRUE)
})

test_that("bhc_agreement bez tabulky mimic-iv-bhc radí spustit snakemake parquet_mimic", {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_error(bhc_agreement(con), "parquet_mimic", fixed = TRUE)
})
