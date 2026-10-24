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
  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement počítá targety s dvojitou mezerou, zalomením nebo jiným bílým znakem", {
  texts <- rep("Brief Hospital Course:\nalpha beta\n", 5)
  targets <- c(
    n1 = "alpha beta", n2 = "alpha\nbeta", n3 = "alpha  beta", n4 = "alpha\u00a0beta",
    n5 = " alpha beta "
  )
  out <- bhc_agreement(agreement_con(setNames(texts, names(targets)), targets))

  expect_equal(out$necisty_target, 3)
})

test_that("bhc_agreement počítá necisty_target jen ze spárovaných zpráv", {
  targets <- c(n1 = "alpha beta", mimo = "a  b")
  out <- bhc_agreement(agreement_con(c(n1 = "Brief Hospital Course:\nalpha beta\n"), targets))

  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement: shoda s targetem s okrajovými mezerami", {
  texts <- c(n1 = "Brief Hospital Course:\nPt ok\n")
  out <- bhc_agreement(agreement_con(texts, c(n1 = " Pt ok ")))

  expect_equal(category_row(out, "shoda")$n, 1)
})

test_that("bhc_agreement uvádí jen_bhc a jen_korpus zvlášť, i když se liší", {
  texts <- c(
    n1 = "Brief Hospital Course:\nalpha\n", n2 = "Brief Hospital Course:\nalpha\n",
    n3 = "Brief Hospital Course:\nalpha\n"
  )
  targets <- c(n1 = "alpha", mimo1 = "x", mimo2 = "y")
  out <- bhc_agreement(agreement_con(texts, targets))

  expect_equal(out$sparovano, 1)
  expect_equal(out$jen_bhc, 2)
  expect_equal(out$jen_korpus, 2)
})

test_that("bhc_agreement: medián sudého počtu rozdílů je nižší z prostředních hodnot", {
  texts <- rep("Brief Hospital Course:\nalpha beta gamma delta\n", 4)
  targets <- c(n1 = "alpha beta gamma", n2 = "alpha beta", n3 = "alpha", n4 = "alpha beta gamma delt")
  out <- bhc_agreement(agreement_con(setNames(texts, names(targets)), targets))

  # rozdíly 6, 12, 17, 1: seřazeně 1, 6, 12, 17, quantile_disc(0,5) je 6
  expect_equal(category_row(out, "konec_pozdeji")$median_rozdilu, 6)
})

test_that("BHC_WHITESPACE odpovídá Pythonovu \\s přes všechny code pointy a čištění je sloučí", {
  codepoints <- c(
    9:13, 28:32, 133, 160, 5760, 8192:8202, 8232, 8233, 8239, 8287, 12288
  )
  expect_length(codepoints, 29L)
  inputs <- paste0("a", intToUtf8(codepoints, multiple = TRUE), "b")

  expect_equal(clean_in_sql(inputs), rep("a b", length(codepoints)))
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

test_that("bhc_agreement zastaví render na note_id NULL", {
  texts <- c(n1 = "Brief Hospital Course:\nalpha\n")
  rows <- data.frame(note_id = c("n1", NA), target = c("a", "b"), stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), `mimic-iv-bhc` = rows)
  segment_corpus(con, sections_module(), section_headers_path())

  expect_error(bhc_agreement(con), "duplicit", fixed = TRUE)
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

fake_agreement <- function(
    n = c(1, 40, 2, 5, 3, 4, 45), median = c(NA, NA, 7, 12, 30, 9, NA),
    jen_bhc = 11, jen_korpus = 6, necisty_target = 0) {
  list(
    sparovano = sum(n), jen_bhc = jen_bhc, jen_korpus = jen_korpus, necisty_target = necisty_target,
    kategorie = data.frame(
      kategorie = names(BHC_CATEGORIES), popisek = unname(BHC_CATEGORIES),
      n = n, median_rozdilu = median, stringsAsFactors = FALSE
    )
  )
}

test_that("bhc_table: řádky v pořadí BHC_CATEGORIES bez ohledu na pořadí agregátu", {
  agreement <- fake_agreement()
  shuffled <- agreement
  shuffled$kategorie <- agreement$kategorie[c(7, 3, 1, 5, 2, 6, 4), ]
  table <- bhc_table(shuffled)

  expect_length(table, length(BHC_LAYOUTS[["tbl-bhc"]]$col.names))
  expect_equal(table[[1]], unname(BHC_CATEGORIES))
  expect_equal(table[[2]], number_text(c(1, 40, 2, 5, 3, 4, 45)))
  expect_equal(table[[3]], percent_text(c(1, 40, 2, 5, 3, 4, 45), 100))
  expect_equal(table[[4]], c("–", "–", "7", "12", "30", "9", "–"))
})

test_that("bhc_plot: úrovně v pořadí BHC_CATEGORIES a podíl ze spárovaných zpráv", {
  agreement <- fake_agreement()
  shuffled <- agreement
  shuffled$kategorie <- agreement$kategorie[c(7, 3, 1, 5, 2, 6, 4), ]
  plot <- bhc_plot(shuffled)

  expect_equal(levels(plot$data$popisek), unname(BHC_CATEGORIES))
  expect_equal(plot$data$podil[plot$data$popisek == BHC_CATEGORIES[["shoda"]]], 0.4)
  expect_true(all(bhc_plot(fake_agreement(n = rep(0, 7)))$data$podil == 0))
})

test_that("bhc nad prázdným průnikem: tabulka, graf i export projdou, věta nelze posoudit", {
  empty <- bhc_agreement(agreement_con(c(n1 = "Allergies:\nx\n"), c(jiny = "text")))
  dir <- withr::local_tempdir()
  layout <- BHC_LAYOUTS[["tbl-bhc"]]

  expect_no_error(ggplot2::ggplot_build(bhc_plot(empty)))
  expect_no_error(export_table(
    bhc_table(empty), col.names = layout$col.names, align = layout$align,
    widths = layout$widths, label = "tbl-bhc", dir = dir
  ))
  expect_true(file.exists(file.path(dir, "tbl-bhc.tex")))
  expect_equal(bhc_table(empty)[[3]], rep("–", length(BHC_CATEGORIES)))
  text <- bhc_text(empty)
  expect_match(text, "nelze posoudit")
  expect_match(text, "žádná zpráva korpusu nemá pár")
  expect_no_match(text, "prázdný")
})

test_that("bhc_text nad neprázdným korpusem a BHC bez společného note_id uvede správný důvod", {
  text <- bhc_text(bhc_agreement(agreement_con(c(n1 = "Brief Hospital Course:\nx\n"), c(jiny = "t"))))

  expect_match(text, "nelze posoudit")
  expect_match(text, "žádná zpráva korpusu nemá pár")
})

test_that("bhc_text jmenuje podíl shody, počet spárovaných zpráv a populace mimo průnik", {
  text <- bhc_text(fake_agreement())

  expect_match(text, "Počet spárovaných zpráv: 100.", fixed = TRUE)
  expect_match(text, paste0("u 40,0", NBSP, "% spárovaných zpráv (40", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, paste0("jen v BHC (11", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, paste0("jen v korpusu (6", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, "Neshody: Sekce nenalezena", fixed = TRUE)
  expect_no_match(text, "Neshody: Shoda")
})

test_that("bhc_text skloňuje počty: 1 zpráva, 2 zprávy, 5 zpráv", {
  for (case in list(c(1, 1, 2), c(2, 2, 5), c(5, 5, 1))) {
    text <- bhc_text(fake_agreement(
      n = c(0, case[[1]], 0, 0, 0, 0, 0), jen_bhc = case[[2]], jen_korpus = case[[3]]
    ))
    expect_match(text, paste0("spárovaných zpráv (", count_phrase(case[[1]], NOTE_FORMS), ")"), fixed = TRUE)
    expect_match(text, paste0("jen v BHC (", count_phrase(case[[2]], NOTE_FORMS), ")"), fixed = TRUE)
    expect_match(text, paste0("jen v korpusu (", count_phrase(case[[3]], NOTE_FORMS), ")"), fixed = TRUE)
  }
  expect_equal(count_phrase(c(1, 2, 5), NOTE_FORMS), paste0(c(1, 2, 5), NBSP, c("zpráva", "zprávy", "zpráv")))
})

test_that("bhc_text uvede počet targetů, které čištěním zjevně neprošly", {
  expect_match(bhc_text(fake_agreement(necisty_target = 3)), paste0("(3", NBSP, "zprávy)"), fixed = TRUE)
})

CUT_TEXTS <- c(
  n1 = "Brief Hospital Course:\nstory one\nDischarge Diagnosis:\ntail one\n",
  n2 = "Brief Hospital Course:\nstory two\nDischarge Diagnosis:\ntail two\n",
  n3 = "Brief Hospital Course:\nstory three\nPast Medical History:\ntail three\n",
  n4 = "Brief Hospital Course:\nstory four\nAttending: tail four\n",
  n5 = "Brief Hospital Course:\nstory five\nDischarge Diagnosis:\nx\n"
)
CUT_TARGETS <- c(
  n1 = "story one Discharge Diagnosis: tail one",
  n2 = "story two Discharge Diagnosis: tail two",
  n3 = "story three Past Medical History: tail three",
  n4 = "story four Attending: tail four",
  n5 = "story five"
)

cut_con <- function(texts = CUT_TEXTS, targets = CUT_TARGETS, env = parent.frame()) {
  con <- agreement_con(texts, targets, env = env)
  bhc_agreement(con)
  con
}

test_that("bhc_cut_sections rozloží konec dřív podle části, která úsek uřízla", {
  cut <- bhc_cut_sections(cut_con(), read_section_headers(section_headers_path()))

  expect_equal(cut$sekce, c("discharge_diagnosis", "past_medical_history", "other"))
  expect_equal(cut$n, c(2, 1, 1))
  expect_equal(cut$popisek[c(1, 3)], c("Diagnóza při propuštění", BHC_FIELD_LABEL))
  expect_equal(cut$diagnosticka, c(TRUE, FALSE, FALSE))
  expect_equal(cut$celkem, rep(4, 3))
})

test_that("bhc_cut_sections bez tabulky párů zastaví s návodem", {
  con <- agreement_con()
  expect_error(bhc_cut_sections(con, read_section_headers(section_headers_path())), "bhc_agreement", fixed = TRUE)
})

test_that("bhc_cut_sections bez zprávy v kategorii konec dřív vrátí prázdnou tabulku", {
  con <- cut_con(c(n1 = CUT_TEXTS[["n5"]]), c(n1 = CUT_TARGETS[["n5"]]))
  cut <- bhc_cut_sections(con, read_section_headers(section_headers_path()))
  expect_equal(nrow(cut), 0L)
  expect_match(bhc_cut_text(cut), "nelze posoudit")
  expect_no_error(export_table(
    bhc_cut_table(cut),
    col.names = BHC_LAYOUTS[["tbl-bhc-konec"]]$col.names,
    align = BHC_LAYOUTS[["tbl-bhc-konec"]]$align,
    widths = BHC_LAYOUTS[["tbl-bhc-konec"]]$widths,
    dir = withr::local_tempdir(), label = "tbl-bhc-konec"
  ))
})

test_that("bhc_cut_text jmenuje nejčastější část a počet uříznutí diagnostickou sekcí", {
  text <- bhc_cut_text(bhc_cut_sections(cut_con(), read_section_headers(section_headers_path())))

  expect_match(text, "Diagnóza při propuštění (2 zprávy, 50,0 %)", fixed = TRUE)
  expect_match(text, "diagnostická sekce u 50,0 % z nich (2 zprávy)", fixed = TRUE)
})

test_that("bhc_cut_table: sekce, popisek, zprávy a podíl z kategorie konec dřív", {
  table <- bhc_cut_table(bhc_cut_sections(cut_con(), read_section_headers(section_headers_path())))
  expect_equal(table$sekce, c("discharge_diagnosis", "past_medical_history", "other"))
  expect_equal(table$n, c("2", "1", "1"))
  expect_equal(table$podil, c("50,0", "25,0", "25,0"))
})
