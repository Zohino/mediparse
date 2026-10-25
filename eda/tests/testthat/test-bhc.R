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

OTHER_KEY <- sections_module()$OTHER

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
  n11 = "Brief Hospital Course: old case\n\nPast Medical History:\nx\n\nBrief Hospital Course:\nnew case\n",
  n12 = "Brief Hospital Course:\nfirst part.\n\nPast Medical History:\nhx\n",
  n13 = "Brief Hospital Course:\nstory\n\nDischarge Medications:\nasa daily\n",
  n14 = "Brief Hospital Course:\nstory\n\n  Discharge Medications:\nasa daily\n",
  n15 = "Brief Hospital Course:\nstory\n\n\tDischarge Medications:\nasa daily\n",
  n16 = "Brief Hospital Course:\nstory\nAttending: tail\n",
  n17 = "Brief Hospital Course:\nstory\n",
  n18 = paste0(
    "Brief Hospital Course:\nstory\nDischarge medications:\nasa\n\n",
    "Medications on Admission:\nx\n\nDischarge Medications:\nasa\n"
  )
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
  n12 = "first part. unrelated words",
  n13 = "story Discharge Medications: asa",
  n14 = "story Discharge Medications: asa daily",
  n15 = "story Discharge Medications: asa daily",
  n16 = "story Attending: tail",
  n17 = "story and more",
  n18 = "story Discharge medications: asa",
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

test_that("bhc_agreement zařadí zprávy do všech osmi kategorií se správným n", {
  out <- bhc_agreement(agreement_con(), OTHER_KEY)

  expect_equal(out$kategorie$kategorie, names(BHC_CATEGORIES))
  expect_equal(
    setNames(out$kategorie$n, out$kategorie$kategorie),
    c(
      nenalezena = 1, shoda = 1, cela_sekce = 4, konec_driv = 4, konec_pozdeji = 3,
      zacatek_driv = 1, zacatek_pozdeji = 1, jina = 2
    )
  )
  expect_equal(out$sparovano, 17)
  expect_equal(out$jen_bhc, 1)
  expect_equal(out$jen_korpus, 1)
  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement uvádí medián rozdílu délky jen u kategorií 3 až 7", {
  out <- bhc_agreement(agreement_con(), OTHER_KEY)
  median_of <- function(key) category_row(out, key)$median_rozdilu

  # rozdíly 16, 27, 16, 9 (n12, n13, n16, n17): seřazeně 9, 16, 16, 27, quantile_disc(0,5) je 16
  expect_equal(median_of("konec_driv"), 16)
  cela <- c(
    nchar(AGREEMENT_TARGETS[["n2"]]) - nchar("first part."),
    nchar(AGREEMENT_TARGETS[["n14"]]) - nchar("story"),
    nchar(AGREEMENT_TARGETS[["n15"]]) - nchar("story"),
    nchar(AGREEMENT_TARGETS[["n18"]]) - nchar("story")
  )
  expect_equal(median_of("cela_sekce"), sort(cela)[2])
  expect_equal(median_of("konec_pozdeji"), 12)
  expect_equal(median_of("zacatek_pozdeji"), nchar("start middle part") - nchar("middle part"))
  expect_false(is.na(median_of("zacatek_driv")))
  expect_true(all(is.na(c(median_of("nenalezena"), median_of("shoda"), median_of("jina")))))
})

test_that("bhc_agreement: úsek čištěný jednou je shodou s už vyčištěným targetem", {
  texts <- c(n1 = "Brief Hospital Course:\nBrief Hospital\nCourse: stable\n")
  targets <- c(n1 = "Brief Hospital Course: stable")
  out <- bhc_agreement(agreement_con(texts, targets), OTHER_KEY)

  expect_equal(category_row(out, "shoda")$n, 1)
  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement počítá targety s dvojitou mezerou, zalomením nebo jiným bílým znakem", {
  texts <- rep("Brief Hospital Course:\nalpha beta\n", 5)
  targets <- c(
    n1 = "alpha beta", n2 = "alpha\nbeta", n3 = "alpha  beta", n4 = "alpha\u00a0beta",
    n5 = " alpha beta "
  )
  out <- bhc_agreement(agreement_con(setNames(texts, names(targets)), targets), OTHER_KEY)

  expect_equal(out$necisty_target, 3)
})

test_that("bhc_agreement počítá necisty_target jen ze spárovaných zpráv", {
  targets <- c(n1 = "alpha beta", mimo = "a  b")
  out <- bhc_agreement(agreement_con(c(n1 = "Brief Hospital Course:\nalpha beta\n"), targets), OTHER_KEY)

  expect_equal(out$necisty_target, 0)
})

test_that("bhc_agreement: shoda s targetem s okrajovými mezerami", {
  texts <- c(n1 = "Brief Hospital Course:\nPt ok\n")
  out <- bhc_agreement(agreement_con(texts, c(n1 = " Pt ok ")), OTHER_KEY)

  expect_equal(category_row(out, "shoda")$n, 1)
})

test_that("bhc_agreement uvádí jen_bhc a jen_korpus zvlášť, i když se liší", {
  texts <- c(
    n1 = "Brief Hospital Course:\nalpha\n", n2 = "Brief Hospital Course:\nalpha\n",
    n3 = "Brief Hospital Course:\nalpha\n"
  )
  targets <- c(n1 = "alpha", mimo1 = "x", mimo2 = "y")
  out <- bhc_agreement(agreement_con(texts, targets), OTHER_KEY)

  expect_equal(out$sparovano, 1)
  expect_equal(out$jen_bhc, 2)
  expect_equal(out$jen_korpus, 2)
})

test_that("bhc_agreement: medián sudého počtu rozdílů je nižší z prostředních hodnot", {
  texts <- rep("Brief Hospital Course:\nalpha beta gamma delta\n", 4)
  targets <- c(n1 = "alpha beta gamma", n2 = "alpha beta", n3 = "alpha", n4 = "alpha beta gamma delt")
  out <- bhc_agreement(agreement_con(setNames(texts, names(targets)), targets), OTHER_KEY)

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
  out <- bhc_agreement(agreement_con(texts, c(n1 = "new case")), OTHER_KEY)

  expect_equal(category_row(out, "jina")$n, 1)
  expect_equal(category_row(out, "zacatek_driv")$n, 0)
})

test_that("bhc_agreement: prázdný průnik dá nuly ve všech kategoriích", {
  out <- bhc_agreement(agreement_con(c(n1 = "Allergies:\nx\n"), c(jiny = "text")), OTHER_KEY)

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

  expect_error(bhc_agreement(con, OTHER_KEY), "duplicit", fixed = TRUE)
})

test_that("bhc_agreement zastaví render na duplicitním note_id nebo NULL v targetu", {
  texts <- c(n1 = "Brief Hospital Course:\nalpha\n")
  duplicated <- data.frame(note_id = c("n1", "n1"), target = c("a", "b"), stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), `mimic-iv-bhc` = duplicated)
  segment_corpus(con, sections_module(), section_headers_path())
  expect_error(bhc_agreement(con, OTHER_KEY), "duplicit", fixed = TRUE)

  missing <- data.frame(note_id = "n1", target = NA_character_, stringsAsFactors = FALSE)
  con <- corpus_con(unname(texts), `mimic-iv-bhc` = missing)
  segment_corpus(con, sections_module(), section_headers_path())
  expect_error(bhc_agreement(con, OTHER_KEY), "NULL", fixed = TRUE)
})

test_that("bhc_agreement bez tabulky mimic-iv-bhc radí spustit snakemake parquet_mimic", {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  expect_error(bhc_agreement(con, OTHER_KEY), "parquet_mimic", fixed = TRUE)
})

fake_agreement <- function(
    n = c(1, 40, 6, 2, 5, 3, 4, 39), median = c(NA, NA, 8, 7, 12, 30, 9, NA),
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
  shuffled$kategorie <- agreement$kategorie[c(8, 4, 1, 6, 2, 7, 3, 5), ]
  table <- bhc_table(shuffled)

  expect_length(table, length(BHC_LAYOUTS[["tbl-bhc"]]$col.names))
  expect_equal(table[[1]], unname(BHC_CATEGORIES))
  expect_equal(table[[2]], number_text(c(1, 40, 6, 2, 5, 3, 4, 39)))
  expect_equal(table[[3]], percent_text(c(1, 40, 6, 2, 5, 3, 4, 39), 100))
  expect_equal(table[[4]], c("–", "–", "8", "7", "12", "30", "9", "–"))
})

test_that("bhc_plot: úrovně v pořadí BHC_CATEGORIES a podíl ze spárovaných zpráv", {
  agreement <- fake_agreement()
  shuffled <- agreement
  shuffled$kategorie <- agreement$kategorie[c(8, 4, 1, 6, 2, 7, 3, 5), ]
  plot <- bhc_plot(shuffled)

  expect_equal(levels(plot$data$popisek), unname(BHC_CATEGORIES))
  expect_equal(plot$data$podil[plot$data$popisek == BHC_CATEGORIES[["shoda"]]], 0.4)
  expect_true(all(bhc_plot(fake_agreement(n = rep(0, 8)))$data$podil == 0))
})

test_that("bhc nad prázdným průnikem: tabulka, graf i export projdou, věta nelze posoudit", {
  empty <- bhc_agreement(agreement_con(c(n1 = "Allergies:\nx\n"), c(jiny = "text")), OTHER_KEY)
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
  text <- bhc_text(bhc_agreement(agreement_con(c(n1 = "Brief Hospital Course:\nx\n"), c(jiny = "t")), OTHER_KEY))

  expect_match(text, "nelze posoudit")
  expect_match(text, "žádná zpráva korpusu nemá pár")
})

test_that("bhc_text jmenuje podíl shody, počet spárovaných zpráv a populace mimo průnik", {
  text <- bhc_text(fake_agreement())

  expect_match(text, "Počet spárovaných zpráv: 100.", fixed = TRUE)
  expect_match(text, paste0("u 40,0", NBSP, "% spárovaných zpráv (40", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, paste0("jen v BHC (11", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, paste0("jen v korpusu (6", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, paste0("u 46,0", NBSP, "% spárovaných zpráv (46", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(text, "ne správnost hranic", fixed = TRUE)
  expect_match(text, "Shoda nebo reference, která obsahuje celou naši následující sekci", fixed = TRUE)
  expect_no_match(text, "přesně")
  expect_match(text, "Neshody: Sekce nenalezena", fixed = TRUE)
  expect_no_match(text, "Neshody: Shoda")
})

test_that("bhc_text skloňuje počty: 1 zpráva, 2 zprávy, 5 zpráv", {
  for (case in list(c(1, 1, 2), c(2, 2, 5), c(5, 5, 1))) {
    text <- bhc_text(fake_agreement(
      n = c(0, case[[1]], 0, 0, 0, 0, 0, 0), jen_bhc = case[[2]], jen_korpus = case[[3]]
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
  n5 = "Brief Hospital Course:\nstory five\nDischarge Diagnosis:\nx\n",
  n6 = "Brief Hospital Course:\nstory six\n"
)
CUT_TARGETS <- c(
  n1 = "story one Discharge Diagnosis: tail one",
  n2 = "story two Discharge Diagnosis: tail two",
  n3 = "story three Past Medical History: tail three",
  n4 = "story four Attending: tail four",
  n5 = "story five",
  n6 = "story six and more"
)

cut_con <- function(texts = CUT_TEXTS, targets = CUT_TARGETS, env = parent.frame()) {
  con <- agreement_con(texts, targets, env = env)
  bhc_agreement(con, OTHER_KEY)
  con
}

test_that("bhc_cut_sections rozloží konec dřív podle části, která úsek uřízla", {
  cut <- bhc_cut_sections(cut_con(), read_section_headers(section_headers_path()))

  expect_equal(cut$sekce, c("discharge_diagnosis", "past_medical_history", "other", "bez_nasledujici"))
  expect_equal(cut$n, c(2, 1, 1, 1))
  expect_equal(
    cut$popisek[c(1, 3, 4)], c("Diagnóza při propuštění", BHC_FIELD_LABEL, "Bez následující části")
  )
  expect_equal(cut$diagnosticka, c(TRUE, FALSE, FALSE, FALSE))
  expect_equal(cut$celkem, rep(5, 4))
})

test_that("bhc_cut_sections počítá obě kategorie konce dřív a BHC bez následující části", {
  con <- cut_con()
  pairs <- DBI::dbGetQuery(con, sprintf("SELECT kategorie, count(*) AS n FROM %s GROUP BY 1", BHC_PAIRS_TABLE))
  cut <- bhc_cut_sections(con, read_section_headers(section_headers_path()))

  expect_equal(pairs$n[pairs$kategorie == "cela_sekce"], 3)
  expect_equal(pairs$n[pairs$kategorie == "konec_driv"], 2)
  expect_equal(sum(cut$n), 5)
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

  expect_match(text, "Diagnóza při propuštění (2 zprávy, 40,0 %)", fixed = TRUE)
  expect_match(text, "diagnostická sekce u 40,0 % z nich (2 zprávy)", fixed = TRUE)
})

test_that("bhc_cut_table: sekce, popisek, zprávy a podíl z těchto zpráv", {
  table <- bhc_cut_table(bhc_cut_sections(cut_con(), read_section_headers(section_headers_path())))
  expect_equal(table$sekce, c("discharge_diagnosis", "past_medical_history", "other", "bez_nasledujici"))
  expect_equal(table$n, c("2", "1", "1", "1"))
  expect_equal(table$podil, c("40,0", "20,0", "20,0", "20,0"))
})

CAUSE_STORY <- "Brief Hospital Course:\nstory\n"
CAUSE_TEXTS <- c(
  n1 = paste0(CAUSE_STORY, "\nDischarge Medications:\nasa\n"),
  n2 = paste0(CAUSE_STORY, "\nDischarge Medications:\nasa\n"),
  n3 = paste0(CAUSE_STORY, "\n\tDischarge Medications:\nasa\n"),
  n4 = paste0(CAUSE_STORY, "\n  Discharge Medications:\nasa\n"),
  n5 = paste0(CAUSE_STORY, "\nDISCHARGE MEDICATIONS:\nasa\n"),
  n6 = paste0(CAUSE_STORY, "Discharge medications:\nasa\n"),
  n7 = paste0(CAUSE_STORY, "Attending: tail\n"),
  n8 = CAUSE_STORY,
  n9 = paste0(
    CAUSE_STORY, "Discharge medications:\nasa\n\nMedications on Admission:\nx\n\n",
    "Discharge Medications:\nasa\n"
  ),
  n10 = paste0("Medications on Admission:\nx\n\n", CAUSE_STORY, "\nDischarge Medications:\nasa\n"),
  n11 = paste0(CAUSE_STORY, "Medications on Admission:\nasa\n"),
  n12 = paste0(CAUSE_STORY, "Discharge Medications:\nasa\n\nMedications on Admission:\nx\n"),
  n13 = paste0(CAUSE_STORY, "Discharge Medications:\nasa\n"),
  n14 = paste0(CAUSE_STORY, "Discharge medications:\nasa\nDischarge Medications:\nasa\n")
)
CAUSE_TARGETS <- c(
  n1 = "story Discharge Medications: asa",
  n2 = "story and more",
  n3 = "story and more",
  n4 = "story and more",
  n5 = "story and more",
  n6 = "story and more",
  n7 = "story and more",
  n8 = "story and more",
  n9 = "story Discharge medications: asa",
  n10 = "story and more",
  n11 = "story and more",
  n12 = "story and more",
  n13 = "story",
  n14 = "story and more"
)
# Pořadí řádků BHC_CAUSES; n13 je shoda, do populace konce dřív nepatří. n14 je falešný podnadpis
# bez Medications on Admission, jehož skutečná hlavička leží v těle téže části.
CAUSE_COUNTS <- c(2, 11, 9, 1, 1, 2, 7, 1, 3, 1, 1, 2, 9, 2, 1, 3, 2)

cause_con <- function(texts = CAUSE_TEXTS, targets = CAUSE_TARGETS, env = parent.frame()) {
  con <- agreement_con(texts, targets, env = env)
  bhc_agreement(con, OTHER_KEY)
  con
}

cause_lexicon <- function() read_section_headers(section_headers_path())

test_that("bhc_cut_causes rozloží konec dřív do pěti skupin vlastností v pevném pořadí", {
  causes <- bhc_cut_causes(cause_con(), cause_lexicon())

  expect_equal(names(causes), c("skupina", "klic", "vlastnost", "n", "celkem"))
  expect_equal(causes$klic, BHC_CAUSES$klic)
  expect_equal(causes$vlastnost, BHC_CAUSES$popisek)
  expect_equal(causes$skupina, BHC_CAUSES$skupina_popisek)
  expect_equal(causes$n, CAUSE_COUNTS)
  expect_equal(causes$celkem, rep(13, nrow(causes)))
})

test_that("bhc_cut_causes: každá vyčerpávající skupina dává celkem", {
  causes <- bhc_cut_causes(cause_con(), cause_lexicon())
  exhaustive <- !BHC_CAUSES$skupina %in% c("soubeh", "odsazeni")
  sums <- tapply(causes$n[exhaustive], BHC_CAUSES$skupina[exhaustive], sum)

  expect_equal(as.numeric(sums), rep(13, 4))
})

test_that("bhc_cut_causes počítá tabulátorem odsazené synonymum jako synonymum i jako odsazené", {
  texts <- c(n1 = CAUSE_TEXTS[["n3"]])
  causes <- bhc_cut_causes(cause_con(texts, c(n1 = "story and more")), cause_lexicon())
  count_of <- function(key) causes$n[BHC_CAUSES$klic == key]

  expect_equal(count_of("synonymum"), 1)
  expect_equal(count_of("jiny"), 0)
  expect_equal(count_of("odsazena"), 1)
})

test_that("bhc_cut_causes bez zprávy s koncem dřív vrátí nuly a celkem 0", {
  causes <- bhc_cut_causes(
    cause_con(c(n1 = CAUSE_TEXTS[["n13"]]), c(n1 = "story")), cause_lexicon()
  )

  expect_equal(causes$n, rep(0, nrow(BHC_CAUSES)))
  expect_equal(causes$celkem, rep(0, nrow(BHC_CAUSES)))
})

test_that("bhc_cut_causes bez tabulky párů zastaví s návodem", {
  expect_error(bhc_cut_causes(agreement_con(), cause_lexicon()), "bhc_agreement", fixed = TRUE)
})

test_that("bhc_cause_table: podíly z celkem, skupina jen na prvním řádku skupiny", {
  table <- bhc_cause_table(bhc_cut_causes(cause_con(), cause_lexicon()))

  expect_equal(names(table), c("skupina", "vlastnost", "n", "podil"))
  expect_equal(table$n, number_text(CAUSE_COUNTS))
  expect_equal(table$podil, percent_text(CAUSE_COUNTS, 13))
  first_of_group <- !duplicated(BHC_CAUSES$skupina)
  expect_equal(table$skupina[first_of_group], unique(BHC_CAUSES$skupina_popisek))
  expect_equal(table$skupina[!first_of_group], rep("", sum(!first_of_group)))
})

test_that("bhc_cause_text jmenuje počty a podíly a nic nerozhoduje", {
  text <- bhc_cause_text(bhc_cut_causes(cause_con(), cause_lexicon()))

  expect_match(text, paste0("Populace: 13", NBSP, "zpráv"), fixed = TRUE)
  expect_match(
    text, paste0("celou naši následující sekci u 15,4", NBSP, "% zpráv (2", NBSP, "zprávy)"), fixed = TRUE
  )
  expect_match(text, paste0("chybí u 69,2", NBSP, "% zpráv (9", NBSP, "zpráv)"), fixed = TRUE)
  expect_match(
    text, paste0("v následující části nebo za ní u 15,4", NBSP, "% zpráv (2", NBSP, "zprávy)"), fixed = TRUE
  )
  expect_match(
    text, paste0("Souběh: polknutá jediná sekce u 7,7", NBSP, "% zpráv (1", NBSP, "zpráva)"), fixed = TRUE
  )
  expect_match(text, paste0("podnadpis u 23,1", NBSP, "% zpráv (3", NBSP, "zprávy)"), fixed = TRUE)
  expect_match(text, paste0("Odsazená je hlavička u 15,4", NBSP, "% zpráv (2", NBSP, "zprávy)"), fixed = TRUE)
  expect_match(text, "nerozhoduje, zda je hranice parseru správná", fixed = TRUE)
})

test_that("bhc_cause_text u prázdné populace říká, že příčinu nelze posoudit", {
  causes <- bhc_cut_causes(cause_con(c(n1 = CAUSE_TEXTS[["n13"]]), c(n1 = "story")), cause_lexicon())

  expect_match(bhc_cause_text(causes), "nelze posoudit", fixed = TRUE)
})

test_that("bhc_cause_text: zpráva s delším neprefixovým targetem do populace nepatří", {
  texts <- c(n1 = "Brief Hospital Course:\nmiddle part\n")
  causes <- bhc_cut_causes(cause_con(texts, c(n1 = "start middle part")), cause_lexicon())

  expect_equal(causes$celkem[[1]], 0)
  expect_match(bhc_cause_text(causes), "nemá náš úsek jako vlastní prefix targetu", fixed = TRUE)
})

test_that("bhc_cut_causes: opakování najde skutečnou hlavičku v těle téže části", {
  texts <- c(n1 = CAUSE_TEXTS[["n14"]])
  causes <- bhc_cut_causes(cause_con(texts, c(n1 = "story and more")), cause_lexicon())
  count_of <- function(key) causes$n[causes$klic == key]

  expect_equal(count_of("opakuje"), 1)
  expect_equal(count_of("neopakuje"), 0)
})

SOUBEH_BASE <- "Brief Hospital Course:\nstory\nDischarge Medications:\nasa\n"
SOUBEH_MOA <- "\nMedications on Admission:\nx\n\nDischarge Medications:\nasa\n"

# Obě sady mají tytéž marginály (reference 1/1, Medications on Admission chybí 1 a za 1,
# opakování 1 a 1), ale jinak je zkombinované.
test_that("bhc_cut_causes: souběh rozliší sady se stejnými marginálami", {
  one_sekce <- c(n1 = SOUBEH_BASE, n2 = paste0(SOUBEH_BASE, SOUBEH_MOA))
  swapped <- c(n1 = paste0(SOUBEH_BASE, SOUBEH_MOA), n2 = SOUBEH_BASE)
  complete <- "story Discharge Medications: asa"
  more <- "story and more"
  first <- bhc_cut_causes(cause_con(one_sekce, c(n1 = complete, n2 = more)), cause_lexicon())
  second <- bhc_cut_causes(cause_con(swapped, c(n1 = complete, n2 = more)), cause_lexicon())
  marginals <- !BHC_CAUSES$skupina %in% c("soubeh", "odsazeni")
  joint <- BHC_CAUSES$skupina == "soubeh"

  expect_equal(first$n[marginals], second$n[marginals])
  expect_equal(first$n[joint], c(1, 1))
  expect_equal(second$n[joint], c(0, 1))
})

test_that("bhc_agreement po zprávách: cela_sekce i s pokračováním targetu za následující sekcí", {
  con <- agreement_con()
  bhc_agreement(con, OTHER_KEY)
  pairs <- DBI::dbGetQuery(con, sprintf("SELECT note_id, kategorie FROM %s", BHC_PAIRS_TABLE))
  category_of <- function(id) pairs$kategorie[pairs$note_id == id]

  expected <- c(
    n2 = "cela_sekce", n13 = "konec_driv", n14 = "cela_sekce", n15 = "cela_sekce",
    n16 = "konec_driv", n17 = "konec_driv", n18 = "cela_sekce"
  )
  for (id in names(expected)) {
    expect_equal(category_of(id), expected[[id]], info = id)
  }
})

test_that("tbl-bhc-pricina: export projde i u prázdné tabulky a šířky sedí na čtyři sloupce", {
  layout <- BHC_LAYOUTS[["tbl-bhc-pricina"]]
  empty <- bhc_cut_causes(cause_con(c(n1 = CAUSE_TEXTS[["n13"]]), c(n1 = "story")), cause_lexicon())
  dir <- withr::local_tempdir()

  expect_length(layout$widths, 4)
  expect_no_error(export_table(
    bhc_cause_table(empty), col.names = layout$col.names, align = layout$align,
    widths = layout$widths, label = "tbl-bhc-pricina", dir = dir
  ))
  expect_true(file.exists(file.path(dir, "tbl-bhc-pricina.tex")))
})
