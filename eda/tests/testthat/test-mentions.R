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

test_that("pool pokrývá právě SELECTION_SIZE kapitol", {
  pool <- read_candidates(candidates_path())

  expect_equal(length(unique(vapply(pool, function(entry) entry$kapitola, character(1)))), SELECTION_SIZE)
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

LONG <- strrep("x", SHORT_LIMIT)

CASES <- list(
  dm = list(
    label_cs = "Diabetes", icd9_prefixes = character(), icd10_prefixes = "E11",
    charlson = character(), kapitola = "E",
    keywords_en = c("diabetes", "diabetic"), acronyms_en = c("DM", "DM2")
  ),
  pn = list(
    label_cs = "Pneumonie", icd9_prefixes = character(), icd10_prefixes = "J18",
    charlson = character(), kapitola = "J",
    keywords_en = "pneumonia", acronyms_en = character()
  ),
  ami = list(
    label_cs = "Infarkt", icd9_prefixes = character(), icd10_prefixes = "I21",
    charlson = character(), kapitola = "I",
    keywords_en = "myocardial infarction", acronyms_en = c("MI", "CHF")
  ),
  af = list(
    label_cs = "Fibrilace", icd9_prefixes = character(), icd10_prefixes = "I48",
    charlson = character(), kapitola = "I",
    keywords_en = "atrial fibrillation", acronyms_en = c("AF", "HF")
  )
)

MENTIONS <- list(
  negation = list(cues = c("no", "denies", "family history", "not"), window_chars = 40L),
  symptoms = list(keywords_en = c("fever", "chills", "cough", "chest pain"))
)

mention_con <- function(texts, codes = rep("Z99", length(texts)), env = parent.frame()) {
  hadm <- paste0("h", seq_along(texts))
  notes <- text_notes(paste0(texts, " ", hadm, " ", LONG))
  notes$hadm_id <- hadm
  rows <- data.frame(
    hadm_id = hadm, icd_code = codes, icd_version = "10", stringsAsFactors = FALSE
  )
  mimic_con(discharge = notes, diagnoses_icd = rows, env = env)
}

symptom_row <- function(texts, symptom) {
  con <- mention_con(texts)
  out <- symptom_mentions(con, CASES, MENTIONS)
  out[out$priznak == symptom, ]
}

test_that("re2_escape zneškodní speciální znaky a nerozbije pomlčku ani lomítko", {
  con <- mimic_con()
  specials <- strsplit(".+*?()[]{}|^$\\", "")[[1]]

  for (special in specials) {
    pattern <- paste0("x", re2_escape(special), "y")
    hit <- DBI::dbGetQuery(con, sprintf(
      "SELECT regexp_matches(%s, %s) AS a, regexp_matches('xzy', %s) AS b",
      DBI::dbQuoteString(con, paste0("x", special, "y")),
      DBI::dbQuoteString(con, pattern), DBI::dbQuoteString(con, pattern)
    ))
    expect_true(hit$a, info = special)
    expect_false(hit$b, info = special)
  }
  expect_equal(re2_escape(c("a-fib", "r/o")), c("a-fib", "r/o"))
})

test_that("mention_pattern bez zkratek nemá prázdnou alternativu a nechytá nesouvisející text", {
  con <- mimic_con()
  pattern <- mention_pattern("pneumonia", character())
  matched <- function(text) {
    DBI::dbGetQuery(con, sprintf(
      "SELECT regexp_matches(%s, %s) AS m", DBI::dbQuoteString(con, text),
      DBI::dbQuoteString(con, pattern)
    ))$m
  }

  expect_false(matched("nothing to see here"))
  expect_true(matched("Left PNEUMONIA noted"))
})

test_that("zkratky se hledají s rozlišením velikosti písmen a fráze bez něj", {
  texts <- c(
    "History of DM2 and CHF.", "mi casa, af, hf, dm", "Looking AFTER the MIXED case",
    "Diabetes today", "DIABETES today", "Prior MI noted", "Known AF, no HF."
  )
  con <- mention_con(texts, c("E11", "J18", "I21", "Z99", "Z99", "Z99", "I48"))

  out <- mention_agreement(con, CASES, MENTIONS)

  expect_equal(out$zminka[out$kandidat == "dm"], 3)
  expect_equal(out$zminka[out$kandidat == "ami"], 2)
  expect_equal(out$zminka[out$kandidat == "pn"], 0)
  expect_equal(out$zminka[out$kandidat == "af"], 1)
  expect_equal(out$icd_zminka[out$kandidat == "af"], 1)
})

test_that("mezera ve frázi se shoduje s libovolným bílým znakem včetně zalomení řádku", {
  texts <- c("Prior myocardial\ninfarction.", "Old myocardial  infarction.", "Myocardial infarction.")
  con <- mention_con(texts)

  out <- mention_agreement(con, CASES, MENTIONS)
  fever <- symptom_row(c("Has chest\npain today."), "chest pain")

  expect_equal(out$zminka[out$kandidat == "ami"], 3)
  expect_equal(fever$zminka, 1)
})

test_that("zmínka až na konci velmi dlouhé zprávy se najde", {
  con <- mention_con(paste0(strrep("lorem ipsum ", 1000), "pneumonia"), "J18")

  out <- mention_agreement(con, CASES, MENTIONS)

  expect_gt(DBI::dbGetQuery(con, "SELECT max(length(text)) AS n FROM discharge")$n, 10000)
  expect_equal(out$zminka[out$kandidat == "pn"], 1)
  expect_equal(out$icd_zminka[out$kandidat == "pn"], 1)
})

test_that("negovaná zmínka je zmínka bez tvrzení a kladná zmínka vedle ní tvrzení dá", {
  con <- mention_con(c(
    "No pneumonia.", "No pneumonia on CXR; treated for pneumonia.", "Pneumonia on CXR."
  ))

  out <- mention_agreement(con, CASES, MENTIONS)
  row <- out[out$kandidat == "pn", ]

  expect_equal(row$zminka, 3)
  expect_equal(row$tvrzeni, 2)
})

test_that("negace pokrývá výčet v okně, ale ne tečku ani víc než 40 znaků", {
  filler <- strrep("word ", 12)
  texts <- c(
    "Patient denies fever, chills or cough.",
    "Denies chills. Fever overnight.",
    paste0("No ", filler, "fever"),
    "Family history of diabetes."
  )
  con <- mention_con(texts)

  symptoms <- symptom_mentions(con, CASES, MENTIONS)
  fever <- symptoms[symptoms$priznak == "fever", ]
  cough <- symptoms[symptoms$priznak == "cough", ]
  diabetes <- mention_agreement(con, CASES, MENTIONS)
  diabetes <- diabetes[diabetes$kandidat == "dm", ]

  expect_equal(c(cough$zminka, cough$tvrzeni), c(1, 0))
  expect_equal(c(fever$zminka, fever$tvrzeni), c(3, 2))
  expect_equal(c(diabetes$zminka, diabetes$tvrzeni), c(1, 0))
})

test_that("do populace patří jen zprávy vstupního korpusu a icd se rovná prevalenci", {
  notes <- data.frame(
    note_id = c("n1", "n2", "n3", "n4", "n5"),
    subject_id = c("s1", "s2", "s3", "s4", "s5"),
    hadm_id = c("h1", "h2", "h3", "h4", "h5"),
    text = c(
      paste0("Pneumonia seen. ", LONG), paste0("Pneumonia elsewhere. ", LONG), "Pneumonia",
      paste0("Pneumonia seen. ", LONG), paste0("Pneumonia, no code. ", LONG)
    ),
    stringsAsFactors = FALSE
  )
  codes <- data.frame(
    hadm_id = c("h1", "h2", "h3", "h4"), icd_code = c("J18", "Z99", "J18", "J18"),
    icd_version = "10", stringsAsFactors = FALSE
  )
  con <- mimic_con(discharge = notes, diagnoses_icd = codes)

  out <- mention_agreement(con, CASES, MENTIONS)
  row <- out[out$kandidat == "pn", ]
  prevalence <- candidate_prevalence(con, CASES)

  expect_equal(c(row$celkem, row$zminka, row$icd_zminka), c(2, 2, 1))
  expect_equal(row$icd, prevalence$n[prevalence$kandidat == "pn"])
  expect_equal(row$icd, 1)
})

test_that("zpráva se zmínkou bez kódu kandidáta zvedne zminka, ale ne icd_zminka", {
  con <- mention_con(c("Diabetes noted.", "Diabetes noted twice.", "Other."), c("E11", "Z99", "E11"))

  row <- mention_agreement(con, CASES, MENTIONS)
  row <- row[row$kandidat == "dm", ]

  expect_equal(c(row$icd, row$zminka, row$icd_zminka, row$celkem), c(2, 2, 1, 3))
})

test_that("počty tvrzení kandidáta se dělí na zprávy s kódem a bez něj", {
  con <- mention_con(c("No diabetes.", "Diabetes.", "Diabetes DM2."), c("E11", "E11", "Z99"))

  row <- mention_agreement(con, CASES, MENTIONS)
  row <- row[row$kandidat == "dm", ]

  expect_equal(
    c(row$zminka, row$tvrzeni, row$icd_zminka, row$icd_tvrzeni), c(3, 2, 2, 1)
  )
})

test_that("symptom_mentions spočítá zmínky a tvrzení lexikonu a celkem", {
  texts <- c("Fever and cough.", "No fever.", "Denies fever, cough.", "Nothing.")
  con <- mention_con(texts)

  out <- symptom_mentions(con, CASES, MENTIONS)

  expect_equal(out$priznak, MENTIONS$symptoms$keywords_en)
  expect_equal(out$zminka[out$priznak == "fever"], 3)
  expect_equal(out$tvrzeni[out$priznak == "fever"], 1)
  expect_equal(out$zminka[out$priznak == "chest pain"], 0)
  expect_equal(unique(out$celkem), 4)
})

test_that("mention_flags vytvoří dočasnou tabulku jednou a vrátí její název", {
  con <- mention_con(c("Diabetes."))

  first <- mention_flags(con, CASES, MENTIONS)
  DBI::dbExecute(con, sprintf("DELETE FROM %s", first))
  second <- mention_flags(con, CASES, MENTIONS)

  expect_equal(first, second)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", second))$n, 0)
})

test_that("prázdný korpus vrátí nuly a nespadne", {
  con <- mimic_con()

  agreement <- mention_agreement(con, CASES, MENTIONS)
  symptoms <- symptom_mentions(con, CASES, MENTIONS)

  expect_equal(nrow(agreement), 4L)
  expect_equal(unique(c(agreement$icd, agreement$zminka, agreement$celkem)), 0)
  expect_equal(unique(c(symptoms$zminka, symptoms$celkem)), 0)
})

agreement_row <- function(...) {
  base <- list(
    kandidat = "dm", icd = 100, zminka = 140, tvrzeni = 100, icd_zminka = 80,
    icd_tvrzeni = 60, celkem = 400
  )
  as.data.frame(modifyList(base, list(...)), stringsAsFactors = FALSE)
}

AGREEMENT_COLUMNS <- c("icd", "zminka", "tvrzeni", "icd_zminka", "icd_tvrzeni")

test_that("suppress_agreement ponechá řádek, kde nejmenší buňka má právě deset zpráv", {
  edge <- agreement_row(
    icd = 30, zminka = 40, tvrzeni = 20, icd_zminka = 20, icd_tvrzeni = 10, celkem = 60
  )

  expect_equal(suppress_agreement(agreement_row()), agreement_row())
  expect_equal(suppress_agreement(edge), edge)
  expect_equal(min(agreement_cells(edge)), 10)
})

test_that("suppress_agreement skryje řádek celý, když jediná z šesti atomických buněk má devět zpráv", {
  cases <- list(
    "ICD+ s tvrzením" = agreement_row(icd_tvrzeni = 9, zminka = 181),
    "ICD+ jen negovaná zmínka" = agreement_row(icd_tvrzeni = 71),
    "ICD+ bez zmínky" = agreement_row(icd = 89),
    "bez ICD s tvrzením" = agreement_row(tvrzeni = 69),
    "bez ICD jen negovaná zmínka" = agreement_row(zminka = 129),
    "ani ICD ani zmínka" = agreement_row(celkem = 169)
  )

  for (name in names(cases)) {
    cells <- agreement_cells(cases[[name]])
    expect_equal(sum(cells < DUA_THRESHOLD), 1L, info = name)
    out <- suppress_agreement(cases[[name]])
    expect_true(all(is.na(out[AGREEMENT_COLUMNS])), info = name)
    expect_equal(out$kandidat, "dm", info = name)
    expect_equal(out$celkem, cases[[name]]$celkem, info = name)
  }
})

test_that("suppress_agreement skryje kandidáta i s devíti zprávami ICD+ se zmínkou", {
  out <- suppress_agreement(agreement_row(icd_zminka = 9, icd_tvrzeni = 0))

  expect_true(all(is.na(out[AGREEMENT_COLUMNS])))
})

test_that("align_agreement skryje kandidáta, jehož prevalenci sešit 7 nezveřejnil", {
  agreement <- rbind(agreement_row(kandidat = "dm"), agreement_row(kandidat = "pn"))
  prevalence <- data.frame(kandidat = c("dm", "pn"), n = c(100, NA))

  out <- align_agreement(agreement, prevalence)

  expect_false(anyNA(out[out$kandidat == "dm", AGREEMENT_COLUMNS]))
  expect_true(all(is.na(out[out$kandidat == "pn", AGREEMENT_COLUMNS])))
})

test_that("suppress_symptoms skryje řádek podle dua_binary a rozdílu zmínek a tvrzení", {
  symptoms <- data.frame(
    priznak = c("fever", "cough", "nausea", "edema", "syncope"),
    zminka = c(100, 9, 395, 100, 100),
    tvrzeni = c(90, 9, 395, 91, 9),
    celkem = 400,
    stringsAsFactors = FALSE
  )

  out <- suppress_symptoms(symptoms)

  expect_equal(out$zminka, c(100, NA, NA, NA, NA))
  expect_equal(out$tvrzeni, c(90, NA, NA, NA, NA))
  expect_equal(out$priznak, symptoms$priznak)
})

RUBRIC_POOL <- list(
  dm = list(label_cs = "Diabetes", kapitola = "E"),
  hf = list(label_cs = "Srdeční selhání", kapitola = "I"),
  af = list(label_cs = "Fibrilace", kapitola = "I"),
  copd = list(label_cs = "CHOPN", kapitola = "J"),
  pn = list(label_cs = "Pneumonie", kapitola = "J"),
  ckd = list(label_cs = "CKD", kapitola = "N"),
  sepsis = list(label_cs = "Sepse", kapitola = "A")
)

rubric_inputs <- function(share = c(dm = 0.7, hf = 0.6, af = 0.9, copd = 0.55, pn = 0.8, ckd = 0.5, sepsis = 0.4),
                          prevalence = c(dm = 0.3, hf = 0.2, af = 0.2, copd = 0.2, pn = 0.2, ckd = 0.2, sepsis = 0.1)) {
  keys <- names(RUBRIC_POOL)
  list(
    prevalence = data.frame(
      kandidat = keys, n = unname(prevalence[keys]) * 1000, celkem = 1000,
      stringsAsFactors = FALSE
    ),
    agreement = data.frame(
      kandidat = keys, icd = 200, zminka = 100, tvrzeni = 90,
      icd_zminka = unname(share[keys]) * 200, icd_tvrzeni = 50, celkem = 1000,
      stringsAsFactors = FALSE
    )
  )
}

run_rubric <- function(inputs) {
  rubric(inputs$prevalence, inputs$agreement, RUBRIC_POOL)
}

test_that("rubrika dá výsledek podle pásma a prahu shody a drží pořadí configu", {
  out <- run_rubric(rubric_inputs())

  expect_equal(out$kandidat, names(RUBRIC_POOL))
  expect_equal(out$kapitola, c("E", "I", "I", "J", "J", "N", "A"))
  expect_equal(out$shoda, c(0.7, 0.6, 0.9, 0.55, 0.8, 0.5, 0.4))
  expect_equal(out$pasmo, rep("v", 7L))
  expect_equal(out$verdikt_shody, c(rep("splňuje", 6L), "nesplňuje"))
  expect_equal(
    out$vysledek,
    c("prošel", "prošel", "prošel", "prošel", "prošel", "prošel", "neprošel")
  )
})

test_that("práh shody 0,5 a hranice pásma projdou včetně", {
  inputs <- rubric_inputs(
    share = c(dm = 0.5, hf = 0.5, af = 0.5, copd = 0.5, pn = 0.5, ckd = 0.49, sepsis = 0.5),
    prevalence = c(dm = 0.05, hf = 0.6, af = 0.049, copd = 0.601, pn = 0.2, ckd = 0.2, sepsis = 0.2)
  )

  out <- run_rubric(inputs)

  expect_equal(out$pasmo[1:5], c("v", "v", "pod", "nad", "v"))
  expect_equal(
    out$vysledek,
    c("prošel", "prošel", "neprošel", "neprošel", "prošel", "neprošel", "prošel")
  )
})

test_that("potlačený kandidát je nelze posoudit", {
  inputs <- rubric_inputs()
  inputs$agreement[2, AGREEMENT_COLUMNS] <- NA_real_

  out <- run_rubric(inputs)

  expect_true(is.na(out$shoda[2]))
  expect_equal(out$vysledek[2], "nelze")
  expect_equal(out$verdikt_shody[2], "nelze")
})

SHARES <- function(...) {
  base <- c(dm = 0.7, hf = 0.6, af = 0.9, copd = 0.55, pn = 0.8, ckd = 0.5, sepsis = 0.9)
  replace(base, names(list(...)), unlist(list(...)))
}

test_that("skrytá shoda dá nelze posoudit i u kandidáta mimo pásmo", {
  inputs <- rubric_inputs(prevalence = c(
    dm = 0.01, hf = 0.2, af = 0.9, copd = 0.2, pn = 0.2, ckd = 0.2, sepsis = 0.1
  ))
  inputs$agreement[1:3, AGREEMENT_COLUMNS] <- NA_real_

  out <- run_rubric(inputs)

  expect_equal(out$pasmo[1:3], c("pod", "v", "nad"))
  expect_equal(out$vysledek[1:3], rep("nelze", 3L))
})

test_that("všechny skryté shody dají kind nelze i při viditelné prevalenci mimo pásmo", {
  inputs <- rubric_inputs(prevalence = c(
    dm = 0.01, hf = 0.01, af = 0.01, copd = 0.9, pn = 0.9, ckd = 0.9, sepsis = 0.9
  ))
  inputs$agreement[AGREEMENT_COLUMNS] <- NA_real_

  expect_equal(select_five(run_rubric(inputs))$kind, "nelze")
})

test_that("select_five vybere z každé kapitoly kandidáta s nejvyšší shodou", {
  out <- select_five(run_rubric(rubric_inputs(share = SHARES())))

  expect_equal(out$kind, "vyber")
  expect_equal(out$selected, c("dm", "af", "pn", "ckd", "sepsis"))
  expect_equal(out$filled, character())
  expect_equal(out$passed, names(RUBRIC_POOL))
  expect_equal(out$failed, character())
  expect_equal(out$unknown, character())
})

test_that("při shodě rozhoduje pořadí v configu", {
  inputs <- rubric_inputs(share = SHARES(hf = 0.8, af = 0.8, copd = 0.55, pn = 0.55, ckd = 0.6))

  out <- select_five(run_rubric(inputs))

  expect_equal(out$selected, c("dm", "hf", "copd", "ckd", "sepsis"))
})

test_that("prázdnou kapitolu doplní nejlepší zbývající prošlý kandidát", {
  inputs <- rubric_inputs(share = SHARES(copd = 0.3, pn = 0.2, ckd = 0.6))

  out <- select_five(run_rubric(inputs))

  expect_equal(out$selected, c("dm", "hf", "af", "ckd", "sepsis"))
  expect_equal(out$filled, "hf")
  expect_equal(out$failed, c("copd", "pn"))
})

test_that("méně než pět prošlých znamená, že se vyberou všichni prošlí", {
  inputs <- rubric_inputs(share = SHARES(copd = 0.3, pn = 0.2, ckd = 0.1))

  out <- select_five(run_rubric(inputs))

  expect_equal(out$passed, c("dm", "hf", "af", "sepsis"))
  expect_equal(out$selected, out$passed)
  expect_lt(length(out$selected), SELECTION_SIZE)
})

test_that("když nikdo neprojde, výběr je prázdný a typ zůstává vyber", {
  inputs <- rubric_inputs(share = SHARES(
    dm = 0.1, hf = 0.1, af = 0.1, copd = 0.1, pn = 0.1, ckd = 0.1, sepsis = 0.1
  ))

  out <- select_five(run_rubric(inputs))

  expect_equal(out$kind, "vyber")
  expect_equal(out$selected, character())
  expect_equal(out$failed, names(RUBRIC_POOL))
})

test_that("všichni potlačení a prázdný korpus dají nelze", {
  inputs <- rubric_inputs()
  inputs$agreement[AGREEMENT_COLUMNS] <- NA_real_
  inputs$prevalence$n <- NA_real_
  empty <- list(prevalence = inputs$prevalence[0, ], agreement = inputs$agreement[0, ])

  expect_equal(select_five(run_rubric(inputs))$kind, "nelze")
  expect_equal(select_five(run_rubric(empty))$kind, "nelze")
})

test_that("select_five odmítne pool s víc kapitolami, než kolik má pětice míst", {
  ranked <- data.frame(
    kandidat = letters[1:6], kapitola = LETTERS[1:6], podil = 0.2, pasmo = "v",
    shoda = 0.8, verdikt_shody = "splňuje", vysledek = "prošel", stringsAsFactors = FALSE
  )

  expect_error(select_five(ranked), "kapitol")
})

test_that("rubric_table označí vybrané, doplněné, neprošlé a nelze posoudit", {
  inputs <- rubric_inputs(share = SHARES(copd = 0.3, pn = 0.2, ckd = 0.6))
  inputs$agreement[7, AGREEMENT_COLUMNS] <- NA_real_
  ranked <- run_rubric(inputs)

  out <- rubric_table(ranked, select_five(ranked), RUBRIC_POOL)

  expect_equal(out$kandidat[1:2], c("Diabetes", "Srdeční selhání"))
  expect_equal(
    out$vysledek,
    c("vybrán", "vybrán doplněním", "vybrán", "neprošel", "neprošel", "vybrán", "nelze posoudit")
  )
})

test_that("rubric_table označí prošlého kandidáta, který výběr nepojal", {
  ranked <- run_rubric(rubric_inputs(share = SHARES()))

  out <- rubric_table(ranked, select_five(ranked), RUBRIC_POOL)

  expect_equal(
    out$vysledek,
    c("vybrán", "prošel", "vybrán", "prošel", "vybrán", "vybrán", "vybrán")
  )
})
