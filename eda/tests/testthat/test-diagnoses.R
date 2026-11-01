LONG <- strrep("x", SHORT_LIMIT)

bulk <- function(code, version, n, tag = code) {
  hadm <- paste0(tag, "-v", version, "-", seq_len(n))
  data.frame(hadm_id = hadm, icd_code = code, icd_version = version, stringsAsFactors = FALSE)
}

notes_for <- function(codes) {
  hadm <- unique(codes$hadm_id)
  data.frame(
    note_id = paste0("n-", hadm),
    subject_id = paste0("s-", hadm),
    hadm_id = hadm,
    text = paste0(hadm, LONG),
    stringsAsFactors = FALSE
  )
}

diag_con <- function(codes, notes = notes_for(codes), titles = NULL, env = parent.frame()) {
  mimic_con(discharge = notes, diagnoses_icd = codes, d_icd_diagnoses = titles, env = env)
}

candidate <- function(label, icd9, icd10, charlson = character()) {
  list(label_cs = label, icd9_prefixes = icd9, icd10_prefixes = icd10, charlson = charlson)
}

POOL <- list(
  hf = candidate("Srdeční selhání", "428", "I50", "chf"),
  dm = candidate("Diabetes", "250", "E11", c("diab", "diabwc")),
  ckd = candidate("Ledviny", "585", "N18", "rend"),
  pn = candidate("Pneumonie", "486", "J18")
)

multi <- function(hadm, codes, versions = "10") {
  data.frame(
    hadm_id = hadm, icd_code = codes, icd_version = versions, stringsAsFactors = FALSE
  )
}

test_that("do populace patří jen zprávy vstupního korpusu a skupina se počítá jednou", {
  codes <- rbind(
    multi("h1", "I500"),
    multi("h3", "I500"),
    multi("h4", "I500"),
    multi("h5", c("I500", "I501", "I502"))
  )
  notes <- notes_for(codes)
  notes$text[notes$hadm_id == "h3"] <- "krátká"
  notes$text[notes$hadm_id == "h4"] <- notes$text[notes$hadm_id == "h1"]
  extra <- data.frame(
    note_id = "n-bez-kodu", subject_id = "s-bez", hadm_id = "h2",
    text = paste0("bez-kodu", LONG), stringsAsFactors = FALSE
  )
  con <- diag_con(codes, rbind(notes, extra))

  prevalence <- candidate_prevalence(con, POOL["hf"])

  expect_equal(era_summary(con)$notes, 2)
  expect_equal(prevalence$n, 2)
  expect_equal(prevalence$celkem, 2)
})

test_that("kód se shoduje s předponou jen ve své verzi, bez ohledu na písmena a mezery", {
  codes <- rbind(
    multi("a", "4280", "9"),
    multi("b", "I500", "10"),
    multi("c", "I50", "9"),
    multi("d", "4280", "10"),
    multi("e", "i500 ", "10"),
    multi("f", "I500", "sentinel-version-1"),
    multi("g", "I500", NA_character_)
  )
  con <- diag_con(codes)

  out <- candidate_prevalence(con, POOL["hf"])

  expect_equal(out$n, 3)
  expect_equal(out$n9, 1)
  expect_equal(out$n10, 2)
})

test_that("zpráva s kódy obou verzí je v obou jmenovatelích a v řádku obě", {
  codes <- rbind(
    multi("a", "4280", "9"),
    multi("b", "I500", "10"),
    multi("c", c("4280", "I500"), c("9", "10")),
    multi("d", "I500", "x")
  )
  con <- diag_con(codes)

  era <- era_summary(con)
  out <- candidate_prevalence(con, POOL["hf"])

  expect_equal(era$notes, 4)
  expect_equal(c(era$only9, era$only10, era$both, era$neither), c(1, 1, 1, 1))
  expect_equal(era$only9 + era$only10 + era$both + era$neither, era$notes)
  expect_equal(c(era$n9, era$n10), c(2, 2))
  expect_equal(c(out$celkem9, out$celkem10, out$celkem), c(2, 2, 4))
  expect_equal(c(out$n9, out$n10, out$n), c(2, 2, 3))
})

test_that("era_summary spočítá zprávy s kódem kapitoly R a 780–799", {
  codes <- rbind(
    multi("a", "R079", "10"),
    multi("b", "7806", "9"),
    multi("c", "7999", "9"),
    multi("d", "R079", "9"),
    multi("e", "I10", "10"),
    multi("f", "401", "10")
  )
  con <- diag_con(codes)

  expect_equal(era_summary(con)$symptom, 3)
})

test_that("charlson_counts zařadí kódy obou verzí a započte kategorii jednou na zprávu", {
  codes <- rbind(
    multi("a", c("I500", "4280"), c("10", "9")),
    multi("b", "C787", "10"),
    multi("c", "4280", "9")
  )
  con <- diag_con(codes)

  out <- charlson_counts(con)
  count <- function(name) out$n[out$kategorie == name]

  expect_equal(out$kategorie, names(CHARLSON_LABELS))
  expect_equal(count("chf"), 2)
  expect_equal(count("metacanc"), 1)
  expect_equal(count("canc"), 0)
})

test_that("charlson_counts nad prázdným korpusem vrátí 17 kategorií s nulou", {
  con <- mimic_con()

  out <- charlson_counts(con)

  expect_equal(nrow(out), 17L)
  expect_equal(length(CHARLSON_LABELS), 17L)
  expect_true(all(out$n == 0))
})

ranking_fixture <- function(env = parent.frame()) {
  codes <- rbind(
    bulk("I10", "10", 30),
    bulk("E119", "10", 12),
    bulk("E1165", "10", 10),
    bulk("K219", "10", 10),
    bulk("K210", "10", 10),
    bulk("Z99", "10", 3),
    bulk("4019", "9", 5)
  )
  titles <- data.frame(
    icd_code = c("I10", "E119", "E1165", "K219", "K210", "Z99", "4019"),
    icd_version = c("10", "10", "10", "10", "10", "10", "9"),
    long_title = c(
      "Hypertenze", "Diabetes bez komplikací", "Diabetes s hyperglykemií",
      "Reflux bez ezofagitidy", "Reflux s ezofagitidou", "Závislost na přístroji",
      "Hypertenze v9"
    ),
    stringsAsFactors = FALSE
  )
  diag_con(codes, titles = titles, env = env)
}

test_that("žebříček seskupí na tři znaky, seřadí podle zpráv a kódu a omezí na top", {
  con <- ranking_fixture()

  out <- code_ranking(con, 10L, top = 2)

  expect_equal(out$kod, c("I10", "E11"))
  expect_equal(out$n, c(30, 22))
  expect_equal(code_ranking(con, 10L)$kod, c("I10", "E11", "K21"))
})

test_that("žebříček má jmenovatele ze zpráv s kódem své verze", {
  con <- ranking_fixture()

  out <- code_ranking(con, 10L)

  expect_equal(out$podil, c(30, 22, 20) / 75)
})

test_that("název skupiny je název nejčastějšího plného kódu, při shodě nejmenšího", {
  con <- ranking_fixture()

  out <- code_ranking(con, 10L)

  expect_equal(out$nazev, c("Hypertenze", "Diabetes bez komplikací", "Reflux s ezofagitidou"))
  expect_named(out, c("kod", "nazev", "n", "podil"))
})

test_that("filtr předpon vybere jen kapitolu R, resp. 780–799", {
  codes <- rbind(
    bulk("R079", "10", 12),
    bulk("I10", "10", 20),
    bulk("7806", "9", 12),
    bulk("7999", "9", 11),
    bulk("4019", "9", 20)
  )
  con <- diag_con(codes)

  expect_equal(code_ranking(con, 10L, SYMPTOM_PREFIXES$icd10)$kod, "R07")
  expect_equal(code_ranking(con, 9L, SYMPTOM_PREFIXES$icd9)$kod, c("780", "799"))
})

test_that("práh popisku vynechá devět zpráv, pustí deset a neomezuje doplněk", {
  codes <- rbind(bulk("I10", "10", 30), bulk("Z99", "10", 9), bulk("E119", "10", 10))
  titles <- data.frame(
    icd_code = c("I10", "Z99", "E119"), icd_version = "10",
    long_title = c("Hypertenze", "Závislost na přístroji", "Diabetes"),
    stringsAsFactors = FALSE
  )
  con <- diag_con(codes, titles = titles)

  out <- code_ranking(con, 10L)

  expect_equal(out$kod, c("I10", "E11"))
  expect_equal(out$n, c(30, 10))
  expect_false(any(grepl("Z99|Závislost", unlist(out))))

  tight <- diag_con(rbind(bulk("I10", "10", 40), bulk("Z99", "10", 2)))
  tight_out <- code_ranking(tight, 10L)
  expect_equal(tight_out$kod, "I10")
  expect_equal(tight_out$n, 40)
  expect_equal(tight_out$podil, 40 / 42)
})

test_that("žebříček prázdného korpusu je prázdná tabulka", {
  out <- code_ranking(mimic_con(), 10L)

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kod", "nazev", "n", "podil"))
})

test_that("kandidát bez zpráv má nulu a malý kandidát zachová počet", {
  codes <- rbind(bulk("I500", "10", 30), bulk("N189", "10", 3), bulk("I10", "10", 40))
  con <- diag_con(codes)

  raw <- candidate_prevalence(con, POOL)
  row <- function(frame, key) frame[frame$kandidat == key, ]

  expect_equal(raw$kandidat, names(POOL))
  expect_equal(row(raw, "pn")$n, 0)
  expect_equal(row(raw, "hf")$n, 30)
  expect_equal(row(raw, "ckd")$n, 3)
  expect_equal(row(raw, "hf")$celkem, 73)
})

group <- function(id, n, codes, version = "10") {
  grid <- expand.grid(k = seq_len(n), code = codes, stringsAsFactors = FALSE)
  data.frame(
    hadm_id = sprintf("g%s-%d", id, grid$k), icd_code = grid$code, icd_version = version,
    stringsAsFactors = FALSE
  )
}

cooccurrence_fixture <- function(env = parent.frame()) {
  codes <- rbind(
    group(1, 25, c("I50", "E11")),
    group(2, 15, "I50"),
    group(3, 30, "E11"),
    group(4, 3, c("I50", "N18")),
    group(5, 9, "N18"),
    group(6, 10, c("I50", "J18"))
  )
  diag_con(codes, env = env)
}

test_that("souběh počítá P(sloupec | řádek) ze správných počtů a nemá diagonálu", {
  con <- cooccurrence_fixture()

  out <- cooccurrence(con, POOL)
  cell <- function(row, column) out[out$radek == row & out$sloupec == column, ]

  expect_false(any(out$radek == out$sloupec))
  expect_equal(nrow(out), length(POOL) * (length(POOL) - 1))
  expect_equal(c(cell("hf", "dm")$n_ij, cell("hf", "dm")$n_i), c(25, 53))
  expect_equal(c(cell("dm", "hf")$n_ij, cell("dm", "hf")$n_i), c(25, 55))
  expect_equal(cell("ckd", "dm")$n_ij, 0)
})

test_that("souběh zachová malý počet a podíl ze jmenovatele řádku", {
  co <- data.frame(radek = "a", sloupec = "b", n_ij = 3, n_i = 50, n_j = 40)

  out <- cooccurrence_cells(co)

  expect_equal(out$n_ij, 3)
  expect_equal(out$podil, 0.06)
})

test_that("pásmo prevalence rozliší pod, v pásmu, nad a neznámé", {
  out <- band_verdict(c(0.04, 0.05, 0.3, 0.6, 0.61, NA))

  expect_equal(out, c("pod", "v", "v", "v", "nad", "nelze"))
})

published_prevalence <- function(n, total = 1000) {
  keys <- names(n)
  total <- rep_len(total, length(n))
  data.frame(
    kandidat = keys, n = n, n9 = n, n10 = n, celkem = total,
    celkem9 = total, celkem10 = total, stringsAsFactors = FALSE
  )
}

test_that("předvýběr pustí kandidáty v pásmu a ostatní jmenuje s důvodem", {
  published <- published_prevalence(c(hf = 300, dm = 20, ckd = 700, pn = 0))

  out <- preselection(published)

  expect_equal(out$kind, "vyber")
  expect_equal(out$in_band, "hf")
  expect_equal(out$below, c("dm", "pn"))
  expect_equal(out$above, "ckd")
})

test_that("nepokryté Charlsonovy kategorie v pásmu se určí podle pole charlson", {
  counts <- data.frame(
    kategorie = c("chf", "diab", "pvd", "dementia", "aids", "cpd"),
    n = c(300, 200, 100, 30, 400, 0),
    stringsAsFactors = FALSE
  )

  out <- uncovered_charlson(counts, POOL, total = 1000)

  expect_equal(out$kind, "audit")
  expect_equal(out$categories, c("pvd", "aids"))
  expect_equal(uncovered_charlson(counts[1:2, ], POOL, total = 1000)$categories, character())
})

test_that("dvojice s P(j | i) ≥ 0,5 se jmenují", {
  cells <- data.frame(
    radek = c("hf", "dm", "ckd", "pn"),
    sloupec = c("dm", "hf", "hf", "hf"),
    n_ij = c(30, 10, 0, 12),
    n_i = c(60, 60, 5, 100),
    podil = c(0.5, 10 / 60, 0, 0.12),
    stringsAsFactors = FALSE
  )

  out <- redundant_pairs(cells)

  expect_equal(out$kind, "dvojice")
  expect_equal(out$pairs$radek, "hf")
  expect_equal(out$pairs$sloupec, "dm")
})

test_that("éry se liší, když je prevalence v jedné víc než 1,5krát větší", {
  published <- data.frame(
    kandidat = c("hf", "dm", "ckd", "pn", "ami"),
    n = 100,
    n9 = c(30, 15, 0, 5, 10),
    n10 = c(10, 10, 20, 10, 10),
    celkem = 1000,
    celkem9 = 100,
    celkem10 = 100,
    stringsAsFactors = FALSE
  )

  out <- era_gaps(published)

  expect_equal(out$kind, "rozdily")
  expect_equal(out$gaps$kandidat, c("hf", "ckd", "pn"))
})

test_that("příznaky zachovají i čtyři zprávy ze sta", {
  out <- symptom_share(list(notes = 100, symptom = 4))

  expect_equal(out$kind, "share")
  expect_equal(out$n, 4)
  expect_equal(out$share, 0.04)
})

test_that("tabulka éry uvede nulu příznaků i pro prázdný korpus", {
  era <- list(notes = 0, only9 = 0, only10 = 0, both = 0, neither = 0, symptom = 0)

  out <- era_table(era)
  row <- out[out$ukazatel == "Zprávy s kódem kapitoly R (780–799)", ]

  expect_equal(row$hodnota, "0")
  expect_equal(row$podil, "–")
})

test_that("tabulka éry uvede čtyři zprávy s příznaky ze sta", {
  era <- list(notes = 100, only9 = 0, only10 = 100, both = 0, neither = 0, symptom = 4)

  out <- era_table(era)
  row <- out[out$ukazatel == "Zprávy s kódem kapitoly R (780–799)", ]

  expect_equal(row$hodnota, "4")
  expect_equal(row$podil, "4,0")
})

test_that("prázdný korpus vede do větví nelze", {
  con <- mimic_con()
  pool <- read_candidates(candidates_path())
  era <- era_summary(con)
  published <- candidate_prevalence(con, pool)
  cells <- cooccurrence_cells(cooccurrence(con, pool))
  counts <- charlson_counts(con)

  expect_equal(era$notes, 0)
  expect_equal(preselection(published)$kind, "nelze")
  expect_equal(uncovered_charlson(counts, pool, era$notes)$kind, "nelze")
  expect_equal(redundant_pairs(cells)$kind, "nelze")
  expect_equal(era_gaps(published)$kind, "nelze")
  expect_equal(symptom_share(era)$kind, "nelze")
})

test_that("read_candidates načte reálný config a každý kandidát je úplný", {
  pool <- read_candidates(candidates_path())

  expect_length(pool, 10L)
  for (entry in pool) {
    expect_true(nzchar(entry$label_cs))
    expect_true(length(entry$icd9_prefixes) > 0 || length(entry$icd10_prefixes) > 0)
    expect_true(all(entry$charlson %in% names(CHARLSON_LABELS)))
  }
  expect_equal(pool$diabetes$charlson, c("diab", "diabwc"))
  expect_equal(pool$sepsis$icd10_prefixes, c("A40", "A41", "R652"))
})

test_that("Charlson zachová malé počty i jejich podíly", {
  counts <- data.frame(kategorie = c("A", "B"), n = c(3, 12), stringsAsFactors = FALSE)

  published <- charlson_labelled(counts)

  expect_equal(published$n, c(3, 12))
  expect_equal(charlson_table(published, 100)$podil, c("3,0", "12,0"))
})

test_that("tabulka kandidátů uvádí české popisky a pásmo", {
  published <- published_prevalence(c(hf = 300, dm = 3))

  out <- candidate_table(published, POOL)

  expect_equal(out$kandidat, c("Srdeční selhání", "Diabetes"))
  expect_equal(out$pasmo, c("v pásmu", "pod pásmem"))
  expect_equal(out$n, c("300", "3"))
})

test_that("kód v izolovaném prostředí nepotřebuje demographics.R", {
  env <- new.env(parent = baseenv())
  sys.source("../../R/dua.R", envir = env)
  sys.source("../../R/diagnoses.R", envir = env)

  expect_equal(env$UNKNOWN_LABEL, "neuvedeno")
  expect_true(env$HIDDEN_TEXT == "potlačeno")
})

test_that("souběh vrací i počet zpráv kandidáta ve sloupci", {
  out <- cooccurrence(cooccurrence_fixture(), POOL)
  cell <- function(row, column) out[out$radek == row & out$sloupec == column, ]

  expect_equal(c(cell("hf", "dm")$n_j, cell("dm", "hf")$n_j), c(55, 53))
})

test_that("souběh zachová oba směry i malý doplněk", {
  co <- data.frame(
    radek = c("a", "b"), sloupec = c("b", "a"),
    n_ij = 25, n_i = c(100, 30), n_j = c(30, 100), stringsAsFactors = FALSE
  )

  out <- cooccurrence_cells(co)

  expect_equal(out$n_ij, c(25, 25))
  expect_equal(out$podil, c(0.25, 25 / 30))
})

test_that("název skupiny nepochází z kódu pod prahem, jinak je neuvedeno", {
  codes <- rbind(
    bulk("K210", "10", 3), bulk("K211", "10", 3), bulk("K212", "10", 3), bulk("K213", "10", 3),
    bulk("I10", "10", 20)
  )
  titles <- data.frame(
    icd_code = c("K210", "K211", "K212", "K213", "I10"), icd_version = "10",
    long_title = c("Název A", "Název B", "Název C", "Název D", "Hypertenze"),
    stringsAsFactors = FALSE
  )
  con <- diag_con(codes, titles = titles)

  out <- code_ranking(con, 10L)

  expect_equal(out$kod, c("I10", "K21"))
  expect_equal(out$nazev, c("Hypertenze", "neuvedeno"))
  expect_false(any(grepl("Název", unlist(out))))
})

test_that("E-kódy ICD-9 se seskupují na čtyři znaky", {
  codes <- rbind(bulk("E8497", "9", 12), bulk("4019", "9", 20))
  con <- diag_con(codes)

  expect_equal(code_ranking(con, 9L)$kod, c("401", "E849"))
})

test_that("prázdný seznam předpon u žebříčku bere všechny kódy", {
  con <- diag_con(rbind(bulk("I10", "10", 20), bulk("E119", "10", 12)))

  expect_equal(code_ranking(con, 10L, character())$kod, c("I10", "E11"))
})

test_that("candidate_labels vrátí české popisky bez jmen v pořadí klíčů", {
  pool <- list(a = list(label_cs = "Áčko"), b = list(label_cs = "Béčko"))

  expect_equal(candidate_labels(c("b", "a"), pool), c("Béčko", "Áčko"))
  expect_equal(candidate_labels(character(), pool), character())
})

test_that("config_path skládá cesty ke konfiguraci a obě cesty existují", {
  expect_equal(basename(config_path("x.json")), "x.json")
  expect_true(file.exists(candidates_path()))
  expect_true(file.exists(text_mentions_path()))
})

test_that("souběh při nulovém počtu kandidáta řádku nemá podíl", {
  cells <- data.frame(radek = "a", sloupec = "b", n_ij = 0, n_i = 0, n_j = 3)

  out <- cooccurrence_cells(cells)

  expect_equal(out$n_ij, 0)
  expect_true(is.na(out$podil))
})

test_that("předvýběr z prázdné tabulky nelze určit", {
  expect_equal(preselection(published_prevalence(setNames(numeric(), character())))$kind, "nelze")
})

test_that("audit prázdné tabulky Charlsonových kategorií nelze provést", {
  counts <- data.frame(kategorie = character(), n = numeric())

  expect_equal(uncovered_charlson(counts, POOL, 100)$kind, "nelze")
})

test_that("rozdíly ér bez zpráv ICD-9 nelze posoudit", {
  prevalence <- published_prevalence(c(a = 3))
  prevalence$n9 <- 0
  prevalence$n10 <- 3
  prevalence$celkem9 <- 0
  prevalence$celkem10 <- 5

  expect_equal(era_gaps(prevalence)$kind, "nelze")
})
