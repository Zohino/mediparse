corpus_notes <- function(subjects, hadms = paste0("h", seq_along(subjects))) {
  data.frame(
    note_id = paste0("n", seq_along(subjects)),
    subject_id = subjects,
    hadm_id = hadms,
    text = paste0(seq_along(subjects), strrep("x", SHORT_LIMIT)),
    stringsAsFactors = FALSE
  )
}

icd_for <- function(hadms) {
  data.frame(
    hadm_id = hadms,
    icd_code = paste0("K", seq_along(hadms)),
    icd_version = "10",
    stringsAsFactors = FALSE
  )
}

patient_rows <- function(subjects, gender = "F", age = "60", year = "2150") {
  data.frame(
    subject_id = subjects,
    anchor_year_group = "2014 - 2016",
    gender = gender,
    anchor_age = age,
    anchor_year = year,
    stringsAsFactors = FALSE
  )
}

admission_rows <- function(
    hadms, subjects = hadms, admittime = "2150-03-01 10:00:00", insurance = "Medicare",
    language = "English", marital = "MARRIED", race = "WHITE") {
  data.frame(
    hadm_id = hadms,
    subject_id = subjects,
    admittime = admittime,
    insurance = insurance,
    language = language,
    marital_status = marital,
    race = race,
    stringsAsFactors = FALSE
  )
}

one_note <- function(patients = NULL, admissions = NULL, env = parent.frame()) {
  mimic_con(
    discharge = corpus_notes("1", "h1"),
    diagnoses_icd = icd_for("h1"),
    patients = patients,
    admissions = admissions,
    env = env
  )
}

demographic_row <- function(con) {
  DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", demographic_notes(con)))
}

test_that("věk při přijetí se počítá z anchor_age, anchor_year a roku přijetí", {
  con <- one_note(
    patient_rows("1", age = "60", year = "2150"),
    admission_rows("h1", "1", admittime = "2153-06-01 08:00:00")
  )
  out <- demographic_row(con)

  expect_equal(out$age, 63)
  expect_false(out$age_masked)
  expect_equal(age_band(out$age), "60–69")
})

test_that("anchor_age 91 je maskovaný, patří do pásma 90+ a nepočítá se ze součtu", {
  con <- one_note(
    patient_rows("1", age = "91", year = "2150"),
    admission_rows("h1", "1", admittime = "2152-01-01 00:00:00")
  )
  out <- demographic_row(con)

  expect_true(out$age_masked)
  expect_equal(out$age, 90)
  expect_equal(age_band(out$age), "90+")
})

test_that("nemaskovaný věk nad 89 se seřízne do pásma 90+", {
  con <- one_note(
    patient_rows("1", age = "88", year = "2150"),
    admission_rows("h1", "1", admittime = "2155-01-01 00:00:00")
  )
  out <- demographic_row(con)

  expect_false(out$age_masked)
  expect_equal(out$age, 90)
  expect_equal(age_band(out$age), "90+")
})

test_that("nečíselné vstupy věku dají neznámý věk bez chyby", {
  canary <- "sentinel-patients-anchor_age-1"
  con <- one_note(
    patient_rows("1", age = canary, year = "sentinel-year"),
    admission_rows("h1", "1", admittime = "sentinel-time")
  )
  out <- demographic_row(con)

  expect_true(is.na(out$age))
  expect_false(out$age_masked)
  expect_equal(age_band(out$age), "neuvedeno")
})

test_that("zpráva bez přijetí a bez pacienta má atributy neuvedeno", {
  no_admission <- demographic_row(one_note(patient_rows("1"), NULL))
  no_patient <- demographic_row(one_note(NULL, admission_rows("h1", "1")))

  expect_false(no_admission$has_admission)
  expect_true(no_admission$has_patient)
  expect_equal(no_admission$insurance, "neuvedeno")
  expect_equal(no_admission$race, "neuvedeno")
  expect_true(is.na(no_admission$age))
  expect_true(no_patient$has_admission)
  expect_false(no_patient$has_patient)
  expect_equal(no_patient$gender, "neuvedeno")
  expect_true(is.na(no_patient$age))
})

test_that("slovníky mapují známé hodnoty, neznámé na jiné a NULL na neuvedeno", {
  values <- list(
    insurance = c("Medicare", "Medicaid", "Private", "Other", "Tricare", NA),
    marital = c("MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "COMPLICATED", NA),
    language = c("English", "Klingon", "?", NA, "English", "English")
  )
  n <- 6
  hadms <- paste0("h", seq_len(n))
  subjects <- as.character(seq_len(n))
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    patients = patient_rows(subjects, gender = c("F", "M", "X", NA, "F", "M")),
    admissions = admission_rows(
      hadms, subjects,
      insurance = values$insurance, marital = values$marital, language = values$language
    )
  )
  out <- DBI::dbGetQuery(con, sprintf(
    "SELECT * FROM %s ORDER BY length(note_id), note_id", demographic_notes(con)
  ))

  expect_equal(out$insurance, c("Medicare", "Medicaid", "soukromé", "jiné pojištění", "jiné", "neuvedeno"))
  expect_equal(out$marital, c(
    "v manželství", "svobodní", "ovdovělí", "rozvedení", "jiné", "neuvedeno"
  ))
  expect_equal(out$language, c(
    "angličtina", "jiný jazyk", "neuvedeno", "neuvedeno", "angličtina", "angličtina"
  ))
  expect_equal(out$gender, c("ženy", "muži", "jiné", "neuvedeno", "ženy", "muži"))
  raw <- c(
    "Private", "Other", "Tricare", "MARRIED", "SINGLE", "WIDOWED", "DIVORCED",
    "COMPLICATED", "Klingon", "English", "F", "M", "X"
  )
  expect_false(any(unlist(out[c("insurance", "marital", "language", "gender")]) %in% raw))
})

test_that("rasa se sloučí podle předpony a neznámá hodnota se nikdy nezobrazí", {
  races <- c(
    "WHITE - OTHER EUROPEAN", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - PUERTO RICAN",
    "SOUTH AMERICAN", "ASIAN - CHINESE", "UNKNOWN", "UNABLE TO OBTAIN",
    "PATIENT DECLINED TO ANSWER", "AMERICAN INDIAN/ALASKA NATIVE", NA
  )
  n <- length(races)
  hadms <- paste0("h", seq_len(n))
  subjects <- as.character(seq_len(n))
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    admissions = admission_rows(hadms, subjects, race = races)
  )
  out <- DBI::dbGetQuery(con, sprintf(
    "SELECT * FROM %s ORDER BY length(note_id), note_id", demographic_notes(con)
  ))

  expect_equal(out$race, c(
    "bílá", "černá", "hispánská", "hispánská", "asijská",
    "neuvedeno", "neuvedeno", "neuvedeno", "jiná", "neuvedeno"
  ))
})

test_that("pohlaví se počítá jednou na pacienta, atributy přijetí jednou na zprávu", {
  con <- mimic_con(
    discharge = corpus_notes(c("1", "1", "1", "2"), c("h1", "h2", "h3", "h4")),
    diagnoses_icd = icd_for(c("h1", "h2", "h3", "h4")),
    patients = patient_rows(c("1", "2"), gender = c("F", "M")),
    admissions = admission_rows(
      c("h1", "h2", "h3", "h4"), c("1", "1", "1", "2"),
      insurance = c("Medicare", "Medicare", "Medicaid", "Medicare")
    )
  )
  counts <- attribute_counts(con)
  of <- function(attribute) counts[counts$atribut == attribute, ]

  expect_equal(of("Pohlaví")$kategorie, c("ženy", "muži"))
  expect_equal(of("Pohlaví")$n, c(1, 1))
  expect_equal(of("Pojištění")$kategorie, c("Medicare", "Medicaid"))
  expect_equal(of("Pojištění")$n, c(3, 1))
  expect_equal(demographic_summary(con)$patients, 2)
  expect_equal(demographic_summary(con)$notes, 4)
})

test_that("populací je vstupní korpus", {
  long <- strrep("x", SHORT_LIMIT)
  con <- mimic_con(
    discharge = data.frame(
      note_id = c("n1", "n2", "n3", "n4", "n5"),
      subject_id = c("1", "2", "3", "4", "5"),
      hadm_id = c("h1", "h2", "h3", "h4", "h5"),
      text = c(paste0("a", long), paste0("b", long), "krátká", paste0("a", long), paste0("c", long)),
      stringsAsFactors = FALSE
    ),
    diagnoses_icd = icd_for(c("h1", "h2", "h3", "h4")),
    patients = patient_rows(as.character(1:5))
  )
  out <- demographic_row(con)

  expect_equal(sort(out$note_id), c("n1", "n2"))
  expect_equal(demographic_summary(con)$notes, 2)
})

test_that("prázdný vstupní korpus dá nulové počty a kvantily NA", {
  con <- mimic_con()
  summary <- demographic_summary(con)

  expect_equal(summary$patients, 0)
  expect_equal(summary$notes, 0)
  expect_equal(summary$masked, 0)
  expect_equal(summary$no_admission, 0)
  expect_equal(summary$no_patient, 0)
  expect_true(is.na(summary$p50))
  expect_equal(nrow(attribute_counts(con)), 0)
  expect_equal(nrow(age_gender_counts(con)), 0)
})

test_that("demographic_notes tabulku vytvoří jednou a opakované volání ji vrátí", {
  con <- one_note(patient_rows("1"), admission_rows("h1", "1"))
  first <- demographic_notes(con)
  DBI::dbExecute(con, sprintf("UPDATE %s SET insurance = 'změněno'", first))

  expect_equal(demographic_notes(con), first)
  expect_equal(demographic_row(con)$insurance, "změněno")
})

test_that("souhrn uvádí kvartily seřezaného věku a počet maskovaných", {
  ages <- c("20", "40", "60", "91")
  n <- length(ages)
  hadms <- paste0("h", seq_len(n))
  subjects <- as.character(seq_len(n))
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    patients = patient_rows(subjects, age = ages),
    admissions = admission_rows(hadms, subjects, admittime = "2150-01-01 00:00:00")
  )
  summary <- demographic_summary(con)

  expect_equal(summary$p50, 50)
  expect_equal(summary$p25, 35)
  expect_equal(summary$p75, 67.5)
  expect_equal(summary$masked, 1)
})

test_that("zprávy bez přijetí a bez pacienta se v souhrnu počítají zvlášť", {
  con <- mimic_con(
    discharge = corpus_notes(c("1", "2", "3"), c("h1", "h2", "h3")),
    diagnoses_icd = icd_for(c("h1", "h2", "h3")),
    patients = patient_rows(c("1", "2")),
    admissions = admission_rows(c("h1", "h3"), c("1", "3"))
  )
  summary <- demographic_summary(con)

  expect_equal(summary$notes, 3)
  expect_equal(summary$no_admission, 1)
  expect_equal(summary$no_patient, 1)
})

test_that("pořadí kategorií je pevné a nezávisí na četnosti", {
  races <- c("ASIAN", "ASIAN", "ASIAN", "BLACK", "WHITE", "WHITE", "PATIENT DECLINED TO ANSWER", "OTHER")
  n <- length(races)
  hadms <- paste0("h", seq_len(n))
  subjects <- as.character(seq_len(n))
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    admissions = admission_rows(hadms, subjects, race = races)
  )
  counts <- attribute_counts(con)
  race <- counts[counts$atribut == "Rasa", ]

  expect_equal(race$kategorie, c("bílá", "černá", "asijská", "jiná", "neuvedeno"))
  expect_equal(unique(counts$atribut), c("Pohlaví", "Věkové pásmo", "Pojištění", "Rodinný stav", "Rasa", "Jazyk"))
})

test_that("pásma věku jsou vzestupně a neuvedeno je na konci", {
  expect_equal(AGE_BANDS, c(
    "do 29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–89", "90+", "neuvedeno"
  ))
  expect_equal(
    age_band(c(0, 29, 30, 89, 90, NA)),
    c("do 29", "do 29", "30–39", "80–89", "90+", "neuvedeno")
  )
})

test_that("age_gender_counts řadí pásma podle pevného pořadí", {
  ages <- c("85", "20", "20", "55")
  n <- length(ages)
  hadms <- paste0("h", seq_len(n))
  subjects <- as.character(seq_len(n))
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    patients = patient_rows(subjects, gender = c("F", "F", "M", "F"), age = ages),
    admissions = admission_rows(hadms, subjects, admittime = "2150-01-01 00:00:00")
  )
  out <- age_gender_counts(con)

  expect_equal(names(out), c("pohlavi", "pasmo", "n"))
  expect_equal(sum(out$n), 4)
  expect_equal(out$pasmo[out$pohlavi == "ženy"], c("do 29", "50–59", "80–89"))
})

pub <- function(atribut, kategorie, n) {
  data.frame(atribut = atribut, kategorie = kategorie, n = n, stringsAsFactors = FALSE)
}

test_that("majority_gender vrátí většinové pohlaví a podíl mezi známými", {
  out <- majority_gender(pub("Pohlaví", c("ženy", "muži", "neuvedeno"), c(30, 60, 10)))

  expect_equal(out$kind, "majority")
  expect_equal(out$label, "muži")
  expect_equal(out$n, 60)
  expect_equal(out$total, 90)
})

test_that("majority_gender dá nelze, když chybí známá pohlaví", {
  none <- pub("Pohlaví", "neuvedeno", 5)

  expect_equal(majority_gender(none)$kind, "nelze")
  expect_equal(majority_gender(none[0, ])$kind, "nelze")
  expect_equal(majority_gender(pub("Pohlaví", "ženy", 50))$kind, "nelze")
})

test_that("largest_race vybere i malou největší skupinu bez neuvedeno a jiná", {
  counts <- pub("Rasa", c("bílá", "černá", "jiná", "neuvedeno"), c(4, 2, 9, 10))
  out <- largest_race(counts, total = 25)

  expect_equal(out$kind, "group")
  expect_equal(out$label, "bílá")
  expect_equal(out$n, 4)
  expect_equal(largest_race(counts[0, ], total = 0)$kind, "nelze")
})

test_that("maskovaný věk i graf zachovají malé počty", {
  subjects <- as.character(1:5)
  hadms <- paste0("h", subjects)
  con <- mimic_con(
    discharge = corpus_notes(subjects, hadms),
    diagnoses_icd = icd_for(hadms),
    patients = patient_rows(subjects, gender = c("F", "F", "F", "M", "M"),
                           age = c("91", "91", "91", "90", "90")),
    admissions = admission_rows(hadms, subjects)
  )
  summary <- demographic_summary(con)
  counts <- attribute_counts(con)
  cells <- age_gender_counts(con)

  expect_equal(counts$n[counts$kategorie == "90+"], 5)
  expect_equal(number_text(summary$masked), "3")
  expect_equal(cells$pohlavi, c("ženy", "muži"))
  expect_equal(cells$pasmo, c("90+", "90+"))
  expect_equal(cells$n, c(3, 2))
})

test_that("characteristics_table dá jmenovatele pacienty u pohlaví a zprávy jinde", {
  published <- data.frame(
    atribut = c("Pohlaví", "Pohlaví", "Rasa", "Rasa"),
    kategorie = c("ženy", "muži", "bílá", "černá"),
    n = c(20, 80, 150, 50),
    stringsAsFactors = FALSE
  )
  out <- characteristics_table(published, list(patients = 100, notes = 200))

  expect_equal(names(out), c("atribut", "kategorie", "n", "podil"))
  expect_equal(out$atribut, c("Pohlaví", "", "Rasa", ""))
  expect_equal(out$podil, c("20,0", "80,0", "75,0", "25,0"))
  expect_equal(out$n, c("20", "80", "150", "50"))
})

test_that("summary_table uvádí medián s kvartily a spojí prázdné hodnoty pomlčkou", {
  full <- summary_table(list(
    patients = 100, notes = 1200, p25 = 40, p50 = 55.5, p75 = 70, no_admission = 3, no_patient = 0
  ))
  empty <- summary_table(list(
    patients = 0, notes = 0, p25 = NA, p50 = NA, p75 = NA, no_admission = 0, no_patient = 0
  ))

  expect_equal(full$hodnota[full$ukazatel == "Medián věku při přijetí (kvartily)"], "55,5 (40,0–70,0)")
  expect_equal(empty$hodnota[empty$ukazatel == "Medián věku při přijetí (kvartily)"], "–")
  expect_equal(full$hodnota[full$ukazatel == "Pacienti"], "100")
})

test_that("characteristics_table zvládne prázdné počty atributů", {
  empty <- pub("Rasa", "bílá", 5)[0, ]

  expect_equal(ncol(characteristics_table(empty, list(patients = 0, notes = 0))), 4)
  expect_equal(nrow(characteristics_table(empty, list(patients = 0, notes = 0))), 0)
})
