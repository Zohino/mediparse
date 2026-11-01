notes <- function(subject, hadm, prefix = "n") {
  data.frame(
    note_id = paste0(prefix, seq_along(subject)),
    subject_id = subject,
    hadm_id = hadm,
    stringsAsFactors = FALSE
  )
}

icd <- function(hadm, version) {
  data.frame(
    hadm_id = hadm,
    icd_code = paste0("K", seq_along(hadm)),
    icd_version = version,
    stringsAsFactors = FALSE
  )
}

test_that("patient_summary spočítá pacienty, přijetí a zprávy", {
  con <- mimic_con(discharge = notes(c("1", "1", "1", "2", "3"), c("a", "a", "b", "c", "d")))
  out <- patient_summary(con)

  expect_equal(out$patients, 3)
  expect_equal(out$admissions, 4)
  expect_equal(out$notes, 5)
  expect_equal(out$multi_note_patients, 1)
})

test_that("patient_summary nezapočte zprávu s NULL hadm_id jako přijetí", {
  con <- mimic_con(discharge = notes(c("1", "2", "2"), c("a", NA, "b")))
  out <- patient_summary(con)

  expect_equal(out$admissions, 2)
  expect_equal(out$notes, 3)
})

test_that("patient_summary čte tabulku podle argumentu", {
  con <- mimic_con(
    discharge = notes("1", "a"),
    radiology = notes(c("1", "2"), c("a", "b"), prefix = "r")
  )

  expect_equal(patient_summary(con, "radiology")$notes, 2)
})

test_that("patient_summary zvládne prázdnou tabulku", {
  out <- patient_summary(mimic_con())

  expect_equal(unlist(out, use.names = FALSE), c(0, 0, 0, 0))
})

test_that("notes_per_patient sloučí vysoké počty do koše cap+", {
  subjects <- c(rep("1", 12), rep("2", 10), rep("3", 3), "4")
  con <- mimic_con(discharge = notes(subjects, seq_along(subjects)))
  out <- notes_per_patient(con, cap = 10)

  expect_equal(out$kos, c("1", "3", "10+"))
  expect_equal(out$n, c(1, 1, 2))
})

test_that("notes_per_patient nevypisuje koše bez pacientů a řadí číselně", {
  subjects <- c(rep("1", 2), rep("2", 2), rep("3", 9), rep("4", 1))
  con <- mimic_con(discharge = notes(subjects, seq_along(subjects)))
  out <- notes_per_patient(con, cap = 10)

  expect_equal(out$kos, c("1", "2", "9"))
  expect_equal(out$n, c(1, 2, 1))
})

test_that("notes_per_patient vrátí z prázdné tabulky prázdný data.frame", {
  out <- notes_per_patient(mimic_con())

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kos", "n"))
})

test_that("notes_per_admission počítá zprávy na přijetí a koš cap+", {
  hadm <- c("a", "b", "b", "c", "c", "c", "d", "d", "d", "d", NA, NA)
  con <- mimic_con(discharge = notes(rep("1", length(hadm)), hadm))
  out <- notes_per_admission(con, cap = 3)

  expect_equal(out$kos, c("1", "2", "3+"))
  expect_equal(out$n, c(1, 1, 2))
})

test_that("notes_per_admission vrátí z prázdné tabulky prázdný data.frame", {
  out <- notes_per_admission(mimic_con())

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kos", "n"))
})

test_that("hadm_coverage rozliší chybějící hadm_id, chybějící ICD a pokrytí", {
  con <- mimic_con(
    discharge = notes(c("1", "2", "3", "4"), c(NA, "a", "b", "c")),
    radiology = notes(c("1", "2"), c("a", NA), prefix = "r"),
    diagnoses_icd = icd(c("b", "c", "c", "c"), c("9", "10", "10", "10"))
  )
  out <- hadm_coverage(con)

  expect_equal(out$tabulka, c("discharge", "radiology"))
  expect_equal(out$zprav, c(4, 2))
  expect_equal(out$bez_hadm, c(1, 1))
  expect_equal(out$bez_icd, c(1, 1))
  expect_equal(out$s_icd, c(2, 0))
})

test_that("hadm_coverage započte zprávu s více kódy jednou", {
  con <- mimic_con(
    discharge = notes("1", "a"),
    diagnoses_icd = icd(c("a", "a", "a"), c("10", "10", "10"))
  )
  out <- hadm_coverage(con, tables = "discharge")

  expect_equal(out$zprav, 1)
  expect_equal(out$s_icd, 1)
})

test_that("hadm_coverage sečte selhání a pokrytí do všech zpráv", {
  con <- mimic_con(
    discharge = notes(c("1", "2", "3"), c(NA, "a", "b")),
    diagnoses_icd = icd("b", "9")
  )
  out <- hadm_coverage(con, tables = "discharge")

  expect_equal(out$bez_hadm + out$bez_icd + out$s_icd, out$zprav)
})

test_that("icd_era rozdělí hospitalizace do vylučujících se kategorií", {
  con <- mimic_con(
    discharge = notes(rep("1", 5), c("a", "b", "c", "d", "d")),
    diagnoses_icd = icd(
      c("a", "b", "c", "c", "c"),
      c("9", "10", "9", "10", "10")
    )
  )
  out <- icd_era(con)

  expect_equal(out$era, c("jen ICD-9", "jen ICD-10", "obě éry", "bez kódu"))
  expect_equal(out$n, c(1, 1, 1, 1))
})

test_that("icd_era počítá jen hospitalizace s propouštěcí zprávou a sečte se", {
  con <- mimic_con(
    discharge = notes(c("1", "1", "2", "3"), c("a", "a", "b", NA)),
    diagnoses_icd = icd(c("a", "b", "zcela-jina"), c("10", "10", "9"))
  )
  out <- icd_era(con)

  expect_equal(sum(out$n), 2)
  expect_equal(out$era, "jen ICD-10")
  expect_equal(out$n, 2)
})

test_that("icd_era z prázdné tabulky vrátí prázdný data.frame", {
  out <- icd_era(mimic_con())

  expect_equal(nrow(out), 0L)
  expect_named(out, c("era", "n"))
})

test_that("icd_era_by_group rozpadne éru podle období pacienta", {
  con <- mimic_con(
    discharge = notes(c("1", "2", "3", "4"), c("a", "b", "c", "d")),
    diagnoses_icd = icd(c("a", "b", "c"), c("9", "10", "10")),
    patients = data.frame(
      subject_id = c("1", "2", "3"),
      anchor_year_group = c("2008 - 2010", "2017 - 2019", "2017 - 2019"),
      stringsAsFactors = FALSE
    )
  )
  out <- icd_era_by_group(con)

  expect_named(out, c("obdobi", "era", "n"))
  expect_equal(out$obdobi, c("2008 - 2010", "2017 - 2019", "neznámé"))
  expect_equal(out$era, c("jen ICD-9", "jen ICD-10", "bez kódu"))
  expect_equal(out$n, c(1, 2, 1))
})

test_that("icd_era_by_group sečte období na celkový počet hospitalizací", {
  con <- mimic_con(
    discharge = notes(c("1", "2"), c("a", "b")),
    patients = data.frame(
      subject_id = c("1", "2"),
      anchor_year_group = c("2008 - 2010", "2011 - 2013"),
      stringsAsFactors = FALSE
    )
  )

  expect_equal(sum(icd_era_by_group(con)$n), sum(icd_era(con)$n))
})

test_that("funkce nad syntetickým tvarem nespadnou a éra je celá bez kódu", {
  con <- mimic_con(
    discharge = data.frame(
      note_id = c("s-d-1", "s-d-2"), subject_id = c("s-d-3", "s-d-4"),
      hadm_id = c("s-d-5", "s-d-6"), stringsAsFactors = FALSE
    ),
    radiology = data.frame(
      note_id = "s-r-1", subject_id = "s-r-2", hadm_id = "s-r-3",
      stringsAsFactors = FALSE
    ),
    diagnoses_icd = data.frame(
      hadm_id = "s-i-1", icd_code = "s-i-2", icd_version = "s-i-3",
      stringsAsFactors = FALSE
    ),
    patients = data.frame(
      subject_id = "s-p-1", anchor_year_group = "s-p-2", stringsAsFactors = FALSE
    )
  )

  expect_equal(patient_summary(con)$patients, 2)
  expect_equal(notes_per_patient(con)$kos, "1")
  expect_equal(notes_per_admission(con)$kos, "1")
  expect_equal(hadm_coverage(con)$bez_icd, c(2, 1))
  expect_equal(icd_era(con)$era, "bez kódu")
  expect_equal(icd_era_by_group(con)$era, "bez kódu")
  expect_false(anyNA(unlist(hadm_coverage(con))))
})

coverage_frame <- function(discharge, radiology) {
  data.frame(
    tabulka = c("discharge", "radiology"),
    zprav = c(sum(discharge), sum(radiology)),
    bez_hadm = c(discharge[1], radiology[1]),
    bez_icd = c(discharge[2], radiology[2]),
    s_icd = c(discharge[3], radiology[3])
  )
}

test_that("coverage_verdict porovnává podíl, ne absolutní počet", {
  coverage <- coverage_frame(c(100, 100, 800), c(1000, 4000, 5000))

  expect_equal(coverage_verdict(coverage), "discharge")
})

test_that("coverage_verdict vybere radiologické zprávy s vyšším podílem", {
  coverage <- coverage_frame(c(100, 400, 500), c(10, 90, 8000))

  expect_equal(coverage_verdict(coverage), "radiology")
})

test_that("coverage_verdict porovná i podíl z malých počtů", {
  expect_equal(coverage_verdict(coverage_frame(c(100, 400, 5), c(10, 90, 8000))), "radiology")
  expect_equal(coverage_verdict(coverage_frame(c(1, 1, 8), c(2, 3, 5))), "discharge")
})

test_that("coverage_verdict řekne nelze, když je tabulka prázdná nebo chybí", {
  expect_equal(coverage_verdict(coverage_frame(c(0, 0, 0), c(10, 90, 8000))), "nelze")
  expect_equal(
    coverage_verdict(coverage_frame(c(100, 100, 800), c(10, 10, 80))[1, ]),
    "nelze"
  )
})

era_frame <- function(obdobi, era) {
  data.frame(obdobi = obdobi, era = era, n = rep(50, length(obdobi)), stringsAsFactors = FALSE)
}

test_that("era_transition pozná éry oddělené v čase", {
  out <- era_transition(era_frame(
    c("2008 - 2010", "2017 - 2019"), c("jen ICD-9", "jen ICD-10")
  ))

  expect_equal(out$kind, "separated")
  expect_equal(out$periods_9, "2008 - 2010")
  expect_equal(out$periods_10, "2017 - 2019")
  expect_length(out$both, 0)
})

test_that("era_transition pozná překryv v jednom období", {
  out <- era_transition(era_frame(
    c("2011 - 2013", "2011 - 2013", "2008 - 2010"),
    c("jen ICD-9", "jen ICD-10", "jen ICD-9")
  ))

  expect_equal(out$kind, "overlap")
  expect_equal(out$both, "2011 - 2013")
})

test_that("era_transition vyřadí neznámé období", {
  out <- era_transition(era_frame(
    c("neznámé", "neznámé", "2008 - 2010"),
    c("jen ICD-9", "jen ICD-10", "jen ICD-9")
  ))

  expect_equal(out$kind, "none")
  expect_equal(out$periods_9, "2008 - 2010")
  expect_length(out$periods_10, 0)
})

test_that("era_transition z prázdné tabulky nedoloží přechod", {
  expect_equal(era_transition(era_frame(character(), character()))$kind, "none")
})


test_that("era_transition zachová překryv i při jediné hospitalizaci", {
  breakdown <- era_frame(c("2014 - 2016", "2014 - 2016"), c("jen ICD-10", "jen ICD-9"))
  breakdown$n <- c(1, 50)

  out <- era_transition(breakdown)

  expect_equal(out$kind, "overlap")
  expect_equal(out$periods_9, "2014 - 2016")
  expect_equal(out$periods_10, "2014 - 2016")
  expect_equal(out$both, "2014 - 2016")

  breakdown$n[1] <- 0
  expect_length(era_transition(breakdown)$periods_10, 0)
})
