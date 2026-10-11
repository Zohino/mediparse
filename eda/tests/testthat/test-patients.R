EMPTY_TABLES <- list(
  discharge = c("note_id", "subject_id", "hadm_id"),
  radiology = c("note_id", "subject_id", "hadm_id"),
  diagnoses_icd = c("hadm_id", "icd_code", "icd_version"),
  patients = c("subject_id", "anchor_year_group")
)

text_frame <- function(rows, columns) {
  if (is.null(rows)) {
    return(as.data.frame(
      setNames(rep(list(character()), length(columns)), columns),
      stringsAsFactors = FALSE
    ))
  }
  as.data.frame(rows, stringsAsFactors = FALSE)
}

mimic_con <- function(..., env = parent.frame()) {
  given <- list(...)
  dir <- file.path(withr::local_tempdir(.local_envir = env), "parquet")
  dir.create(dir)
  writeLines('{"prefix":"t","rows":1}', file.path(dir, "synthetic.json"))
  writer <- DBI::dbConnect(duckdb::duckdb())
  for (name in names(EMPTY_TABLES)) {
    frame <- text_frame(given[[name]], EMPTY_TABLES[[name]])
    DBI::dbWriteTable(writer, name, frame)
    DBI::dbExecute(writer, sprintf(
      "COPY %s TO '%s' (FORMAT parquet)", name, file.path(dir, paste0(name, ".parquet"))
    ))
  }
  DBI::dbDisconnect(writer, shutdown = TRUE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, .local_envir = env)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE), envir = env)
  con
}

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

coverage_frame <- function(discharge, radiology) {
  data.frame(
    tabulka = c("discharge", "radiology"),
    zprav = c(sum(discharge), sum(radiology)),
    bez_hadm = c(discharge[1], radiology[1]),
    bez_icd = c(discharge[2], radiology[2]),
    s_icd = c(discharge[3], radiology[3])
  )
}

test_that("coverage_mask potlačí v každém řádku tři skupiny stejným jádrem", {
  mask <- coverage_mask(coverage_frame(c(4, 500, 5000), c(30, 40, 1000)))

  expect_named(mask, c("bez_hadm", "bez_icd", "s_icd"))
  expect_equal(mask$bez_hadm, c(TRUE, FALSE))
  expect_equal(mask$bez_icd, c(TRUE, FALSE))
  expect_equal(mask$s_icd, c(FALSE, FALSE))
})

test_that("coverage_mask skryje dopočitatelnou skupinu i bez n pod prahem", {
  mask <- coverage_mask(coverage_frame(c(4, 20, 5000), c(30, 40, 1000)))

  expect_equal(mask$bez_hadm[1], TRUE)
  expect_equal(mask$bez_icd[1], TRUE)
})

test_that("coverage_verdict porovnává podíl, ne absolutní počet", {
  coverage <- coverage_frame(c(100, 100, 800), c(1000, 4000, 5000))

  expect_equal(coverage_verdict(coverage), "discharge")
})

test_that("coverage_verdict vybere radiologické zprávy s vyšším podílem", {
  coverage <- coverage_frame(c(100, 400, 500), c(10, 90, 8000))

  expect_equal(coverage_verdict(coverage), "radiology")
})

test_that("coverage_verdict řekne nelze, když je skupina potlačená nebo tabulka chybí", {
  expect_equal(coverage_verdict(coverage_frame(c(100, 400, 5), c(10, 90, 8000))), "nelze")
  expect_equal(coverage_verdict(coverage_frame(c(0, 0, 0), c(10, 90, 8000))), "nelze")
  expect_equal(
    coverage_verdict(coverage_frame(c(100, 100, 800), c(10, 10, 80))[1, ]),
    "nelze"
  )
})
