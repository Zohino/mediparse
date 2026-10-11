long_text <- function(seed) paste0(seed, strrep("x", SHORT_LIMIT))

discharge_rows <- function(subject, hadm, text) {
  data.frame(
    note_id = paste0("n", seq_along(subject)),
    subject_id = subject,
    hadm_id = hadm,
    text = text,
    stringsAsFactors = FALSE
  )
}

coded <- function(hadm) {
  data.frame(
    hadm_id = hadm,
    icd_code = paste0("K", seq_along(hadm)),
    icd_version = "10",
    stringsAsFactors = FALSE
  )
}

test_that("corpus_funnel postupně vyřadí zprávy bez kódu ICD, krátké a duplicitní", {
  con <- mimic_con(
    discharge = discharge_rows(
      c("1", "1", "2", "3", "4", "5"),
      c("a", "b", "c", NA, "d", "e"),
      c(long_text("A"), long_text("B"), long_text("C"), long_text("D"), "krátká", long_text("A"))
    ),
    diagnoses_icd = coded(c("a", "a", "b", "d", "e"))
  )
  out <- corpus_funnel(con)

  expect_equal(out$krok, FUNNEL_STEPS)
  expect_equal(out$zprav, c(6, 4, 3, 2))
  expect_equal(out$pacientu, c(5, 3, 2, 1))
  expect_equal(out$vice_zprav, c(1, 1, 1, 1))
})

test_that("corpus_funnel z duplicit ponechá zprávu s nejmenším note_id", {
  con <- mimic_con(
    discharge = discharge_rows(
      c("1", "2", "2"), c("a", "b", "c"), c(long_text("A"), long_text("A"), long_text("B"))
    ),
    diagnoses_icd = coded(c("a", "b", "c"))
  )
  out <- corpus_funnel(con)

  expect_equal(out$zprav[4], 2)
  expect_equal(out$pacientu[4], 2)
  expect_equal(out$vice_zprav[4], 0)
})

test_that("corpus_funnel vrátí nulové kroky pro prázdnou tabulku i bez spojení s ICD", {
  empty <- corpus_funnel(mimic_con())
  unjoined <- corpus_funnel(mimic_con(
    discharge = discharge_rows(c("1", "2"), c("a", "b"), c(long_text("A"), long_text("B"))),
    diagnoses_icd = coded(c("x", "y"))
  ))

  expect_equal(empty$krok, FUNNEL_STEPS)
  expect_equal(empty$zprav, c(0, 0, 0, 0))
  expect_equal(empty$pacientu, c(0, 0, 0, 0))
  expect_equal(unjoined$zprav, c(2, 0, 0, 0))
  expect_equal(unjoined$vice_zprav, c(0, 0, 0, 0))
})

comparison_inputs <- function() {
  list(
    summaries = list(
      discharge = list(patients = 1200),
      radiology = list(patients = 900)
    ),
    coverage = data.frame(
      tabulka = c("discharge", "radiology"),
      zprav = c(2000, 4000),
      bez_hadm = c(0, 2000),
      bez_icd = c(20, 100),
      s_icd = c(1980, 1900)
    ),
    quality = data.frame(
      tabulka = c("discharge", "radiology"),
      zprav = c(2000, 4000),
      prazdne = c(1, 0),
      pod_100 = c(2, 30),
      od_100_do_500 = c(5, 900),
      nad_50000 = c(0, 0),
      duplicity = c(3, 1500)
    ),
    shown = data.frame(
      tabulka = c("discharge", "radiology"),
      metrika = c("znaky", "znaky"),
      p50 = c("9 800", "1 200"),
      p95 = c("21 000", "3 100"),
      stringsAsFactors = FALSE
    )
  )
}

test_that("corpus_comparison srovná oba korpusy po ukazatelích", {
  inputs <- comparison_inputs()
  out <- corpus_comparison(inputs$summaries, inputs$coverage, inputs$quality, inputs$shown)

  expect_equal(names(out), c("ukazatel", "discharge", "radiology"))
  expect_equal(out$ukazatel, COMPARISON_ROWS)
  expect_equal(out$discharge, c(
    "2 000", "1 200", "0,0", "99,0", "9 800", "21 000", "3", "3"
  ))
  expect_equal(out$radiology, c(
    "4 000", "900", "50,0", "47,5", "1 200", "3 100", "30", "1 500"
  ))
})

test_that("corpus_comparison dá pomlčku chybějící nebo prázdné tabulce", {
  inputs <- comparison_inputs()
  coverage <- inputs$coverage
  coverage$zprav[2] <- 0
  out <- corpus_comparison(
    inputs$summaries[1],
    coverage,
    inputs$quality[1, , drop = FALSE],
    inputs$shown[1, , drop = FALSE]
  )

  expect_equal(out$radiology, c("0", "–", "–", "–", "–", "–", "–", "–"))
})
