corpus_case_frame <- function(rows, columns) {
  values <- lapply(columns, function(column) {
    vapply(rows, function(row) {
      if (is.null(row[[column]])) NA_character_ else row[[column]]
    }, character(1))
  })
  as.data.frame(setNames(values, columns), stringsAsFactors = FALSE)
}

cases <- jsonlite::read_json(project_path("tests", "fixtures", "input_corpus_cases.json"))
for (case in cases) {
  test_that(paste("vstupní korpus splňuje sdílený případ", case$name), {
    con <- mimic_con(
      discharge = corpus_case_frame(case$discharge, EMPTY_TABLES$discharge),
      diagnoses_icd = corpus_case_frame(case$diagnoses_icd, EMPTY_TABLES$diagnoses_icd)
    )
    funnel <- corpus_funnel(con)
    selected <- DBI::dbGetQuery(con, input_corpus_sql(con))

    expect_equal(funnel$zprav, vapply(case$funnel, `[[`, numeric(1), 1L))
    expect_equal(funnel$pacientu, vapply(case$funnel, `[[`, numeric(1), 2L))
    expect_setequal(selected$note_id, as.character(unlist(case$selected)))
    expect_equal(funnel$krok, FUNNEL_STEPS)
    expect_equal(names(selected), c("note_id", "subject_id", "hadm_id"))
    expect_equal(funnel$vice_zprav, unlist(case$multiple))
  })
}

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
