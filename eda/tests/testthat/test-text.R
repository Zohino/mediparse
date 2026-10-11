text_notes <- function(texts, prefix = "n") {
  data.frame(
    note_id = paste0(prefix, seq_along(texts)),
    subject_id = as.character(seq_along(texts)),
    hadm_id = as.character(seq_along(texts)),
    text = texts,
    stringsAsFactors = FALSE
  )
}

metrics_of <- function(texts, env = parent.frame()) {
  con <- mimic_con(discharge = text_notes(texts), env = env)
  name <- note_metrics(con, "discharge")
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", DBI::dbQuoteIdentifier(con, name)))
  out[order(as.integer(substring(out$note_id, 2))), , drop = FALSE]
}

test_that("note_metrics vrátí název dočasné tabulky a jde volat opakovaně", {
  con <- mimic_con(discharge = text_notes(c("a b", "c")))

  first <- note_metrics(con, "discharge")
  second <- note_metrics(con, "discharge")

  expect_equal(first, second)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", first))$n, 2)
})

test_that("značka je přesně tři podtržítka a delší úsek se počítá zvlášť", {
  out <- metrics_of(c(
    "ab___cd",
    "___ start",
    "konec ___",
    "___ ___",
    "x ____ y",
    "bez značek",
    "a___ ___b ____ c___"
  ))

  expect_equal(out$markers, c(1, 1, 1, 2, 0, 0, 3))
  expect_equal(out$longer_runs, c(0, 0, 0, 0, 1, 0, 1))
})

test_that("NULL text se měří jako prázdný, takže všechny metriky mají stejný základ", {
  out <- metrics_of(c(NA, ""))

  expect_equal(out$chars, c(0, 0))
  expect_equal(out$words, c(0, 0))
  expect_equal(out$markers, c(0, 0))
})

test_that("slova jsou úseky bez mezer a znaky se počítají včetně konců řádků", {
  out <- metrics_of(c("jedna dva\ntři", "a   b \t c\n\nd", "slovo"))

  expect_equal(out$words, c(3, 4, 1))
  expect_equal(out$chars, c(13, 12, 5))
})

test_that("length_percentiles spočítá percentily a průměr nad známým rozdělením", {
  texts <- vapply(1:1000, function(i) strrep("a", i), character(1))
  con <- mimic_con(discharge = text_notes(texts))
  out <- length_percentiles(con, "discharge")
  chars <- out[out$metrika == "znaky", ]

  expect_equal(chars$tabulka, "discharge")
  expect_equal(chars$n, 1000)
  expect_equal(chars$prumer, 500.5)
  expect_equal(chars$p01, 10)
  expect_equal(chars$p25, 250)
  expect_equal(chars$p50, 500)
  expect_equal(chars$p99, 990)
  expect_setequal(out$metrika, c("znaky", "slova", "znacky"))
})

test_that("length_percentiles počítá NULL text jako prázdný a pro prázdnou tabulku vrátí n = 0", {
  con <- mimic_con(discharge = text_notes(c("abc", NA)))
  out <- length_percentiles(con, c("discharge", "radiology"))

  expect_equal(out$n[out$tabulka == "discharge" & out$metrika == "znaky"], 2)
  expect_equal(out$n[out$tabulka == "radiology" & out$metrika == "znaky"], 0)
  expect_true(is.na(out$p50[out$tabulka == "radiology" & out$metrika == "znaky"]))
})

test_that("n percentilů, zprav značek a průměry sedí napříč funkcemi i s NULL textem", {
  con <- mimic_con(discharge = text_notes(c("___ a", NA, "", "b ___ ___", NA)))
  percentiles <- length_percentiles(con, "discharge")
  markers <- marker_summary(con, "discharge")

  expect_true(all(percentiles$n == markers$zprav))
  expect_equal(markers$zprav, 5)
  mean_markers <- percentiles$prumer[percentiles$metrika == "znacky"]
  expect_equal(mean_markers, markers$znacek / markers$zprav)
})

test_that("agregace čtou hotovou tabulku metrik a text znovu neprocházejí", {
  con <- mimic_con(discharge = text_notes(c("___", "a", "b")))
  name <- note_metrics(con, "discharge")
  DBI::dbExecute(con, sprintf("DELETE FROM %s", DBI::dbQuoteIdentifier(con, name)))

  expect_equal(marker_summary(con, "discharge")$zprav, 0)
  expect_equal(length_percentiles(con, "discharge")$n[1], 0)
})

test_that("format_percentiles skryje percentil bez deseti zpráv na každé straně", {
  texts <- vapply(1:150, function(i) strrep("a", i), character(1))
  con <- mimic_con(discharge = text_notes(texts))
  shown <- format_percentiles(length_percentiles(con, "discharge"))
  chars <- shown[shown$metrika == "znaky", ]

  expect_equal(chars$p01, "–")
  expect_equal(chars$p05, "–")
  expect_equal(chars$p10, "15")
  expect_equal(chars$p50, "75")
  expect_equal(chars$p95, "–")
  expect_equal(chars$p99, "–")
  expect_equal(chars$prumer, "75,5")
})

test_that("format_percentiles skryje průměr při n pod prahem", {
  con <- mimic_con(discharge = text_notes(c("a", "bb", "ccc")))
  shown <- format_percentiles(length_percentiles(con, "discharge"))

  expect_true(all(shown$prumer == "–"))
  expect_true(all(shown$p50 == "–"))
})

test_that("length_histogram dá koš 0 délce 1, prázdné a NULL texty nekreslí a sečte se do neprázdných zpráv", {
  texts <- c("", "a", NA, strrep("a", 9), strrep("a", 10), strrep("a", 1000), strrep("a", 1500))
  con <- mimic_con(discharge = text_notes(texts), radiology = text_notes(c("x", "yy"), prefix = "r"))
  out <- length_histogram(con, c("discharge", "radiology"))
  discharge <- out[out$tabulka == "discharge", ]

  expect_equal(discharge$n[discharge$kos == 0], 1)
  expect_equal(discharge$n[discharge$kos == 0.9], 1)
  expect_equal(discharge$n[discharge$kos == 1], 1)
  expect_equal(discharge$n[discharge$kos == 3], 1)
  expect_equal(discharge$n[discharge$kos == 3.1], 1)
  expect_equal(sum(discharge$n), 5)
  expect_equal(sum(out$n[out$tabulka == "radiology"]), 2)
})

test_that("marker_summary sečte zprávy, značky a delší úseky", {
  con <- mimic_con(
    discharge = text_notes(c("___ a ___", "bez", "x ____ y ___", NA))
  )
  out <- marker_summary(con, c("discharge", "radiology"))
  discharge <- out[out$tabulka == "discharge", ]

  expect_equal(discharge$zprav, 4)
  expect_equal(discharge$znacek, 3)
  expect_equal(discharge$se_znackou, 2)
  expect_equal(discharge$delsi_useky, 1)
  expect_equal(out$zprav[out$tabulka == "radiology"], 0)
  expect_equal(out$znacek[out$tabulka == "radiology"], 0)
})

test_that("na syntetickém tvaru s jedinečnými řetězci nic nespadne a percentily se maskují", {
  texts <- paste0("eda-sentinel-discharge-text-", 1:12)
  con <- mimic_con(discharge = text_notes(texts))

  expect_no_error(shown <- format_percentiles(length_percentiles(con, "discharge")))
  expect_no_error(marker_summary(con, "discharge"))
  expect_no_error(length_histogram(con, "discharge"))
  expect_true(all(shown$p01 == "–"))
  expect_true(all(shown$p99 == "–"))
})
