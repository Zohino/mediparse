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

residual_of <- function(texts, table = "discharge") {
  con <- mimic_con(discharge = text_notes(texts), env = parent.frame())
  residual_masks(con, table)
}

pattern_row <- function(out, pattern) out[out$vzor == pattern, ]

test_that("residual_masks počítá každý vzor zvlášť, zprávy jednou a výskyty všechny", {
  out <- residual_of(c(
    "[**Name 1**] a [**Hospital**] a <b>tučně</b>",
    "bez masky",
    "XXXX a XXX a xx",
    "<br> <br>"
  ))

  expect_equal(pattern_row(out, "[**…**]")$zprav, 1)
  expect_equal(pattern_row(out, "[**…**]")$vyskytu, 2)
  expect_equal(pattern_row(out, "<…>")$zprav, 2)
  expect_equal(pattern_row(out, "<…>")$vyskytu, 4)
  expect_equal(pattern_row(out, "XXX")$zprav, 1)
  expect_equal(pattern_row(out, "XXX")$vyskytu, 2)
})

test_that("XXXX je jeden výskyt a pro prázdnou tabulku jsou nuly", {
  con <- mimic_con(discharge = text_notes("XXXX"))
  out <- residual_masks(con, c("discharge", "radiology"))

  expect_equal(out$vyskytu[out$tabulka == "discharge" & out$vzor == "XXX"], 1)
  expect_true(all(out$zprav[out$tabulka == "radiology"] == 0))
  expect_true(all(out$vyskytu[out$tabulka == "radiology"] == 0))
})

test_that("residual_masks nevrací nalezené řetězce", {
  out <- residual_of("[**Tajné jméno**] <tajne>")

  expect_setequal(names(out), c("tabulka", "vzor", "zprav", "vyskytu"))
  expect_false(any(grepl("Tajn", unlist(out))))
})

quality_of <- function(texts) {
  con <- mimic_con(discharge = text_notes(texts), env = parent.frame())
  note_quality(con, "discharge")
}

test_that("note_quality rozliší NULL, prázdný text a hranice délek", {
  out <- quality_of(c(
    NA, "", strrep("a", 99), strrep("b", 100), strrep("c", 500), strrep("d", 501),
    strrep("e", 50000), strrep("f", 50001)
  ))

  expect_equal(out$zprav, 8)
  expect_equal(out$prazdne, 2)
  expect_equal(out$pod_100, 1)
  expect_equal(out$od_100_do_500, 2)
  expect_equal(out$nad_50000, 1)
  expect_equal(out$duplicity, 0)
})

test_that("note_quality počítá duplicity jako zprávy navíc nad první", {
  expect_equal(quality_of(rep("stejný text", 2))$duplicity, 1)
  expect_equal(quality_of(rep("stejný text", 3))$duplicity, 2)
  expect_equal(quality_of(c("a", "b", "a", "b", "c"))$duplicity, 2)
  expect_equal(quality_of(c(NA, NA, "", ""))$duplicity, 0)
})

age_of <- function(texts) {
  con <- mimic_con(discharge = text_notes(texts), env = parent.frame())
  age_masks(con)
}

test_that("age_masks rozpozná masku věku v různých zápisech", {
  out <- age_of(c(
    "He is a ___ year old man",
    "She is ___ y/o",
    "A ___-year-old woman",
    "A ___ Year Old woman",
    "nic"
  ))

  expect_equal(out$zprav, 5)
  expect_equal(out$bez_masky, 1)
  expect_equal(out$jedna, 4)
  expect_equal(out$vice, 0)
  expect_equal(out$kotva, 1)
})

test_that("age_masks rozdělí zprávy na bez masky, s jednou a s více maskami", {
  out <- age_of(c("___ yo, ___-year-old", "___ y.o.", "nic", NA))

  expect_equal(out$bez_masky, 2)
  expect_equal(out$jedna, 1)
  expect_equal(out$vice, 1)
})

test_that("age_masks nepočítá ___ years ago jako masku věku", {
  out <- age_of(c("diagnosed ___ years ago", "___ yearly"))

  expect_equal(out$bez_masky, 2)
})

test_that("age_masks počítá číselný věk, ne rok", {
  out <- age_of(c("A 67 year old man", "A 67-year-old man", "in 1967 year old", "67 yo", "no age"))

  expect_equal(out$ciselny, 3)
})

test_that("kotva vyžaduje is a nebo is an před maskou", {
  out <- age_of(c("is an ___ yo", "was ___ yo", "is a ___-year-old"))

  expect_equal(out$kotva, 2)
})

test_that("na syntetickém tvaru nic nespadne", {
  texts <- paste0("eda-sentinel-discharge-text-", 1:12)
  con <- mimic_con(discharge = text_notes(texts))

  expect_no_error(residual_masks(con, c("discharge", "radiology")))
  expect_no_error(note_quality(con, c("discharge", "radiology")))
  expect_no_error(age_masks(con))
  expect_equal(age_masks(con)$jedna, 0)
})

test_that("anchor_hidden skryje kotvu i vůči zprávám s maskou věku", {
  age <- list(zprav = 1000, bez_masky = 900, jedna = 60, vice = 40, kotva = 95)

  expect_false(complement_hidden(age$kotva, age$zprav))
  expect_true(anchor_hidden(age))
  expect_false(anchor_hidden(modifyList(age, list(kotva = 50))))
  expect_true(anchor_hidden(modifyList(age, list(kotva = 5))))
})

test_that("age_table skryje rozdělení jako celek a kotvu podle zpráv s maskou", {
  age <- list(zprav = 1000, bez_masky = 900, jedna = 60, vice = 40, kotva = 95, ciselny = 0)
  out <- age_table(age)

  expect_equal(nrow(out), 6L)
  expect_equal(out$n[2:3], c("900", "60"))
  expect_equal(out$n[5], paste0("<", NBSP, "10"))
  expect_equal(out$podil[5], "\u2013")
  expect_equal(out$n[6], paste0("<", NBSP, "10"))
  expect_equal(out$podil[1], "")
})

test_that("quality_table skryje kategorii podle doplňku do celku v každé tabulce", {
  quality <- data.frame(
    tabulka = c("discharge", "radiology"),
    zprav = c(1000, 100),
    prazdne = c(3, 20),
    pod_100 = c(50, 95),
    od_100_do_500 = c(100, 40),
    nad_50000 = c(5, 0),
    duplicity = c(20, 30)
  )
  out <- quality_table(quality)
  hidden <- paste0("<", NBSP, "10")

  expect_equal(out$ukazatel[1], "Zprávy celkem")
  expect_equal(nrow(out), 6L)
  expect_equal(out$n_discharge, c(format_number(1000), hidden, "50", "100", hidden, "20"))
  expect_equal(out$n_radiology, c("100", "20", hidden, "40", hidden, "30"))
  expect_equal(out$podil_radiology[3], "\u2013")
})

test_that("maska věku zachytí i years old a nezachytí years ago", {
  out <- age_of(c("is a ___ years old", "72 years old man", "___ years ago", "___-years-old"))

  expect_equal(out$jedna, 2)
  expect_equal(out$kotva, 1)
  expect_equal(out$ciselny, 1)
  expect_equal(out$bez_masky, 2)
})

test_that("marker_reference_verdict rozliší shodu, shodu se zbytkovými maskami a rozdíl", {
  expect_equal(marker_reference_verdict(60, 5, 60.6), "presne")
  expect_equal(marker_reference_verdict(40, 20, 60.6), "se_zbytkovymi")
  expect_equal(marker_reference_verdict(10, 5, 60.6), "rozdil")
  expect_equal(marker_reference_verdict(NA, NA, 60.6), "nelze")
})

test_that("marker_means vydělí součty počtem zpráv a pro prázdnou tabulku vrátí NA", {
  markers <- data.frame(tabulka = "discharge", zprav = 10, znacek = 600, se_znackou = 9, delsi_useky = 20)
  residual <- data.frame(
    tabulka = "discharge", vzor = c("[**…**]", "<…>", "XXX"), zprav = c(1, 2, 3), vyskytu = c(10, 20, 30)
  )
  out <- marker_means(markers, residual)

  expect_equal(out$presne, 60)
  expect_equal(out$delsi, 2)
  expect_equal(out$zbytkove, 6)

  empty <- marker_means(transform(markers, zprav = 0), residual)
  expect_true(is.na(empty$presne))
})

test_that("age_verdict a residual_verdict se řídí nulou, ne prahem DUA", {
  expect_equal(age_verdict(list(ciselny = 0)), "bez_ciselneho")
  expect_equal(age_verdict(list(ciselny = 3)), "ciselny")
  expect_equal(residual_verdict(data.frame(vyskytu = c(0, 0, 0))), "zadne")
  expect_equal(residual_verdict(data.frame(vyskytu = c(0, 2, 0))), "jsou")
})
