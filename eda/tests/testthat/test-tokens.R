token_cache <- function(env = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = env)
  dir.create(file.path(root, "tokens"))
  writeLines("{}", file.path(root, "tokens", "cache.json"))
  withr::local_envvar(MEDIPARSE_EDA_CACHE = root, .local_envir = env)
  file.path(root, "tokens")
}

write_token_input <- function(con, file, rows) {
  directory <- file.path(Sys.getenv("MEDIPARSE_EDA_CACHE"), "tokens")
  DBI::dbWriteTable(con, "token_input", rows, overwrite = TRUE)
  DBI::dbExecute(con, sprintf("COPY token_input TO %s (FORMAT parquet)", DBI::dbQuoteString(con, file.path(directory, file))))
  DBI::dbExecute(con, "DROP TABLE token_input")
}

fixture_token_counts <- function(con, table, counts) {
  write_token_input(con, paste0("note_tokens_", table, ".parquet"), data.frame(
    note_id = sprintf("n%d", seq_along(counts)), xlmr = as.integer(counts + 2),
    mdeberta = as.integer(counts + 2), translategemma = as.integer(counts), stringsAsFactors = FALSE
  ))
  note_tokens(con, table)
}

window_frame <- function(specials = 2, capacity = 510, scale = 1) {
  data.frame(tokenizer = "xlmr", specials = specials, capacity = capacity, scale = scale, stringsAsFactors = FALSE)
}

window_of <- function(counts, windows = window_frame()) {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", counts)
  window_summary(con, "discharge", windows)
}

sample_of <- function(counts, parts, fits, limit = 4) {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", counts)
  write_token_input(con, "translation_sample.parquet", data.frame(
    note_table = rep("discharge", length(counts)), note_id = sprintf("n%d", seq_along(counts)),
    fits = as.logical(fits), parts = as.integer(parts), unsplittable = parts == 0L,
    stringsAsFactors = FALSE
  ))
  split_sample(con, "discharge", limit)
}

parallel_input <- function(con, zero = FALSE) {
  english <- c(2L, 4L, 10L, 20L)
  czech <- c(3L, 6L, 15L, 30L)
  if (zero) { english <- c(english, 0L); czech <- c(czech, 1L) }
  write_token_input(con, "parallel_tokens.parquet", data.frame(
    pair_id = letters[seq_along(english)], tokenizer = "translategemma", english = english, czech = czech,
    stringsAsFactors = FALSE
  ))
}

test_that("token_percentiles spočítá percentily a průměr délky v tokenech", {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", 1:1000)
  out <- token_percentiles(con, "discharge", "translategemma")

  expect_equal(out$metrika, "translategemma")
  expect_equal(out$n, 1000)
  expect_equal(out$prumer, 500.5)
  expect_equal(out$p50, 500)
  expect_equal(out$p95, 950)
  expect_equal(out$maximum, 1000)
})

test_that("token_percentiles pro prázdnou tabulku vrátí n = 0", {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "radiology", numeric())
  out <- token_percentiles(con, "radiology", c("translategemma", "xlmr"))

  expect_equal(out$n, c(0, 0))
  expect_true(all(is.na(out$p50)))
})

test_that("token_percentiles bez spočtené tabulky selže", {
  con <- mimic_con()

  expect_error(token_percentiles(con, "discharge", "translategemma"), "note_tokens")
})

test_that("zpráva přesně na hranici okna encoderu se vejde a o token delší ne", {
  out <- window_of(c(510, 511))

  expect_equal(out$zprav, 2)
  expect_equal(out$v_okne, 1)
})

test_that("části encoderu pro 512, 513, 1022 a 1023 tokenů", {
  parts <- function(total) window_of(total - 2)$casti

  expect_equal(parts(512), 1)
  expect_equal(parts(513), 2)
  expect_equal(parts(1022), 2)
  expect_equal(parts(1023), 3)
})

test_that("window_summary sečte části a prázdnou zprávu nepočítá do částí", {
  out <- window_of(c(0, 100, 600))

  expect_equal(out$zprav, 3)
  expect_equal(out$v_okne, 2)
  expect_equal(out$casti, 1 + 2)
})

test_that("window_summary s měřítkem přepočte délku na jiný jazyk", {
  out <- window_of(c(300, 400), window_frame(scale = 1.5))

  expect_equal(out$v_okne, 1)
  expect_equal(out$casti, 1 + 2)
})

test_that("window_summary s neceločíselným měřítkem nepřeteče u dlouhé zprávy", {
  out <- window_of(11302, window_frame(scale = 1.34285714285714))

  expect_equal(out$v_okne, 0)
  expect_equal(out$casti, 30)
})

test_that("window_summary vrátí řádek pro každou tabulku a tokenizér", {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", c(1, 2))
  fixture_token_counts(con, "radiology", 1)
  windows <- rbind(
    cbind(tokenizer = "translategemma", window_frame(0, 1)[-1]),
    cbind(tokenizer = "xlmr", window_frame(2, 510)[-1])
  )
  out <- window_summary(con, c("discharge", "radiology"), windows)

  expect_equal(nrow(out), 4)
  expect_setequal(names(out), c("tabulka", "tokenizer", "zprav", "v_okne", "casti"))
  expect_equal(out$v_okne[out$tabulka == "discharge" & out$tokenizer == "translategemma"], 1)
})

test_that("window_summary prázdné tabulky vrátí nuly", {
  out <- window_of(numeric())

  expect_equal(c(out$zprav, out$v_okne, out$casti), c(0, 0, 0))
})

test_that("token_histogram dá koš 0 jedné zprávě a prázdné zprávy nekreslí", {
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", c(0, 1, 9, 10, 1000))
  out <- token_histogram(con, "discharge", "translategemma")

  expect_equal(out$n[out$kos == 0], 1)
  expect_equal(out$n[out$kos == 0.9], 1)
  expect_equal(out$n[out$kos == 1], 1)
  expect_equal(out$n[out$kos == 3], 1)
  expect_equal(sum(out$n), 4)
  expect_setequal(names(out), c("tabulka", "tokenizer", "kos", "n"))
})

test_that("encoder_verdict rozlišuje ořez, většinu v okně a chunkování", {
  expect_equal(encoder_verdict(0.97, 300, 512), "orez")
  expect_equal(encoder_verdict(0.7, 400, 512), "vetsina")
  expect_equal(encoder_verdict(0.3, 900, 512), "chunkovani")
  expect_equal(encoder_verdict(NA_real_, NA_real_, 512), "nelze")
})

test_that("funkce nad tokeny nespadnou na jedinečných řetězcích syntetického tvaru", {
  con <- mimic_con()
  token_cache()
  tables <- c("discharge", "radiology")
  fixture_token_counts(con, "discharge", c(1, 1))
  fixture_token_counts(con, "radiology", 1)

  expect_no_error(token_percentiles(con, tables, c("translategemma", "xlmr")))
  expect_no_error(window_summary(con, tables, window_frame()))
  expect_no_error(token_histogram(con, tables, "xlmr"))
  expect_no_error(format_percentiles(token_percentiles(con, tables, "xlmr")))
})

test_that("translation_limit dává známé hodnoty a hranici zprávy, která se ještě vejde", {
  expect_equal(translation_limit(2048, 2.2, 75), 616.5625)
  expect_equal(translation_limit(100, 1, 0), 50)
})

test_that("spodní mez překladu na hranici limitu: 616 tokenů je jedna část, 617 dvě", {
  windows <- data.frame(
    tokenizer = "translategemma", specials = 0, capacity = translation_limit(2048, 2.2, 75), scale = 1,
    stringsAsFactors = FALSE
  )
  con <- mimic_con()
  token_cache()
  fixture_token_counts(con, "discharge", c(616, 617))
  out <- window_summary(con, "discharge", windows)

  expect_equal(out$v_okne, 1)
  expect_equal(out$casti, 1 + 2)
})

test_that("token_windows složí okna encoderů a překladu a měřítko českého textu", {
  out <- token_windows(c(xlmr = 2, mdeberta = 2), translation = c(gemma = 616.5625), scale = 1.5)

  expect_equal(out$tokenizer, c("xlmr", "mdeberta", "gemma"))
  expect_equal(out$specials, c(2, 2, 0))
  expect_equal(out$capacity, c(510, 510, 616.5625))
  expect_equal(out$scale, c(1.5, 1.5, 1.5))
})

test_that("split_sample porovná přesné dělení se spodní mezí", {
  out <- sample_of(c(3, 8), parts = c(1, 3), fits = c(TRUE, FALSE))

  expect_equal(out$zprav, 2)
  expect_equal(out$deleno, 1)
  expect_equal(out$casti_presne, 1 + 3)
  expect_equal(out$casti_dolni, 1 + 2)
  expect_equal(out$pomer, 4 / 3)
  expect_equal(out$neshod, 0)
  expect_equal(out$nedelitelnych, 0)
})

test_that("split_sample spočítá zprávy, u nichž se vzorec neshoduje s fits", {
  out <- sample_of(c(4, 3), parts = c(1, 1), fits = c(TRUE, TRUE), limit = 3.5)

  expect_equal(out$neshod, 1)
})

test_that("split_sample spočítá nedělitelné zprávy zvlášť a do částí je nezahrne", {
  out <- sample_of(c(8, 6, 2), parts = c(0, 2, 1), fits = c(FALSE, FALSE, TRUE))

  expect_equal(out$nedelitelnych, 1)
  expect_equal(out$casti_presne, 2 + 1)
  expect_equal(out$casti_dolni, 2 + 1)
})

test_that("split_sample vezme jen n zpráv a pro prázdný zdroj vrátí nuly bez poměru", {
  expect_equal(sample_of(c(1, 2, 3), parts = c(1, 1, 1), fits = c(TRUE, TRUE, TRUE))$zprav, 3)
  out <- sample_of(numeric(), integer(), logical())

  expect_equal(c(out$zprav, out$casti_presne, out$casti_dolni), c(0, 0, 0))
  expect_true(is.na(out$pomer))
})

test_that("sample_table formátuje počty, podíl dělených zpráv a poměr", {
  sample <- data.frame(
    tabulka = "discharge", zprav = 1000, nedelitelnych = 0, deleno = 400,
    casti_presne = 1500, casti_dolni = 1400, pomer = 1500 / 1400, neshod = 12,
    stringsAsFactors = FALSE
  )
  out <- sample_table(sample)

  expect_equal(out$tabulka, "discharge")
  expect_equal(out$pomer, "1,07")
  expect_equal(out$deleno_podil, "40,0")
  expect_equal(out$neshod, "12")
})

test_that("translation_verdict rozlišuje korpus bez dělení a s dělením", {
  expect_equal(translation_verdict(0, 100), "nedeli")
  expect_equal(translation_verdict(30, 100), "deli")
  expect_equal(translation_verdict(0, 0), "nelze")
})

test_that("parallel_ratio spočítá medián a 95. percentil poměru CS/EN", {
  con <- mimic_con()
  token_cache()
  parallel_input(con)
  windows <- token_windows(c(translategemma = 0), translation = numeric())
  windows$capacity <- 12
  out <- parallel_ratio(con, windows)

  expect_equal(out$tokenizer, "translategemma")
  expect_equal(out$zprav, 4)
  expect_equal(out$median, 1.5)
  expect_equal(out$p95, 1.5)
})

test_that("parallel_ratio spočítá podíl zpráv v okně v obou jazycích", {
  con <- mimic_con()
  token_cache()
  parallel_input(con)
  windows <- token_windows(c(translategemma = 0), translation = numeric())
  windows$capacity <- 12
  out <- parallel_ratio(con, windows)

  expect_equal(out$v_okne_en, 3)
  expect_equal(out$v_okne_cs, 2)
})

test_that("parallel_ratio ignoruje zprávy bez anglických tokenů při poměru", {
  con <- mimic_con()
  token_cache()
  parallel_input(con, zero = TRUE)
  windows <- token_windows(c(translategemma = 0), translation = numeric())
  out <- parallel_ratio(con, windows)

  expect_equal(out$zprav, 5)
  expect_equal(out$median, 1.5)
})

test_that("token_windows přijme měřítko pojmenované podle tokenizéru", {
  out <- token_windows(
    c(xlmr = 2, mdeberta = 2), translation = c(gemma = 600),
    scale = c(gemma = 1.7, xlmr = 1.3, mdeberta = 1.5)
  )

  expect_equal(out$scale, c(1.3, 1.5, 1.7))
})

test_that("czech_table složí podíl v okně a části v angličtině a v odhadu pro češtinu", {
  english <- data.frame(
    tabulka = "discharge", tokenizer = "xlmr", zprav = 200, v_okne = 80, casti = 500,
    stringsAsFactors = FALSE
  )
  czech <- transform(english, v_okne = 20, casti = 900)
  out <- czech_table(english, czech, "xlmr")

  expect_equal(out$podil_en, "40,0")
  expect_equal(out$podil_cs, "10,0")
  expect_equal(out$casti_en, "500")
  expect_equal(out$casti_cs, "900")
})

test_that("czech_verdict porovná podíl v okně anglicky a česky", {
  expect_equal(czech_verdict(0.9, 0.88), "bez_dopadu")
  expect_equal(czech_verdict(0.9, 0.5), "pokles")
  expect_equal(czech_verdict(NA_real_, 0.5), "nelze")
})

test_that("czech_table vynechá tokenizér, který není encoder", {
  english <- data.frame(
    tabulka = "discharge", tokenizer = c("xlmr", "translategemma"), zprav = 200,
    v_okne = c(80, 90), casti = c(500, 300), stringsAsFactors = FALSE
  )
  out <- czech_table(english, english, c("xlmr", "mdeberta"))

  expect_equal(out$tokenizer, "XLM-R")
})

test_that("parallel_table formátuje poměr a podíl v okně a tokenizér pojmenuje", {
  ratio <- data.frame(
    tokenizer = "xlmr", zprav = 200, median = 1.34, p95 = 1.42, v_okne_en = 74, v_okne_cs = 0,
    stringsAsFactors = FALSE
  )
  out <- parallel_table(ratio)

  expect_equal(out$tokenizer, "XLM-R")
  expect_equal(out$median, "1,34")
  expect_equal(out$p95, "1,42")
  expect_equal(out$podil_en, "37,0")
  expect_equal(out$podil_cs, "0,0")
})

test_that("tokenizer_table zkrátí revizi a uvede okno, úlohu a verzi knihovny", {
  models <- data.frame(
    klic = "xlmr", model = "FacebookAI/xlm-roberta-base",
    revize = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", okno = 512,
    uloha = "encoder", stringsAsFactors = FALSE
  )
  out <- tokenizer_table(models, "5.18.0")

  expect_equal(out$tokenizer, "XLM-R")
  expect_equal(out$revize, "e73636d4f797")
  expect_equal(out$okno, "512")
  expect_equal(out$transformers, "5.18.0")
})

test_that("window_table spočítá podíl v okně a části na zprávu", {
  win <- data.frame(
    tabulka = "discharge", tokenizer = "xlmr", zprav = 1000, v_okne = 250, casti = 4000,
    stringsAsFactors = FALSE
  )
  out <- window_table(win)

  expect_equal(out$tokenizer, "XLM-R")
  expect_equal(out$podil, "25,0")
  expect_equal(out$mimo, "750")
  expect_equal(out$casti, paste0("4", NBSP, "000"))
})

test_that("histogram_shares počítá podíl z neprázdných zpráv každé skupiny", {
  hist <- data.frame(
    tabulka = "discharge", tokenizer = c("xlmr", "xlmr", "mdeberta"), kos = c(1, 2, 1),
    n = c(30, 70, 10), stringsAsFactors = FALSE
  )
  out <- histogram_shares(hist, c("tabulka", "tokenizer"))

  expect_equal(out$podil, c(0.3, 0.7, 1))
  expect_equal(out$dolni, c(1, 2, 1))
  expect_equal(out$tokenizer, c("xlmr", "xlmr", "mdeberta"))
})

test_that("corpus_verdict porovná mediány délky v tokenech obou korpusů", {
  expect_equal(corpus_verdict(2500, 150), "radiologie_kratsi")
  expect_equal(corpus_verdict(150, 2500), "propousteci_kratsi")
  expect_equal(corpus_verdict(100, 100), "stejne")
  expect_equal(corpus_verdict(NA_real_, 100), "nelze")
})

test_that("tokenové fixtury Pythonu splňují všechny kontrakty konzumenta", {
  con <- mimic_con()
  directory <- token_cache()
  files <- c("note_tokens_discharge.parquet", "note_tokens_radiology.parquet", "tokenizers.parquet", "translation_prompt.parquet", "translation_sample.parquet", "parallel_tokens.parquet")
  file.copy(file.path(CONTRACT_FIXTURES, files), directory)
  expect_identical(note_tokens(con, "discharge"), "note_tokens_discharge")
  expect_identical(note_tokens(con, "radiology"), "note_tokens_radiology")
  expect_equal(nrow(tokenizer_settings(con)), 3)
  expect_equal(translation_prompt(con)$prompt_overhead, 3)
  source <- tokens_source(con, "translation_sample.parquet", SPLIT_COLUMNS)
  expect_equal(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", source))$n, 6)
  windows <- token_windows(c(xlmr = 2, mdeberta = 2), c(translategemma = 600))
  expect_equal(parallel_ratio(con, windows)$median, c(1.5, 1.5, 1.5))
  expect_equal(split_sample(con, "discharge", 2)$nedelitelnych, 1)
})

test_that("tokenová cache odmítá BIGINT místo INTEGER", {
  con <- mimic_con()
  directory <- token_cache()
  source <- file.path(CONTRACT_FIXTURES, "note_tokens_discharge.parquet")
  target <- file.path(directory, "note_tokens_discharge.parquet")
  DBI::dbExecute(con, sprintf(
    "COPY (SELECT note_id, xlmr::BIGINT AS xlmr, mdeberta, translategemma FROM read_parquet(%s)) TO %s (FORMAT parquet)",
    DBI::dbQuoteString(con, source), DBI::dbQuoteString(con, target)
  ))
  expect_error(note_tokens(con, "discharge"), "sloupec xlmr")
})

test_that("tokenová cache odmítá NULL na hranici konzumenta", {
  con <- mimic_con()
  token_cache()
  write_token_input(con, "translation_prompt.parquet", data.frame(prompt_overhead = NA_integer_, split_ratio = 2.2))
  expect_error(translation_prompt(con), "NULL")
})

test_that("nepublikovaná tokenová cache odmítne i hotový parquet", {
  con <- mimic_con()
  directory <- token_cache()
  file.copy(file.path(CONTRACT_FIXTURES, "note_tokens_discharge.parquet"), directory)
  unlink(file.path(directory, "cache.json"))
  expect_error(note_tokens(con, "discharge"), "cache není publikovaná")
})
