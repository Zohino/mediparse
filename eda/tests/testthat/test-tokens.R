fake_tokenizer <- function(texts, add_special_tokens = TRUE) {
  if (!is.list(texts)) stop("texts musí být seznam, ne jeden řetězec")
  ids <- lapply(texts, function(text) {
    words <- strsplit(text, "[[:space:]]+")[[1]]
    count <- sum(nzchar(words)) + if (add_special_tokens) 2L else 0L
    seq_len(count)
  })
  list(input_ids = ids)
}

words_of <- function(n) paste(rep("a", n), collapse = " ")

test_that("token_lengths dá prázdnému textu a NA délku prázdného textu", {
  expect_equal(token_lengths(fake_tokenizer, c("", NA, "a b"), TRUE), c(2L, 2L, 4L))
})

test_that("token_lengths bez speciálních tokenů je odečte a drží pořadí vstupu", {
  texts <- c("a b c", "a", "", "a b")

  expect_equal(token_lengths(fake_tokenizer, texts, TRUE), c(5L, 3L, 2L, 4L))
  expect_equal(token_lengths(fake_tokenizer, texts, FALSE), c(3L, 1L, 0L, 2L))
})

test_that("token_lengths zvládne jednu zprávu a žádnou zprávu", {
  expect_equal(token_lengths(fake_tokenizer, "a b", TRUE), 4L)
  expect_equal(token_lengths(fake_tokenizer, character(), TRUE), integer())
})

test_that("token_lengths ořízne okrajové mezery jen na požádání", {
  tok <- function(texts, add_special_tokens = TRUE) {
    list(input_ids = lapply(texts, function(text) seq_len(nchar(text))))
  }

  expect_equal(token_lengths(tok, "  ab \n", FALSE), 6L)
  expect_equal(token_lengths(tok, "  ab \n", FALSE, strip = TRUE), 2L)
  expect_equal(token_lengths(tok, " ab ", FALSE, strip = TRUE), 2L)
})

spec_both <- function() {
  list(
    slova = list(tok = fake_tokenizer, special = FALSE),
    cele = list(tok = fake_tokenizer, special = TRUE)
  )
}

token_rows <- function(con, name) {
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", DBI::dbQuoteIdentifier(con, name)))
  out[order(out$note_id), , drop = FALSE]
}

test_that("note_tokens zapíše řádek na zprávu i přes víc dávek", {
  texts <- c("a", "a b", "a b c", "", NA)
  for (batch in c(2L, 5L, 2000L)) {
    con <- mimic_con(discharge = text_notes(texts))
    name <- note_tokens(con, "discharge", spec_both(), batch = batch)
    out <- token_rows(con, name)

    expect_equal(out$note_id, paste0("n", 1:5))
    expect_equal(out$slova, c(1, 2, 3, 0, 0))
    expect_equal(out$cele, c(3, 4, 5, 2, 2))
  }
})

test_that("note_tokens zvládne dávku o jedné zprávě", {
  con <- mimic_con(discharge = text_notes(c("a", "a b", "a b c")))
  out <- token_rows(con, note_tokens(con, "discharge", spec_both(), batch = 2L))

  expect_equal(out$slova, c(1, 2, 3))
})

test_that("note_tokens vytvoří prázdnou tabulku pro prázdný zdroj", {
  con <- mimic_con()
  name <- note_tokens(con, "discharge", spec_both())
  out <- token_rows(con, name)

  expect_equal(nrow(out), 0)
  expect_equal(names(out), c("note_id", "slova", "cele"))
})

test_that("note_tokens podruhé nepřepočítává", {
  con <- mimic_con(discharge = text_notes(c("a", "a b")))
  name <- note_tokens(con, "discharge", spec_both())
  DBI::dbExecute(con, sprintf("DELETE FROM %s", DBI::dbQuoteIdentifier(con, name)))
  again <- note_tokens(con, "discharge", spec_both())

  expect_equal(again, name)
  expect_equal(nrow(token_rows(con, again)), 0)
})

test_that("note_tokens předá tokenizéru zprávy v pořadí podle note_id", {
  seen <- character()
  tok <- function(texts, add_special_tokens = TRUE) {
    seen <<- c(seen, unlist(texts))
    fake_tokenizer(texts, add_special_tokens)
  }
  notes <- text_notes(c("c", "a", "b"))
  notes$note_id <- c("n3", "n1", "n2")
  con <- mimic_con(discharge = notes)
  note_tokens(con, "discharge", list(x = list(tok = tok, special = FALSE)), batch = 2L)

  expect_equal(seen, c("a", "b", "c"))
})

test_that("note_tokens po sobě nenechá pomocnou tabulku", {
  con <- mimic_con(discharge = text_notes(c("a", "b", "c")))
  note_tokens(con, "discharge", spec_both(), batch = 2L)

  expect_false("note_tokens_discharge_rows" %in% DBI::dbListTables(con))
})

test_that("note_tokens při chybě tokenizéru nenechá ani výslednou, ani pomocnou tabulku", {
  failing <- function(texts, add_special_tokens = TRUE) stop("selhání")
  con <- mimic_con(discharge = text_notes(c("a", "b", "c")))

  expect_error(note_tokens(con, "discharge", list(x = list(tok = failing, special = FALSE))), "selhání")
  expect_false(any(grepl("^note_tokens_", DBI::dbListTables(con))))
})

test_that("token_percentiles spočítá percentily a průměr délky v tokenech", {
  texts <- vapply(1:1000, function(i) words_of(i), character(1))
  con <- mimic_con(discharge = text_notes(texts))
  note_tokens(con, "discharge", list(slova = list(tok = fake_tokenizer, special = FALSE)))
  out <- token_percentiles(con, "discharge", "slova")

  expect_equal(out$metrika, "slova")
  expect_equal(out$n, 1000)
  expect_equal(out$prumer, 500.5)
  expect_equal(out$p50, 500)
  expect_equal(out$p95, 950)
  expect_equal(out$maximum, 1000)
})

test_that("token_percentiles pro prázdnou tabulku vrátí n = 0", {
  con <- mimic_con()
  note_tokens(con, "radiology", spec_both())
  out <- token_percentiles(con, "radiology", c("slova", "cele"))

  expect_equal(out$n, c(0, 0))
  expect_true(all(is.na(out$p50)))
})

test_that("token_percentiles bez spočtené tabulky selže", {
  con <- mimic_con(discharge = text_notes("a"))

  expect_error(token_percentiles(con, "discharge", "slova"), "note_tokens")
})

window_frame <- function(specials = 2, capacity = 510, scale = 1) {
  data.frame(
    tokenizer = "cele", specials = specials, capacity = capacity, scale = scale,
    stringsAsFactors = FALSE
  )
}

window_of <- function(counts, windows = window_frame()) {
  texts <- vapply(counts, words_of, character(1))
  con <- mimic_con(discharge = text_notes(texts))
  note_tokens(con, "discharge", list(cele = list(tok = fake_tokenizer, special = TRUE)))
  window_summary(con, "discharge", windows)
}

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
  con <- mimic_con(
    discharge = text_notes(c("a", "a b")), radiology = text_notes("a", prefix = "r")
  )
  for (table in c("discharge", "radiology")) note_tokens(con, table, spec_both())
  windows <- rbind(
    cbind(tokenizer = "slova", window_frame(0, 1)[-1]),
    cbind(tokenizer = "cele", window_frame(2, 510)[-1])
  )
  out <- window_summary(con, c("discharge", "radiology"), windows)

  expect_equal(nrow(out), 4)
  expect_setequal(names(out), c("tabulka", "tokenizer", "zprav", "v_okne", "casti"))
  expect_equal(out$v_okne[out$tabulka == "discharge" & out$tokenizer == "slova"], 1)
})

test_that("window_summary prázdné tabulky vrátí nuly", {
  con <- mimic_con()
  note_tokens(con, "discharge", spec_both())
  out <- window_summary(con, "discharge", window_frame())

  expect_equal(c(out$zprav, out$v_okne, out$casti), c(0, 0, 0))
})

test_that("encoder_models vrací modely s pevnou revizí a oknem 512", {
  out <- encoder_models()

  expect_equal(out$klic, c("xlmr", "mdeberta"))
  expect_true(all(nchar(out$revize) == 40))
  expect_equal(out$okno, c(512, 512))
})

test_that("token_histogram dá koš 0 jedné zprávě a prázdné zprávy nekreslí", {
  texts <- c("", words_of(1), words_of(9), words_of(10), words_of(1000))
  con <- mimic_con(discharge = text_notes(texts))
  note_tokens(con, "discharge", list(slova = list(tok = fake_tokenizer, special = FALSE)))
  out <- token_histogram(con, "discharge", "slova")

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
  con <- mimic_con(
    discharge = text_notes(c("jedinecny-retezec-d-1", "jedinecny-retezec-d-2")),
    radiology = text_notes("jedinecny-retezec-r-1", prefix = "r")
  )
  tables <- c("discharge", "radiology")
  for (table in tables) note_tokens(con, table, spec_both())

  expect_no_error(token_percentiles(con, tables, c("slova", "cele")))
  expect_no_error(window_summary(con, tables, window_frame()))
  expect_no_error(token_histogram(con, tables, "cele"))
  expect_no_error(format_percentiles(token_percentiles(con, tables, "cele")))
})

test_that("translation_limit dává známé hodnoty a hranici zprávy, která se ještě vejde", {
  expect_equal(translation_limit(2048, 2.2, 75), 616.5625)
  expect_equal(translation_limit(100, 1, 0), 50)
})

test_that("spodní mez překladu na hranici limitu: 616 tokenů je jedna část, 617 dvě", {
  windows <- data.frame(
    tokenizer = "slova", specials = 0, capacity = translation_limit(2048, 2.2, 75), scale = 1,
    stringsAsFactors = FALSE
  )
  con <- mimic_con(discharge = text_notes(c(words_of(616), words_of(617))))
  note_tokens(con, "discharge", list(slova = list(tok = fake_tokenizer, special = FALSE)))
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

fake_settings <- function(split = function(text, fits) text) {
  list(
    model = "m", revision = "r", window = 2048L, split_ratio = 2.2,
    prepare = list(
      `_token_ids` = function(tok, text) seq_len(75L + length(strsplit(text, " ")[[1]])),
      `_fits` = function(tok, text) length(strsplit(text, " ")[[1]]) <= 3L
    ),
    note_parts = list(split = split)
  )
}

test_that("translation_overhead odečte od délky promptu délku zdroje bez speciálních tokenů", {
  tok <- function(texts, add_special_tokens = TRUE) {
    list(input_ids = lapply(texts, function(text) seq_len(length(strsplit(text, " ")[[1]]))))
  }

  expect_equal(translation_overhead(fake_settings(), tok), 75L)
})

test_that("translation_fits volá _fits z přípravy překladu", {
  fits <- translation_fits(fake_settings(), tok = NULL)

  expect_true(fits("a b c"))
  expect_false(fits("a b c d"))
})

unsplittable <- function(message) {
  stop(structure(
    class = c("note_parts.UnsplittableError", "python.builtin.ValueError", "error", "condition"),
    list(message = message, call = NULL)
  ))
}

test_that("translation_split vrátí NULL pro nedělitelnou zprávu a jinou chybu nechá projít", {
  fits <- function(text) TRUE
  nedelitelna <- translation_split(fake_settings(function(text, fits) unsplittable("řádek")), fits)
  jina <- translation_split(fake_settings(function(text, fits) stop("jiná chyba")), fits)

  expect_null(nedelitelna("a"))
  expect_error(jina("a"), "jiná chyba")
  expect_equal(translation_split(fake_settings(), fits)("a b"), "a b")
})

test_that("translation_settings čte konstanty ze skutečné přípravy překladu bez stahování modelu", {
  settings <- translation_settings(Sys.getenv("PIXI_PROJECT_ROOT"))

  expect_equal(settings$window, 2048)
  expect_equal(settings$split_ratio, 2.2)
  expect_equal(settings$model, "google/translategemma-27b-it")
  expect_equal(nchar(settings$revision), 40)
  expect_true(is.function(settings$prepare$`_fits`))
  expect_true(is.function(settings$note_parts$split))
})

chunk_split <- function(size) {
  function(text) {
    words <- strsplit(text, " ")[[1]]
    if (length(words) <= 4L) {
      return(text)
    }
    parts <- split(words, ceiling(seq_along(words) / size))
    vapply(parts, paste, character(1), collapse = " ", USE.NAMES = FALSE)
  }
}

word_count <- function(text) length(strsplit(text, " ")[[1]])

sample_of <- function(texts, split = chunk_split(3), limit = 4, n = 1000L) {
  con <- mimic_con(discharge = if (length(texts) > 0L) text_notes(texts))
  split_sample(
    con, "discharge",
    fits = function(text) word_count(text) <= 4L,
    split = split, length_of = word_count, limit = limit, n = n
  )
}

test_that("sample_notes vybírá deterministicky a menší tabulka vrátí všechno", {
  con <- mimic_con(discharge = text_notes(vapply(1:20, words_of, character(1))))

  first <- sample_notes(con, "discharge", 5L)
  second <- sample_notes(con, "discharge", 5L)

  expect_equal(first$note_id, second$note_id)
  expect_equal(nrow(first), 5)
  expect_equal(nrow(sample_notes(con, "discharge", 100L)), 20)
})

test_that("sample_notes vynechá prázdné a chybějící texty", {
  con <- mimic_con(discharge = text_notes(c("", NA, "a")))

  expect_equal(sample_notes(con, "discharge", 10L)$text, "a")
})

test_that("split_sample porovná přesné dělení se spodní mezí", {
  out <- sample_of(c(words_of(3), words_of(8)))

  expect_equal(out$zprav, 2)
  expect_equal(out$deleno, 1)
  expect_equal(out$casti_presne, 1 + 3)
  expect_equal(out$casti_dolni, 1 + 2)
  expect_equal(out$pomer, 4 / 3)
  expect_equal(out$neshod, 0)
  expect_equal(out$nedelitelnych, 0)
})

test_that("split_sample spočítá zprávy, u nichž se vzorec neshoduje s fits", {
  out <- sample_of(c(words_of(4), words_of(3)), limit = 3.5)

  expect_equal(out$neshod, 1)
})

test_that("split_sample spočítá nedělitelné zprávy zvlášť a do částí je nezahrne", {
  split <- function(text) if (word_count(text) > 6L) NULL else chunk_split(3)(text)
  out <- sample_of(c(words_of(8), words_of(6), words_of(2)), split = split)

  expect_equal(out$nedelitelnych, 1)
  expect_equal(out$casti_presne, 2 + 1)
  expect_equal(out$casti_dolni, 2 + 1)
})

test_that("split_sample vezme jen n zpráv a pro prázdný zdroj vrátí nuly bez poměru", {
  expect_equal(sample_of(vapply(1:10, words_of, character(1)), n = 3L)$zprav, 3)

  out <- sample_of(character())
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

parallel_dirs <- function(en, cs, env = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = env)
  dirs <- c(file.path(root, "en"), file.path(root, "cs"))
  lapply(dirs, dir.create)
  for (name in names(en)) writeLines(en[[name]], file.path(dirs[1], paste0(name, ".txt")))
  for (name in names(cs)) writeLines(cs[[name]], file.path(dirs[2], paste0(name, ".txt")))
  dirs
}

test_that("read_parallel páruje soubory podle jména nezávisle na pořadí výpisu", {
  dirs <- parallel_dirs(
    list(b = "two words", a = "one"), list(a = "jedno slovo", b = "dva slova tady")
  )
  out <- read_parallel(dirs[1], dirs[2])

  expect_equal(out$id, c("a", "b"))
  expect_equal(trimws(out$en), c("one", "two words"))
  expect_equal(trimws(out$cs), c("jedno slovo", "dva slova tady"))
})

test_that("read_parallel při chybějícím protějšku vyvolá chybu se jménem souboru", {
  dirs <- parallel_dirs(list(a = "x", b = "y"), list(a = "x"))

  expect_error(read_parallel(dirs[1], dirs[2]), "b")
  expect_error(read_parallel(dirs[2], dirs[1]), "b")
})

test_that("read_parallel čte UTF-8 a prázdný adresář je chyba", {
  dirs <- parallel_dirs(list(a = "x"), list(a = "příliš žluťoučký kůň"))

  expect_equal(trimws(read_parallel(dirs[1], dirs[2])$cs), "příliš žluťoučký kůň")
  empty <- parallel_dirs(list(), list())
  expect_error(read_parallel(empty[1], empty[2]), "žádné")
})

parallel_pairs <- function() {
  data.frame(
    id = c("a", "b", "c", "d"),
    en = c(words_of(2), words_of(4), words_of(10), words_of(20)),
    cs = c(words_of(3), words_of(6), words_of(15), words_of(30)),
    stringsAsFactors = FALSE
  )
}

test_that("parallel_ratio spočítá medián a 95. percentil poměru CS/EN", {
  spec <- list(slova = list(tok = fake_tokenizer, special = FALSE))
  windows <- token_windows(c(slova = 0), translation = numeric())
  windows$capacity <- 12
  out <- parallel_ratio(parallel_pairs(), spec, windows)

  expect_equal(out$tokenizer, "slova")
  expect_equal(out$zprav, 4)
  expect_equal(out$median, 1.5)
  expect_equal(out$p95, 1.5)
})

test_that("parallel_ratio spočítá podíl zpráv v okně v obou jazycích", {
  spec <- list(slova = list(tok = fake_tokenizer, special = FALSE))
  windows <- token_windows(c(slova = 0), translation = numeric())
  windows$capacity <- 12
  out <- parallel_ratio(parallel_pairs(), spec, windows)

  expect_equal(out$v_okne_en, 3)
  expect_equal(out$v_okne_cs, 2)
})

test_that("parallel_ratio ignoruje zprávy bez anglických tokenů při poměru", {
  pairs <- rbind(parallel_pairs(), data.frame(id = "e", en = "", cs = "x", stringsAsFactors = FALSE))
  spec <- list(slova = list(tok = fake_tokenizer, special = FALSE))
  windows <- token_windows(c(slova = 0), translation = numeric())
  out <- parallel_ratio(pairs, spec, windows)

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
