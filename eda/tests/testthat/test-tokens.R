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
