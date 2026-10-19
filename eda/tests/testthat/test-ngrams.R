# Zprávy rozdělí skutečný parser, takže offsety v tabulce odpovídají textu; `discharge` dodá mimic_con.
ngram_con <- function(texts, env = parent.frame()) {
  notes <- if (length(texts) == 0L) NULL else text_notes(texts)
  con <- mimic_con(discharge = notes, env = env)
  chunk <- if (is.null(notes)) {
    list(note_id = character(), text = character())
  } else {
    list(note_id = notes$note_id, text = notes$text)
  }
  frame <- section_rows(sections_module(), section_headers_path(), chunk)
  write_temp(con, SECTIONS_TABLE, frame)
  con
}

no_floor <- function(total) 1
counts_of <- function(texts, sections = NULL, stopwords = character(), floor = no_floor) {
  # Spojení vzniká tady, ne jako líný argument: parent.frame() by pak ukazoval na cizí rámec.
  con <- ngram_con(texts)
  ngram_counts(con, stopwords, sections = sections, floor_fn = floor)
}
grams <- function(counts, size) counts$ngram[counts$n == size]

test_that("token jsou malá písmena délky aspoň 2, maska, čísla a jednopísmenné úseky vypadnou", {
  counts <- counts_of("Allergies:\nPain ___ 120/80 at 10:30 a X-ray b\n")

  expect_setequal(grams(counts, 1L), c("pain", "at", "ray"))
})

test_that("stop slova vypadnou před skládáním n-gramů", {
  counts <- counts_of("Chief Complaint:\nshortness of breath\n", stopwords = c("of"))

  expect_setequal(grams(counts, 2L), "shortness breath")
  expect_setequal(grams(counts, 1L), c("shortness", "breath"))
  expect_length(grams(counts, 3L), 0L)
})

test_that("n-gram nepřejde přes hranici sekcí a hlavičky se nepočítají", {
  counts <- counts_of("Allergies:\nnone known\n\nChief Complaint:\nfever chills\n")

  expect_setequal(grams(counts, 1L), c("none", "known", "fever", "chills"))
  expect_setequal(grams(counts, 2L), c("none known", "fever chills"))
})

test_that("zprávy jsou dokumentová frekvence, výskyty jsou počet v textu", {
  counts <- counts_of(c(
    "Allergies:\nrash rash rash\n",
    "Allergies:\nrash\n"
  ))
  row <- counts[counts$n == 1L & counts$ngram == "rash", ]

  expect_equal(row$zpravy, 2)
  expect_equal(row$vyskyty, 4)
  expect_equal(row$jmenovatel, 2)
})

test_that("n-gram ve dvou sekcích téže zprávy je pro celek jedna zpráva", {
  counts <- counts_of("Allergies:\nrash\n\nChief Complaint:\nrash\n")
  row <- counts[counts$ngram == "rash", ]

  expect_equal(row$zpravy, 1)
  expect_equal(row$vyskyty, 2)
})

test_that("brána vynechá n-gram pod publishable_floor i s mnoha výskyty v jedné zprávě", {
  common <- rep("Allergies:\nrash itch\n", DUA_THRESHOLD)
  texts <- c(common, "Allergies:\nsecret secret secret secret secret secret\n")
  counts <- counts_of(texts, floor = publishable_floor)

  expect_true("rash" %in% grams(counts, 1L))
  expect_false("secret" %in% grams(counts, 1L))
})

test_that("po sekcích jsou jen zadané sekce a jmenovatel je počet zpráv se sekcí", {
  counts <- counts_of(
    c("Allergies:\nrash\n", "Allergies:\nrash\n\nChief Complaint:\nrash\n", "Name: rash\n"),
    sections = c("allergies", "chief_complaint")
  )
  allergies <- counts[counts$sekce == "allergies" & counts$ngram == "rash", ]

  expect_setequal(counts$sekce, c("allergies", "chief_complaint"))
  expect_equal(allergies$zpravy, 2)
  expect_equal(allergies$jmenovatel, 2)
  expect_equal(counts$jmenovatel[counts$sekce == "chief_complaint"], 1)
})

test_that("celek počítá i text other a má sekci NA", {
  counts <- counts_of("Name: rash\n\nAllergies:\nitch\n")

  expect_true(all(is.na(counts$sekce)))
  expect_setequal(grams(counts, 1L), c("name", "rash", "itch"))
  expect_equal(unique(counts$jmenovatel), 1)
})

test_that("jmenovatel brány je počet zpráv: n-gram na prahu projde, pod prahem ne", {
  texts <- c(rep("Allergies:\nrash\n", 10L), rep("Allergies:\nitch\n", 9L), rep("Allergies:\nfever\n", 981L))
  counts <- counts_of(texts, floor = publishable_floor)

  expect_setequal(grams(counts, 1L), c("rash", "fever"))
})

test_that("ngram_ranking řadí zprávy, výskyty a řetězec a bere limit na sekci a n", {
  counts <- data.frame(
    sekce = NA_character_, n = 1L,
    ngram = c("b", "a", "c", "d"), zpravy = c(5, 5, 5, 9), vyskyty = c(7, 7, 9, 1),
    jmenovatel = 10, stringsAsFactors = FALSE
  )
  ranked <- ngram_ranking(counts[sample(4), ], limit = 3L)

  expect_equal(ranked$ngram, c("d", "c", "a"))
})

test_that("stop_words vrátí 318 slov sklearn včetně negací", {
  words <- stop_words()

  expect_type(words, "character")
  expect_length(words, 318L)
  expect_true(all(c("no", "not") %in% words))
})

test_that("ngram_counts nad prázdnou tabulkou vrátí typovaný prázdný data.frame", {
  counts <- counts_of(character())

  expect_equal(names(counts), c("sekce", "n", "ngram", "zpravy", "vyskyty", "jmenovatel"))
  expect_equal(nrow(counts), 0L)
})
