# Token je maximální úsek písmen a-z délky aspoň 2 po převodu na malá písmena (jako `token_pattern`
# sklearn): maska `___`, čísla, časy a dávkování vypadnou. Pravidlo žije jen tady.
TOKEN_SQL <- "regexp_extract_all(lower(%s), '[a-z]{2,}')"
NGRAM_SIZES <- 1:3
NGRAM_RANKING <- 15L
SECTION_NGRAMS <- 5L
# Fráze šablony je n-gram, který má aspoň polovina zpráv; vlastní konstanta, protože má jiný důvod
# změny než CANDIDATE_COVERAGE (kandidáti na lexikon) a SKELETON_COVERAGE (sestavení kostry).
BOILERPLATE_COVERAGE <- 0.5
STOP_TABLE <- "ngram_stop_input"
TOKENS_TABLE <- "ngram_tokens_input"
NGRAM_COLUMNS <- c("sekce", "n", "ngram", "zpravy", "vyskyty", "jmenovatel")

# Stejná stop slova jako `TfidfVectorizer(stop_words = "english")`, tedy i negace "no" a "not".
stop_words <- function() {
  frozen <- reticulate::import("sklearn.feature_extraction.text", convert = FALSE)$ENGLISH_STOP_WORDS
  as.character(unlist(reticulate::import_builtins()$sorted(frozen)))
}

empty_counts <- function() {
  data.frame(
    sekce = character(), n = integer(), ngram = character(),
    zpravy = numeric(), vyskyty = numeric(), jmenovatel = numeric(),
    stringsAsFactors = FALSE
  )
}

# Text těla vyřízne z `discharge` podle offsetů z parseru (code pointy, jako `substring`), takže
# pravidlo hranice sekce zůstává jen v parseru. Stop slova se odfiltrují před skládáním n-gramů,
# stejně jako v `TfidfVectorizer`; `ANTI JOIN` je proti `list_filter` s literálem čtyřikrát rychlejší.
# Tabulka tokenů má jeden řádek na část zprávy a seznam slov v pořadí textu.
ngram_tokens <- function(con, stopwords, sections) {
  write_temp(con, STOP_TABLE, data.frame(w = stopwords, stringsAsFactors = FALSE))
  scope <- if (is.null(sections)) {
    ""
  } else {
    sprintf("WHERE s.section IN (%s)", paste(DBI::dbQuoteString(con, sections), collapse = ", "))
  }
  key <- if (is.null(sections)) "NULL::VARCHAR" else "s.section"
  DBI::dbExecute(con, sprintf("
    CREATE OR REPLACE TEMP TABLE %1$s AS
    WITH bodies AS (
      SELECT s.note_id, s.position, %2$s AS sekce,
        %3$s AS words
      FROM %4$s s JOIN discharge d ON d.note_id = s.note_id
      %5$s
    ),
    words AS (
      SELECT note_id, position, sekce, unnest(words) AS w, generate_subscripts(words, 1) AS i
      FROM bodies
    )
    SELECT note_id, position, sekce, list(w ORDER BY i) AS t
    FROM words ANTI JOIN %6$s USING (w)
    GROUP BY note_id, position, sekce
  ", TOKENS_TABLE, key, sprintf(
    TOKEN_SQL, "substring(d.text, s.body_start + 1, s.body_end - s.body_start)"
  ), SECTIONS_TABLE, scope, STOP_TABLE))
  TOKENS_TABLE
}

# Jmenovatel brány: u celku zprávy vstupního korpusu (`segment_corpus` je plní z něj), u sekce
# zprávy, které ji mají.
ngram_denominators <- function(con, sections) {
  if (is.null(sections)) {
    total <- DBI::dbGetQuery(con, sprintf("SELECT count(DISTINCT note_id) AS n FROM %s", SECTIONS_TABLE))$n
    return(data.frame(sekce = NA_character_, jmenovatel = as.numeric(total), stringsAsFactors = FALSE))
  }
  out <- DBI::dbGetQuery(con, sprintf("
    SELECT section AS sekce, count(DISTINCT note_id) AS jmenovatel FROM %s
    WHERE section IN (%s) GROUP BY section
  ", SECTIONS_TABLE, paste(DBI::dbQuoteString(con, sections), collapse = ", ")))
  out$jmenovatel <- as.numeric(out$jmenovatel)
  out
}

ngram_list_sql <- function(size) {
  if (size == 1L) {
    return("t")
  }
  parts <- sprintf("t[i + %d]", seq_len(size) - 1L)
  sprintf("[concat_ws(' ', %s) FOR i IN range(1, len(t) - %d + 2)]", paste(parts, collapse = ", "), size)
}

# Brána DUA je v SQL, takže do R jde jen to, co se smí zveřejnit. `sections = NULL` je celek
# (dokumentová frekvence přes všechny části zprávy), jinak se počítá po zadaných sekcích.
ngram_counts <- function(con, stopwords, sections = NULL, floor_fn = publishable_floor) {
  tokens <- ngram_tokens(con, stopwords, sections)
  on.exit(DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", tokens)), add = TRUE)
  denominators <- ngram_denominators(con, sections)
  denominators$prah <- as.numeric(floor_fn(denominators$jmenovatel))
  limits <- write_temp(con, "ngram_floor_input", denominators)
  per_size <- lapply(NGRAM_SIZES, function(size) {
    DBI::dbGetQuery(con, sprintf("
      WITH grams AS (
        SELECT note_id, sekce, unnest(%2$s) AS ngram FROM %1$s
      ),
      per_note AS (
        SELECT note_id, sekce, ngram, count(*) AS vyskyty FROM grams GROUP BY ALL
      )
      SELECT p.sekce, %3$d AS n, p.ngram, count(*) AS zpravy, sum(p.vyskyty) AS vyskyty,
        any_value(f.jmenovatel) AS jmenovatel
      FROM per_note p JOIN %4$s f ON f.sekce IS NOT DISTINCT FROM p.sekce
      GROUP BY p.sekce, p.ngram
      HAVING count(*) >= any_value(f.prah)
    ", tokens, ngram_list_sql(size), size, limits))
  })
  out <- do.call(rbind, c(list(empty_counts()), per_size))
  out$n <- as.integer(out$n)
  out[, NGRAM_COLUMNS]
}

# Řazení jako `ranked_headers` (počet sestupně, pak řetězec), navíc výskyty; `radix` řadí řetězce
# bez ohledu na locale, aby pořadí nezáviselo na stroji.
ngram_ranking <- function(counts, limit) {
  ordered <- counts[order(counts$sekce, counts$n, -counts$zpravy, -counts$vyskyty, counts$ngram, method = "radix"), ]
  rank <- stats::ave(
    seq_len(nrow(ordered)), ordered$sekce, ordered$n, FUN = seq_along
  )
  rownames(ordered) <- NULL
  ordered[rank <= limit, , drop = FALSE]
}
