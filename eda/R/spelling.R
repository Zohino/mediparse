VARIANTS_TABLE <- "note_variants_input"
VARIANT_COLUMNS <- c(note_id = "VARCHAR", variant = "VARCHAR", diagnosis = "VARCHAR", occurrences = "INTEGER")
TOKEN_FREQUENCY_TABLE <- "token_frequency"
TARGET_MIN_LENGTH <- 6L
NEAR_MISS_RATIO <- 10L
NEAR_MISS_LIMIT <- 40L
TYPO_MIN_LENGTH <- 5L
TYPO_RATIO <- 100L

SPELLING_LAYOUTS <- list(
  `tbl-varianty` = list(
    col.names = c("Varianta", "Diagnóza", "Výskyty", "Zprávy bez kódu diagnózy (%)"),
    align = "llrr", widths = c(30, 30, 14, 26)
  ),
  `tbl-blizke-tvary` = list(
    col.names = c("Tvar", "Cíl", "Zprávy s tvarem", "Zprávy s cílem"),
    align = "llrr", widths = c(30, 30, 20, 20)
  ),
  `tbl-preklepy` = list(
    col.names = c("Slova ve slovníku", "Kandidáti", "Podíl slovníku (%)", "Podíl výskytů ve zprávách (%)"),
    align = "rrrr", widths = c(22, 18, 28, 32)
  )
)

note_variants <- function(con) {
  feature_table(con, VARIANTS_TABLE, "note_variants.parquet", VARIANT_COLUMNS)
}

variant_summary <- function(con) {
  note_variants(con)
  candidates <- read_candidates(candidates_path())
  candidate_notes(con, candidates)
  terms <- DBI::dbGetQuery(con, sprintf(
    "SELECT variant, diagnosis, sum(occurrences) AS vyskyty FROM %s GROUP BY variant, diagnosis ORDER BY variant",
    VARIANTS_TABLE
  ))
  terms$label_cs <- candidate_labels(terms$diagnosis, candidates)
  terms$vyskyty <- as.numeric(terms$vyskyty)
  terms$podil <- vapply(seq_len(nrow(terms)), function(i) {
    cells <- term_cells(con, VARIANTS_TABLE, "variant", terms$variant[[i]], terms$diagnosis[[i]])
    denominator <- cells[["oba"]] + cells[["jen_termin"]]
    if (denominator > 0) cells[["jen_termin"]] / denominator else NA_real_
  }, numeric(1))
  terms
}

variant_table <- function(summary) {
  data.frame(
    varianta = summary$variant, diagnoza = summary$label_cs,
    vyskyty = number_text(summary$vyskyty), bez_kodu = percent_text(summary$podil, 1),
    stringsAsFactors = FALSE
  )
}

variant_text <- function(summary) {
  if (nrow(summary) == 0L || !any(summary$vyskyty > 0)) {
    return(paste("Výskyt variant názvů diagnóz", UNDECIDABLE_TEXT))
  }
  sprintf(
    "Z předem daného seznamu se vyskytuje %s. Tabulka uvádí přesné výskyty a podíl zpráv s variantou bez kódu příslušné diagnózy.",
    count_phrase(sum(summary$vyskyty > 0), c("varianta", "varianty", "variant"))
  )
}

token_frequency <- function(con) {
  if (TOKEN_FREQUENCY_TABLE %in% DBI::dbListTables(con)) return(TOKEN_FREQUENCY_TABLE)
  note_preprocessing(con)
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    SELECT token, count(*) AS df
    FROM (SELECT unnest(vocabulary_raw) AS token FROM %s)
    GROUP BY token
  ", TOKEN_FREQUENCY_TABLE, PREPROCESSING_TABLE))
  TOKEN_FREQUENCY_TABLE
}

spelling_targets <- function(candidates, keys) {
  words <- lapply(candidates[keys], function(entry) {
    tokens <- unlist(strsplit(tolower(entry$keywords_en), "[^a-z]+"), use.names = FALSE)
    unique(tokens[grepl("^[a-z]+$", tokens) & nchar(tokens) >= TARGET_MIN_LENGTH])
  })
  tokens <- as.character(unlist(words, use.names = FALSE))
  counts <- table(tokens)
  as.character(sort(names(counts)[counts == 1L]))
}

near_miss_forms <- function(con, targets) {
  table <- token_frequency(con)
  if (length(targets) == 0L) {
    return(data.frame(tvar = character(), cil = character(), df = numeric(), cil_df = numeric()))
  }
  quoted <- paste(DBI::dbQuoteString(con, targets), collapse = ", ")
  out <- DBI::dbGetQuery(con, sprintf("
    SELECT r.token AS tvar, f.token AS cil, r.df, f.df AS cil_df
    FROM %1$s r JOIN %1$s f ON abs(length(r.token) - length(f.token)) <= 1
    WHERE f.token IN (%2$s) AND r.token NOT IN (%2$s)
      AND r.df >= %3$d AND f.df >= %4$d * r.df
      AND damerau_levenshtein(r.token, f.token) = 1
    ORDER BY r.df DESC, r.token, f.token
  ", table, quoted, LABEL_MIN_NOTES, NEAR_MISS_RATIO))
  as_numeric_columns(out, c("df", "cil_df"))
}

near_miss_table <- function(forms) {
  rows <- utils::head(forms, NEAR_MISS_LIMIT)
  data.frame(
    tvar = rows$tvar, cil = rows$cil,
    zpravy = number_text(rows$df), cil_zpravy = number_text(rows$cil_df),
    stringsAsFactors = FALSE
  )
}

near_miss_text <- function(forms) {
  if (nrow(forms) == 0L) {
    return(paste("Blízké tvary názvů diagnóz nad zvolenými prahy", UNDECIDABLE_TEXT))
  }
  sprintf(
    "Podmínky splňuje %s; tabulka ukazuje nejvýše %s podle dokumentové frekvence.",
    count_phrase(length(unique(forms$tvar)), c("tvar", "tvary", "tvarů")),
    count_phrase(NEAR_MISS_LIMIT, c("dvojice", "dvojice", "dvojic"))
  )
}

typo_estimate <- function(con) {
  table <- token_frequency(con)
  # Společné okolí po vypuštění znaku omezuje páry před ověřením vzdálenosti;
  # křížové spojení v sondě briefu vycházelo na desítky minut.
  out <- DBI::dbGetQuery(con, sprintf("
    WITH words AS (
      SELECT token, df FROM %1$s WHERE regexp_full_match(token, '[a-z]+') AND length(token) >= %2$d
    ), deletions AS (
      SELECT token, df, token AS d FROM words
      UNION ALL
      SELECT token, df, token[:i-1] || token[i+1:] AS d
      FROM words, LATERAL (SELECT unnest(range(1, length(token)+1)) AS i)
    ), candidates AS (
      SELECT DISTINCT r.token
      FROM deletions r JOIN deletions f USING (d)
      WHERE f.df >= %3$d * r.df
        AND damerau_levenshtein(r.token, f.token) = 1
    )
    SELECT count(*) AS slovnik, count(c.token) AS kandidati,
      coalesce(sum(w.df), 0) AS vyskyty,
      coalesce(sum(w.df) FILTER (WHERE c.token IS NOT NULL), 0) AS vyskyty_kandidatu
    FROM words w LEFT JOIN candidates c USING (token)
  ", table, TYPO_MIN_LENGTH, TYPO_RATIO))
  as_numeric_columns(out, c("slovnik", "kandidati", "vyskyty", "vyskyty_kandidatu"))
}

typo_table <- function(estimate) {
  data.frame(
    slovnik = number_text(estimate$slovnik), kandidati = number_text(estimate$kandidati),
    podil_slovniku = percent_text(estimate$kandidati, estimate$slovnik),
    podil_vyskytu = percent_text(estimate$vyskyty_kandidatu, estimate$vyskyty),
    stringsAsFactors = FALSE
  )
}

typo_text <- function(estimate) {
  if (estimate$slovnik == 0) {
    return(paste("Podíl kandidátů na překlep ve slovníku", UNDECIDABLE_TEXT))
  }
  sprintf(
    "Z %s splňuje definici kandidáta %s (%s slovníku, %s součtu dokumentových frekvencí). Jde o popisný odhad podle definice. Může zahrnovat legitimní slova a vynechává překlepy ve vzdálenosti větší než 1, tokeny kratší než 5 nebo s číslicí a překlepy s poměrem DF pod stanoveným prahem.",
    count_phrase(estimate$slovnik, c("slova", "slov", "slov")),
    count_phrase(estimate$kandidati, c("slovo", "slova", "slov")),
    percent_label(estimate$kandidati, estimate$slovnik),
    percent_label(estimate$vyskyty_kandidatu, estimate$vyskyty)
  )
}
