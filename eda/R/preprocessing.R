PREPROCESSING_TABLE <- "note_preprocessing_input"
ACRONYMS_TABLE <- "note_acronyms_input"
PREPROCESSING_COLUMNS <- c(
  note_id = "VARCHAR", tokens_raw = "INTEGER", tokens_preprocessed = "INTEGER",
  vocabulary_raw = "VARCHAR[]", vocabulary_preprocessed = "VARCHAR[]"
)
ACRONYM_COLUMNS <- c(note_id = "VARCHAR", acronym = "VARCHAR", occurrences = "INTEGER")
# Pořadí variant určuje pořadí řádků tabulky.
PREPROCESSING_VARIANTS <- c(raw = "tokens_raw", preprocessed = "tokens_preprocessed")
PREPROCESSING_LABELS <- c(raw = "Bez předzpracování", preprocessed = "S předzpracováním")
TOKEN_FORMS <- NGRAM_KINDS[["1"]]$forms
# Zkratka AF je víceznačná (afebrile), proto ji věta o zkratkách srovnává s ostatními.
AMBIGUOUS_ACRONYM <- "AF"

PREPROCESSING_LAYOUTS <- list(
  `tbl-predzpracovani` = list(
    col.names = c("Varianta", "Slovník (tokeny)", "Medián tokenů na zprávu", "p95 tokenů", "Průměr tokenů"),
    align = "lrrrr", widths = c(28, 18, 20, 16, 18)
  ),
  `tbl-zkratky` = list(
    col.names = c("Zkratka", "Diagnóza", "Výskyty", "Zprávy bez kódu diagnózy (%)"),
    align = "llrr", widths = c(16, 38, 18, 28)
  )
)

corpus_acronyms <- function(con) {
  source <- feature_source(con, "note_acronyms.parquet", ACRONYM_COLUMNS)
  DBI::dbGetQuery(con, paste("SELECT DISTINCT acronym FROM", source, "ORDER BY acronym"))$acronym
}

note_preprocessing <- function(con) {
  feature_table(con, PREPROCESSING_TABLE, "note_preprocessing.parquet", PREPROCESSING_COLUMNS)
  feature_table(con, ACRONYMS_TABLE, "note_acronyms.parquet", ACRONYM_COLUMNS)
  PREPROCESSING_TABLE
}

preprocessing_summary <- function(con) {
  selects <- mapply(percentile_sql, names(PREPROCESSING_VARIANTS), unname(PREPROCESSING_VARIANTS), USE.NAMES = FALSE)
  tokens <- DBI::dbGetQuery(con, paste(sprintf(selects, PREPROCESSING_TABLE), collapse = " UNION ALL "))
  tokens <- as_numeric_columns(tokens, c("n", "prumer", "minimum", "maximum", PERCENTILE_COLUMNS))
  tokens <- tokens[match(names(PREPROCESSING_VARIANTS), tokens$metrika), , drop = FALSE]
  vocabulary <- DBI::dbGetQuery(con, sprintf("
    SELECT 'raw' AS variant, count(DISTINCT token) AS slovnik
    FROM (SELECT unnest(vocabulary_raw) AS token FROM %1$s)
    UNION ALL
    SELECT 'preprocessed' AS variant, count(DISTINCT token) AS slovnik
    FROM (SELECT unnest(vocabulary_preprocessed) AS token FROM %1$s)
  ", PREPROCESSING_TABLE))
  slovnik <- as.numeric(vocabulary$slovnik[match(names(PREPROCESSING_VARIANTS), vocabulary$variant)])
  slovnik[is.na(slovnik)] <- 0
  out <- cbind(varianta = names(PREPROCESSING_VARIANTS), slovnik = slovnik, tokens[setdiff(names(tokens), "metrika")])
  rownames(out) <- NULL
  out
}

preprocessing_table <- function(summary) {
  formatted <- format_percentiles(summary)
  data.frame(
    varianta = unname(PREPROCESSING_LABELS[summary$varianta]),
    slovnik = number_text(summary$slovnik),
    median = formatted$p50,
    p95 = formatted$p95,
    prumer = formatted$prumer,
    stringsAsFactors = FALSE
  )
}

preprocessing_text <- function(summary) {
  raw <- summary[summary$varianta == "raw", ]
  prepared <- summary[summary$varianta == "preprocessed", ]
  if (raw$n == 0) {
    return(paste("Dopad předzpracování na slovník a délku zprávy", UNDECIDABLE_TEXT))
  }
  change <- if (raw$slovnik > 0) {
    sprintf(" (%s)", percent_change_text(prepared$slovnik, raw$slovnik))
  } else {
    ""
  }
  sprintf(
    "Slovník analyzéru před prořezáním má bez předzpracování %s, s předzpracováním %s%s. Zpráva má bez předzpracování v mediánu %s (p95 %s), s předzpracováním %s (p95 %s).",
    count_phrase(raw$slovnik, TOKEN_FORMS), count_phrase(prepared$slovnik, TOKEN_FORMS), change,
    count_phrase(raw$p50, TOKEN_FORMS), number_text(raw$p95),
    count_phrase(prepared$p50, TOKEN_FORMS), number_text(prepared$p95)
  )
}

percent_change_text <- function(after, before) {
  change <- 100 * (after - before) / before
  value <- dua_decimal(abs(change))
  if (value == dua_decimal(0)) {
    return("beze změny")
  }
  paste0(if (change < 0) "pokles o " else "nárůst o ", value, NBSP, "%")
}

acronym_cells <- function(con, acronym, key) {
  cells <- DBI::dbGetQuery(con, sprintf("
    SELECT
      count(*) FILTER (WHERE has_acronym AND has_code) AS oba,
      count(*) FILTER (WHERE has_acronym AND NOT has_code) AS jen_zkratka,
      count(*) FILTER (WHERE NOT has_acronym AND has_code) AS jen_kod,
      count(*) FILTER (WHERE NOT has_acronym AND NOT has_code) AS zadne
    FROM (
      SELECT p.occurrences > 0 AS has_acronym,
        EXISTS (
          SELECT 1 FROM %2$s c WHERE c.kandidat = %3$s AND c.note_id = p.note_id
        ) AS has_code
      FROM %4$s p WHERE p.acronym = %1$s
    )
  ", DBI::dbQuoteString(con, acronym), CANDIDATES_TABLE, DBI::dbQuoteString(con, key), ACRONYMS_TABLE))
  vapply(cells, as.numeric, numeric(1))
}

acronym_owner <- function(acronym, candidates) {
  owners <- names(candidates)[vapply(candidates, function(entry) acronym %in% entry$acronyms_en, logical(1))]
  if (length(owners) != 1L) {
    stop(sprintf("Zkratka %s musí patřit právě jedné diagnóze v configu, patří %d.", acronym, length(owners)))
  }
  owners
}

acronym_summary <- function(con, acronyms, candidates) {
  candidate_notes(con, candidates)
  rows <- lapply(acronyms, function(acronym) {
    key <- acronym_owner(acronym, candidates)
    cells <- acronym_cells(con, acronym, key)
    occurrences <- as.numeric(DBI::dbGetQuery(con, sprintf(
      "SELECT coalesce(sum(occurrences), 0) AS n FROM %s WHERE acronym = %s", ACRONYMS_TABLE, DBI::dbQuoteString(con, acronym)
    ))$n)
    with_acronym <- cells[["oba"]] + cells[["jen_zkratka"]]
    data.frame(
      zkratka = acronym,
      kandidat = key,
      label_cs = candidates[[key]]$label_cs,
      vyskyty = occurrences,
      podil = if (with_acronym > 0) cells[["jen_zkratka"]] / with_acronym else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  if (length(rows) == 0L) {
    return(data.frame(
      zkratka = character(), kandidat = character(), label_cs = character(),
      vyskyty = numeric(), podil = numeric(), stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

acronym_table <- function(summary) {
  data.frame(
    zkratka = summary$zkratka,
    label_cs = summary$label_cs,
    vyskyty = number_text(summary$vyskyty),
    bez_kodu = percent_text(summary$podil, 1),
    stringsAsFactors = FALSE
  )
}

acronym_text <- function(summary) {
  if (nrow(summary) == 0L) {
    return(paste("Podíl zpráv se zkratkou bez kódu diagnózy", UNDECIDABLE_TEXT))
  }
  focus <- summary[summary$zkratka == AMBIGUOUS_ACRONYM, ]
  others <- summary$podil[summary$zkratka != AMBIGUOUS_ACRONYM & !is.na(summary$podil)]
  if (nrow(focus) == 0L || is.na(focus$podil)) {
    return(sprintf(
      "Podíl zpráv se zkratkou %s bez kódu diagnózy nelze posoudit: zkratka nemá výskyt.",
      AMBIGUOUS_ACRONYM
    ))
  }
  if (length(others) == 0L) {
    return(sprintf(
      "Zpráv se zkratkou %s je bez kódu diagnózy %s; ostatní zkratky nemají výskyt.",
      AMBIGUOUS_ACRONYM, percent_label(focus$podil, 1)
    ))
  }
  sprintf(
    "Zpráv se zkratkou %s je bez kódu diagnózy %s, u ostatních zkratek v mediánu %s.",
    AMBIGUOUS_ACRONYM, percent_label(focus$podil, 1), percent_label(stats::median(others), 1)
  )
}
