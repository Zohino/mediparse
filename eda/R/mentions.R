text_mentions_path <- function() config_path("text_mentions.json")

read_text_mentions <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  list(
    negation = list(
      cues = as.character(unlist(raw$negation$cues)),
      window_chars = as.integer(raw$negation$window_chars)
    ),
    symptoms = list(keywords_en = as.character(unlist(raw$symptoms$keywords_en)))
  )
}

FLAGS_TABLE <- "mention_flags_input"
AGREEMENT_COUNTS <- c("icd", "zminka", "tvrzeni", "icd_zminka", "icd_tvrzeni")

phrase_escape <- function(x) {
  gsub(" ", "\\s+", re2_escape(x), fixed = TRUE)
}

re2_escape <- function(x) {
  gsub("([.+*?(){}|^$\\\\\\[\\]])", "\\\\\\1", x, perl = TRUE)
}

mention_pattern <- function(keywords, acronyms) {
  alternatives <- c(
    if (length(keywords) > 0L) sprintf("(?i:%s)", paste(phrase_escape(keywords), collapse = "|")),
    re2_escape(acronyms)
  )
  if (length(alternatives) == 0L) {
    stop("Zmínka potřebuje aspoň jedno klíčové slovo nebo zkratku.")
  }
  sprintf("\\b(?:%s)\\b", paste(alternatives, collapse = "|"))
}

negated_pattern <- function(cues, window, mention) {
  sprintf(
    "(?i:\\b(?:%s)\\b)[^.;\\n]{0,%d}%s",
    paste(phrase_escape(cues), collapse = "|"), as.integer(window), mention
  )
}

mention_terms <- function(candidates, mentions) {
  window <- mentions$negation$window_chars
  cues <- mentions$negation$cues
  term <- function(column, mention) {
    list(
      column = column, mention = mention,
      negated = negated_pattern(cues, window, mention)
    )
  }
  candidate_terms <- lapply(seq_along(candidates), function(index) {
    entry <- candidates[[index]]
    term(
      sprintf("k%03d", index), mention_pattern(entry$keywords_en, entry$acronyms_en)
    )
  })
  symptom_terms <- lapply(seq_along(mentions$symptoms$keywords_en), function(index) {
    term(
      sprintf("p%03d", index),
      mention_pattern(mentions$symptoms$keywords_en[[index]], character())
    )
  })
  c(candidate_terms, symptom_terms)
}

mention_flags <- function(con, candidates, mentions) {
  if (FLAGS_TABLE %in% DBI::dbListTables(con)) {
    return(FLAGS_TABLE)
  }
  quote <- function(x) DBI::dbQuoteString(con, x)
  terms <- mention_terms(candidates, mentions)
  found <- vapply(terms, function(term) {
    sprintf("coalesce(regexp_matches(d.text, %s), false) AS %s_m", quote(term$mention), term$column)
  }, character(1))
  asserted <- vapply(terms, function(term) {
    sprintf(
      "CASE WHEN %s_m THEN regexp_matches(regexp_replace(text, %s, '', 'g'), %s) ELSE false END AS %s_t",
      term$column, quote(term$negated), quote(term$mention), term$column
    )
  }, character(1))
  kept <- unlist(lapply(terms, function(term) paste0(term$column, c("_m", "_t"))))
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    WITH corpus AS (%s),
    found AS (
      SELECT c.note_id, d.text, %s
      FROM corpus c JOIN discharge d ON d.note_id = c.note_id
    )
    SELECT note_id, %s, %s FROM found
  ", FLAGS_TABLE, input_corpus_sql(con), paste(found, collapse = ", "),
    paste(sprintf("%s_m", vapply(terms, `[[`, character(1), "column")), collapse = ", "),
    paste(asserted, collapse = ", ")))
  FLAGS_TABLE
}

count_flags <- function(column, within = NULL) {
  scope <- if (is.null(within)) "" else sprintf(" AND %s", within)
  list(
    zminka = sprintf("count(*) FILTER (WHERE %s_m%s)", column, scope),
    tvrzeni = sprintf("count(*) FILTER (WHERE %s_t%s)", column, scope)
  )
}

mention_agreement <- function(con, candidates, mentions) {
  flags <- mention_flags(con, candidates, mentions)
  notes <- candidate_notes(con, candidates)
  terms <- mention_terms(candidates, mentions)
  selects <- vapply(seq_along(candidates), function(index) {
    key <- DBI::dbQuoteString(con, names(candidates)[[index]])
    column <- terms[[index]]$column
    coded <- sprintf("note_id IN (SELECT note_id FROM %s WHERE kandidat = %s)", notes, key)
    all <- count_flags(column)
    coded_counts <- count_flags(column, coded)
    sprintf("
      SELECT %s AS kandidat,
        (SELECT count(DISTINCT note_id) FROM %s WHERE kandidat = %s) AS icd,
        %s AS zminka, %s AS tvrzeni, %s AS icd_zminka, %s AS icd_tvrzeni,
        count(*) AS celkem
      FROM %s
    ", key, notes, key, all$zminka, all$tvrzeni, coded_counts$zminka, coded_counts$tvrzeni, flags)
  }, character(1))
  out <- DBI::dbGetQuery(con, paste(selects, collapse = " UNION ALL "))
  out[names(out) != "kandidat"] <- lapply(out[names(out) != "kandidat"], as.numeric)
  out$kandidat <- names(candidates)[match(out$kandidat, names(candidates))]
  out[match(names(candidates), out$kandidat), , drop = FALSE]
}

symptom_mentions <- function(con, candidates, mentions) {
  flags <- mention_flags(con, candidates, mentions)
  terms <- mention_terms(candidates, mentions)
  keywords <- mentions$symptoms$keywords_en
  selects <- vapply(seq_along(keywords), function(index) {
    counts <- count_flags(terms[[length(candidates) + index]]$column)
    sprintf(
      "SELECT %s AS priznak, %s AS zminka, %s AS tvrzeni, count(*) AS celkem FROM %s",
      DBI::dbQuoteString(con, keywords[[index]]), counts$zminka, counts$tvrzeni, flags
    )
  }, character(1))
  out <- DBI::dbGetQuery(con, paste(selects, collapse = " UNION ALL "))
  out[c("zminka", "tvrzeni", "celkem")] <- lapply(out[c("zminka", "tvrzeni", "celkem")], as.numeric)
  out[match(keywords, out$priznak), , drop = FALSE]
}

hide_row <- function(df, hidden, columns) {
  df[hidden, columns] <- NA_real_
  rownames(df) <- NULL
  df
}

agreement_cells <- function(df) {
  with(df, cbind(
    icd_tvrzeni,
    icd_zminka - icd_tvrzeni,
    icd - icd_zminka,
    tvrzeni - icd_tvrzeni,
    (zminka - tvrzeni) - (icd_zminka - icd_tvrzeni),
    celkem - icd - zminka + icd_zminka
  ))
}

suppress_agreement <- function(df) {
  cells <- agreement_cells(df)
  hidden <- rowSums(is.na(cells) | cells < DUA_THRESHOLD) > 0L
  hide_row(df, hidden, AGREEMENT_COUNTS)
}

align_agreement <- function(df, prevalence) {
  published <- prevalence$n[match(df$kandidat, prevalence$kandidat)]
  hide_row(df, is.na(published), AGREEMENT_COUNTS)
}

suppress_symptoms <- function(df) {
  shown <- !is.na(dua_binary(df$zminka, df$celkem)) &
    !is.na(dua_binary(df$tvrzeni, df$celkem)) &
    df$zminka - df$tvrzeni >= DUA_THRESHOLD
  hide_row(df, !shown, c("zminka", "tvrzeni"))
}
