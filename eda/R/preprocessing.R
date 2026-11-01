PREPROCESSING_TABLE <- "note_preprocessing_input"
VOCABULARY_TABLE <- "preprocessing_vocabulary_input"
VOCABULARY_COLUMNS <- c(variant = "VARCHAR", token = "VARCHAR")
# Varianty vectorizeru (klíč je hodnota `variant` z mostu) a sloupec s tokeny na zprávu; pořadí
# určuje pořadí řádků tabulky. Shodu klíčů s konstantami mostu hlídá test.
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

preprocessing_module <- function() reticulate::import("mediparse.entrypoints.eda_preprocessing")

# Texty a identifikátory musí do Pythonu jako nepojmenovaný seznam ze stejného důvodu jako u
# `section_rows`; typy se skládají ručně, protože prázdný sloupec by přišel jako `list()`.
preprocessing_rows <- function(module, path, acronyms, chunk) {
  rows <- module$note_preprocessing(
    python_path(path), as.list(unname(chunk$note_id)), as.list(unname(chunk$text))
  )
  counts <- lapply(setNames(acronyms, acronyms), function(acronym) as.integer(unlist(rows[[acronym]])))
  data.frame(
    note_id = as.character(unlist(rows$note_id)),
    tokens_raw = as.integer(unlist(rows$tokens_raw)),
    tokens_preprocessed = as.integer(unlist(rows$tokens_preprocessed)),
    counts,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

vocabulary_rows <- function(module, path, chunk) {
  rows <- module$note_vocabulary(python_path(path), as.list(unname(chunk$text)))
  data.frame(
    variant = as.character(unlist(rows$variant)),
    token = as.character(unlist(rows$token)),
    stringsAsFactors = FALSE
  )
}

corpus_acronyms <- function(module, path) {
  as.character(unlist(module$diagnosis_acronyms(python_path(path))))
}

note_preprocessing <- function(con, module, path, batch = 2000L) {
  acronyms <- corpus_acronyms(module, path)
  columns <- c(
    note_id = "VARCHAR", tokens_raw = "INTEGER", tokens_preprocessed = "INTEGER",
    setNames(rep("INTEGER", length(acronyms)), acronyms)
  )
  batched_table(
    con, PREPROCESSING_TABLE, columns, input_texts_sql(con),
    function(chunk) preprocessing_rows(module, path, acronyms, chunk), batch = batch
  )
}

# Slovník je sjednocení distinct tokenů dávek, počítá ho DuckDB; Python nedrží stav mezi dávkami.
note_vocabulary <- function(con, module, path, batch = 2000L) {
  batched_table(
    con, VOCABULARY_TABLE, VOCABULARY_COLUMNS, input_texts_sql(con),
    function(chunk) vocabulary_rows(module, path, chunk), batch = batch
  )
}

preprocessing_summary <- function(con) {
  selects <- mapply(percentile_sql, names(PREPROCESSING_VARIANTS), unname(PREPROCESSING_VARIANTS), USE.NAMES = FALSE)
  tokens <- DBI::dbGetQuery(con, paste(sprintf(selects, PREPROCESSING_TABLE), collapse = " UNION ALL "))
  tokens <- as_numeric_columns(tokens, c("n", "prumer", "minimum", "maximum", PERCENTILE_COLUMNS))
  tokens <- tokens[match(names(PREPROCESSING_VARIANTS), tokens$metrika), , drop = FALSE]
  vocabulary <- DBI::dbGetQuery(con, sprintf(
    "SELECT variant, count(DISTINCT token) AS slovnik FROM %s GROUP BY variant", VOCABULARY_TABLE
  ))
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
    sprintf(" (změna o %s)", percent_change_text(prepared$slovnik, raw$slovnik))
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
  paste0(ifelse(change > 0, "+", ""), dua_decimal(change), NBSP, "%")
}

acronym_cells <- function(con, acronym, key) {
  cells <- DBI::dbGetQuery(con, sprintf("
    SELECT
      count(*) FILTER (WHERE has_acronym AND has_code) AS oba,
      count(*) FILTER (WHERE has_acronym AND NOT has_code) AS jen_zkratka,
      count(*) FILTER (WHERE NOT has_acronym AND has_code) AS jen_kod,
      count(*) FILTER (WHERE NOT has_acronym AND NOT has_code) AS zadne
    FROM (
      SELECT p.%1$s > 0 AS has_acronym,
        EXISTS (
          SELECT 1 FROM %2$s c WHERE c.kandidat = %3$s AND c.note_id = p.note_id
        ) AS has_code
      FROM %4$s p
    )
  ", DBI::dbQuoteIdentifier(con, acronym), CANDIDATES_TABLE, DBI::dbQuoteString(con, key), PREPROCESSING_TABLE))
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
      "SELECT coalesce(sum(%s), 0) AS n FROM %s", DBI::dbQuoteIdentifier(con, acronym), PREPROCESSING_TABLE
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
