ENCODER_WINDOW <- 512L
ENCODER_FULL_SHARE <- 0.95

encoder_models <- function() {
  data.frame(
    klic = c("xlmr", "mdeberta"),
    model = c("FacebookAI/xlm-roberta-base", "microsoft/mdeberta-v3-base"),
    revize = c(
      "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
      "a0484667b22365f84929a935b5e50a51f71f159d"
    ),
    okno = c(ENCODER_WINDOW, ENCODER_WINDOW),
    stringsAsFactors = FALSE
  )
}

load_tokenizer <- function(model, revision) {
  transformers <- reticulate::import("transformers")
  transformers$logging$set_verbosity_error()
  transformers$AutoTokenizer$from_pretrained(model, revision = revision)
}

strip_text <- function(texts) {
  trimws(texts, whitespace = "[\\h\\v]")
}

token_lengths <- function(tok, texts, special, strip = FALSE) {
  if (length(texts) == 0L) {
    return(integer())
  }
  texts <- ifelse(is.na(texts), "", texts)
  if (strip) texts <- strip_text(texts)
  ids <- tok(as.list(texts), add_special_tokens = special)$input_ids
  vapply(ids, length, integer(1))
}

tokens_table_name <- function(table) paste0("note_tokens_", table)

tokens_for <- function(con, table) {
  name <- tokens_table_name(table)
  if (!name %in% DBI::dbListTables(con)) {
    stop(sprintf("Tabulka %s neexistuje, nejdřív zavolej note_tokens().", name))
  }
  name
}

batch_counts <- function(chunk, tokenizers) {
  counts <- lapply(tokenizers, function(spec) {
    token_lengths(spec$tok, chunk$text, spec$special, isTRUE(spec$strip))
  })
  data.frame(note_id = chunk$note_id, counts, check.names = FALSE, stringsAsFactors = FALSE)
}

note_tokens <- function(con, table, tokenizers, batch = 2000L) {
  name <- tokens_table_name(table)
  if (name %in% DBI::dbListTables(con)) {
    return(name)
  }
  rows <- paste0(name, "_rows")
  quote <- function(identifier) DBI::dbQuoteIdentifier(con, identifier)
  columns <- vapply(names(tokenizers), function(column) {
    paste(quote(column), "INTEGER")
  }, character(1))
  DBI::dbExecute(con, sprintf(
    "CREATE TEMP TABLE %s (note_id VARCHAR, %s)", quote(name), paste(columns, collapse = ", ")
  ))
  done <- FALSE
  on.exit({
    DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", quote(rows)))
    if (!done) DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", quote(name)))
  })
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    SELECT row_number() OVER (ORDER BY note_id) AS rn, note_id, text FROM %s
  ", quote(rows), quote(table)))
  total <- as.numeric(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM %s", quote(rows)))$n)
  for (index in seq_len(ceiling(total / batch))) {
    chunk <- DBI::dbGetQuery(con, sprintf(
      "SELECT note_id, text FROM %s WHERE rn BETWEEN %.0f AND %.0f ORDER BY rn",
      quote(rows), (index - 1) * batch + 1, index * batch
    ))
    DBI::dbAppendTable(con, name, batch_counts(chunk, tokenizers))
  }
  done <- TRUE
  name
}

per_tokenizer <- function(con, tables, columns, query) {
  rows <- lapply(tables, function(table) {
    source <- DBI::dbQuoteIdentifier(con, tokens_for(con, table))
    selects <- vapply(columns, function(column) {
      query(source, column, DBI::dbQuoteIdentifier(con, column))
    }, character(1))
    out <- DBI::dbGetQuery(con, paste(selects, collapse = " UNION ALL "))
    cbind(tabulka = rep(table, nrow(out)), out, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

token_percentiles <- function(con, tables, columns) {
  out <- per_tokenizer(con, tables, columns, function(source, name, column) {
    sprintf(percentile_sql(name, column), source)
  })
  as_numeric_columns(out, c("n", "prumer", "minimum", "maximum", PERCENTILE_COLUMNS))
}

token_histogram <- function(con, tables, columns) {
  out <- per_tokenizer(con, tables, columns, function(source, name, column) {
    sprintf("
      SELECT %s AS tokenizer, %s AS kos, count(*) AS n
      FROM %s
      WHERE %s > 0
      GROUP BY kos
    ", DBI::dbQuoteString(con, name), log_bin_sql(column), source, column)
  })
  out <- out[order(out$tabulka, out$tokenizer, out$kos), ]
  rownames(out) <- NULL
  as_numeric_columns(out, c("kos", "n"))
}

number_literal <- function(x) sprintf("CAST(%.15g AS DOUBLE)", x)

window_summary <- function(con, tables, windows) {
  rows <- lapply(seq_len(nrow(windows)), function(index) {
    window <- windows[index, ]
    column <- DBI::dbQuoteIdentifier(con, window$tokenizer)
    content <- sprintf("(%s - %s) * %s", column, number_literal(window$specials), number_literal(window$scale))
    capacity <- number_literal(window$capacity)
    per_tokenizer(con, tables, window$tokenizer, function(source, name, column) {
      sprintf("
        SELECT
          %s AS tokenizer,
          count(*) AS zprav,
          count(*) FILTER (WHERE %s <= %s) AS v_okne,
          coalesce(sum(CASE WHEN %s > 0 THEN ceil(%s / %s) END), 0) AS casti
        FROM %s
      ", DBI::dbQuoteString(con, name), content, capacity, content, content, capacity, source)
    })
  })
  out <- do.call(rbind, rows)
  out <- out[order(match(out$tabulka, tables)), c("tabulka", "tokenizer", "zprav", "v_okne", "casti")]
  rownames(out) <- NULL
  as_numeric_columns(out, c("zprav", "v_okne", "casti"))
}

encoder_verdict <- function(share, median, window) {
  if (is.na(share) || is.na(median)) {
    return("nelze")
  }
  if (share >= ENCODER_FULL_SHARE) {
    return("orez")
  }
  if (median <= window) "vetsina" else "chunkovani"
}
