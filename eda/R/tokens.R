ENCODER_FULL_SHARE <- 0.95
CZECH_TOLERANCE <- 0.05

tokens_table_name <- function(table) paste0("note_tokens_", table)

tokens_for <- function(con, table) {
  name <- tokens_table_name(table)
  if (!name %in% DBI::dbListTables(con)) {
    stop(sprintf("Tabulka %s neexistuje, nejdřív zavolej note_tokens().", name))
  }
  name
}

TOKEN_COLUMNS <- c(note_id = "VARCHAR", xlmr = "INTEGER", mdeberta = "INTEGER", translategemma = "INTEGER")
TOKENIZER_COLUMNS <- c(
  tokenizer = "VARCHAR", model = "VARCHAR", revision = "VARCHAR", window = "INTEGER",
  role = "VARCHAR", special_tokens = "INTEGER", transformers = "VARCHAR"
)
PROMPT_COLUMNS <- c(prompt_overhead = "INTEGER", split_ratio = "DOUBLE", sample_limit = "INTEGER")
SPLIT_COLUMNS <- c(note_table = "VARCHAR", note_id = "VARCHAR", fits = "BOOLEAN", parts = "INTEGER", unsplittable = "BOOLEAN")
PARALLEL_COLUMNS <- c(pair_id = "VARCHAR", tokenizer = "VARCHAR", english = "INTEGER", czech = "INTEGER")

tokens_source <- function(con, file, columns) {
  root <- Sys.getenv("MEDIPARSE_EDA_CACHE")
  if (root == "") stop("Chybí MEDIPARSE_EDA_CACHE; spusťte render skriptem EDA.")
  feature_source(con, file, columns, directory = file.path(root, "tokens"), command = "mediparse-eda-note-tokens")
}

note_tokens <- function(con, table) {
  name <- tokens_table_name(table)
  if (name %in% DBI::dbListTables(con)) return(name)
  source <- tokens_source(con, paste0(name, ".parquet"), TOKEN_COLUMNS)
  DBI::dbExecute(con, sprintf(
    "CREATE TEMP VIEW %s AS SELECT * FROM %s", DBI::dbQuoteIdentifier(con, name), source
  ))
  name
}

tokenizer_settings <- function(con) {
  source <- tokens_source(con, "tokenizers.parquet", TOKENIZER_COLUMNS)
  DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", source))
}

encoder_window <- function(settings) {
  windows <- unique(settings$window[settings$role == "encoder"])
  if (length(windows) != 1L) stop("Encodery musí mít jediné společné okno.")
  windows[[1]]
}

translation_prompt <- function(con) {
  source <- tokens_source(con, "translation_prompt.parquet", PROMPT_COLUMNS)
  DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", source))
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

translation_limit <- function(window, split_ratio, overhead) {
  (window - overhead) / (1 + split_ratio)
}

token_windows <- function(encoders, translation, encoder_window, scale = 1) {
  tokenizer <- c(names(encoders), names(translation))
  scales <- if (is.null(names(scale))) rep_len(scale, length(tokenizer)) else unname(scale[tokenizer])
  data.frame(
    tokenizer = tokenizer,
    specials = c(unname(encoders), rep(0, length(translation))),
    capacity = c(encoder_window - unname(encoders), unname(translation)),
    scale = scales,
    stringsAsFactors = FALSE
  )
}

split_sample <- function(con, table, limit) {
  source <- tokens_source(con, "translation_sample.parquet", SPLIT_COLUMNS)
  notes <- DBI::dbGetQuery(con, sprintf(
    "SELECT s.*, t.translategemma FROM %s s JOIN %s t USING (note_id) WHERE s.note_table = %s",
    source, DBI::dbQuoteIdentifier(con, tokens_for(con, table)), DBI::dbQuoteString(con, table)
  ))
  lengths <- notes$translategemma
  splittable <- !notes$unsplittable
  exact <- notes$parts[splittable]
  lower <- sum(ceiling(lengths[splittable] / limit))
  data.frame(
    tabulka = table,
    zprav = nrow(notes),
    nedelitelnych = sum(!splittable),
    deleno = sum(exact > 1),
    casti_presne = sum(exact),
    casti_dolni = lower,
    pomer = if (lower > 0) sum(exact) / lower else NA_real_,
    neshod = sum((lengths <= limit) != notes$fits),
    stringsAsFactors = FALSE
  )
}

ratio_text <- function(x) {
  ifelse(is.na(x), "–", format(round(x, 2), nsmall = 2, decimal.mark = ",", trim = TRUE))
}

sample_table <- function(sample) {
  data.frame(
    tabulka = sample$tabulka,
    zprav = format_number(sample$zprav),
    deleno = format_number(sample$deleno),
    deleno_podil = percent_text(sample$deleno, sample$zprav),
    casti_presne = format_number(sample$casti_presne),
    casti_dolni = format_number(sample$casti_dolni),
    pomer = ratio_text(sample$pomer),
    neshod = format_number(sample$neshod),
    nedelitelnych = format_number(sample$nedelitelnych),
    stringsAsFactors = FALSE
  )
}

translation_verdict <- function(divided, total) {
  if (total == 0) {
    return("nelze")
  }
  if (divided == 0) "nedeli" else "deli"
}

TOKENIZER_LABELS <- c(xlmr = "XLM-R", mdeberta = "mDeBERTa", translategemma = "translategemma")

tokenizer_label <- function(keys) {
  unname(TOKENIZER_LABELS[keys])
}

parallel_ratio <- function(con, windows) {
  source <- tokens_source(con, "parallel_tokens.parquet", PARALLEL_COLUMNS)
  pairs <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", source))
  rows <- lapply(windows$tokenizer, function(key) {
    selected <- pairs[pairs$tokenizer == key, ]
    en <- selected$english
    cs <- selected$czech
    capacity <- windows$capacity[windows$tokenizer == key]
    ratio <- cs[en > 0] / en[en > 0]
    data.frame(
      tokenizer = key,
      zprav = nrow(selected),
      median = unname(quantile(ratio, 0.5, type = 1)),
      p95 = unname(quantile(ratio, 0.95, type = 1)),
      v_okne_en = sum(en <= capacity),
      v_okne_cs = sum(cs <= capacity),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

parallel_table <- function(ratio) {
  data.frame(
    tokenizer = tokenizer_label(ratio$tokenizer),
    zprav = format_number(ratio$zprav),
    median = ratio_text(ratio$median),
    p95 = ratio_text(ratio$p95),
    podil_en = percent_text(ratio$v_okne_en, ratio$zprav),
    podil_cs = percent_text(ratio$v_okne_cs, ratio$zprav),
    stringsAsFactors = FALSE
  )
}

czech_table <- function(english, czech, tokenizers) {
  keep <- english$tokenizer %in% tokenizers
  english <- english[keep, ]
  czech <- czech[keep, ]
  data.frame(
    tabulka = english$tabulka,
    tokenizer = tokenizer_label(english$tokenizer),
    zprav = format_number(english$zprav),
    podil_en = percent_text(english$v_okne, english$zprav),
    podil_cs = percent_text(czech$v_okne, czech$zprav),
    casti_en = format_number(english$casti),
    casti_cs = format_number(czech$casti),
    stringsAsFactors = FALSE
  )
}

czech_verdict <- function(english, czech) {
  if (is.na(english) || is.na(czech)) {
    return("nelze")
  }
  if (english - czech <= CZECH_TOLERANCE) "bez_dopadu" else "pokles"
}

tokenizer_table <- function(models, version) {
  data.frame(
    tokenizer = tokenizer_label(models$klic),
    model = models$model,
    revize = substr(models$revize, 1, 12),
    okno = format_number(models$okno),
    uloha = models$uloha,
    transformers = rep(version, nrow(models)),
    stringsAsFactors = FALSE
  )
}

window_table <- function(win) {
  data.frame(
    tabulka = win$tabulka,
    tokenizer = tokenizer_label(win$tokenizer),
    zprav = format_number(win$zprav),
    v_okne = format_number(win$v_okne),
    podil = percent_text(win$v_okne, win$zprav),
    mimo = format_number(win$zprav - win$v_okne),
    casti = format_number(win$casti),
    stringsAsFactors = FALSE
  )
}

corpus_verdict <- function(discharge, radiology) {
  if (is.na(discharge) || is.na(radiology)) {
    return("nelze")
  }
  if (radiology < discharge) {
    return("radiologie_kratsi")
  }
  if (radiology > discharge) "propousteci_kratsi" else "stejne"
}
