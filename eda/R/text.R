PERCENTILES <- c(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
PERCENTILE_COLUMNS <- sprintf("p%02d", as.integer(round(100 * PERCENTILES)))
LENGTH_METRICS <- c(znaky = "chars", slova = "words", znacky = "markers")

SHORT_LIMIT <- 100L
MEDIUM_LIMIT <- 500L
LONG_LIMIT <- 50000L
REFERENCE_MARKERS <- 60.6
REFERENCE_TOLERANCE <- 0.05

RESIDUAL_PATTERNS <- c(
  "[**\u2026**]" = "\\[\\*\\*[^\\]]*\\*\\*\\]",
  "<\u2026>" = "</?[A-Za-z][^>]*>",
  "XXX" = "X{3,}"
)
RESIDUAL_COLUMNS <- c("brackets", "html", "xxx")

AGE_MASK <- "(?i)___[ -]?(years?[ -]?old|yo|y/o|y\\.o\\.)"
AGE_ANCHOR <- "(?i)is an? ___[ -]?(years?[ -]?old|yo|y/o)"
AGE_NUMERIC <- "(?i)\\b[0-9]{1,3}[ -]?(years?[ -]?old|y/o|yo)\\b"

metrics_table_name <- function(table) paste0("note_metrics_", table)

note_metrics <- function(con, table) {
  name <- metrics_table_name(table)
  DBI::dbExecute(con, sprintf("
    CREATE OR REPLACE TEMP TABLE %s AS
    SELECT
      note_id,
      length(text) AS chars,
      len(regexp_extract_all(text, '\\S+')) AS words,
      len(list_filter(regexp_extract_all(text, '_+'), x -> length(x) = 3)) AS markers,
      len(list_filter(regexp_extract_all(text, '_+'), x -> length(x) > 3)) AS longer_runs,
      len(regexp_extract_all(text, %s)) AS brackets,
      len(regexp_extract_all(text, %s)) AS html,
      len(regexp_extract_all(text, %s)) AS xxx,
      len(regexp_extract_all(text, %s)) AS age_masks,
      regexp_matches(text, %s) AS age_anchor,
      regexp_matches(text, %s) AS age_numeric,
      CASE WHEN length(text) > 0 THEN hash(text) END AS fingerprint
    FROM (SELECT note_id, coalesce(text, '') AS text FROM %s)
  ",
    DBI::dbQuoteIdentifier(con, name),
    DBI::dbQuoteString(con, RESIDUAL_PATTERNS[[1]]),
    DBI::dbQuoteString(con, RESIDUAL_PATTERNS[[2]]),
    DBI::dbQuoteString(con, RESIDUAL_PATTERNS[[3]]),
    DBI::dbQuoteString(con, AGE_MASK),
    DBI::dbQuoteString(con, AGE_ANCHOR),
    DBI::dbQuoteString(con, AGE_NUMERIC),
    DBI::dbQuoteIdentifier(con, table)
  ))
  name
}

metrics_for <- function(con, table) {
  name <- metrics_table_name(table)
  if (name %in% DBI::dbListTables(con)) {
    return(name)
  }
  note_metrics(con, table)
}

per_table <- function(con, tables, query) {
  rows <- lapply(tables, function(table) {
    metrics <- DBI::dbQuoteIdentifier(con, metrics_for(con, table))
    out <- DBI::dbGetQuery(con, query(metrics))
    cbind(tabulka = rep(table, nrow(out)), out, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

as_numeric_columns <- function(df, columns) {
  df[columns] <- lapply(df[columns], as.numeric)
  df
}

percentile_sql <- function(metric, column) {
  quantiles <- sprintf("quantile_disc(%s, %s) AS %s", column, PERCENTILES, PERCENTILE_COLUMNS)
  sprintf(
    "SELECT '%s' AS metrika, count(%s) AS n, avg(%s) AS prumer, min(%s) AS minimum, max(%s) AS maximum, %s FROM %%s",
    metric, column, column, column, column, paste(quantiles, collapse = ", ")
  )
}

length_percentiles <- function(con, tables) {
  selects <- mapply(percentile_sql, names(LENGTH_METRICS), LENGTH_METRICS, USE.NAMES = FALSE)
  out <- per_table(con, tables, function(metrics) {
    paste(sprintf(selects, metrics), collapse = " UNION ALL ")
  })
  as_numeric_columns(out, c("n", "prumer", "minimum", "maximum", PERCENTILE_COLUMNS))
}

format_percentiles <- function(percentiles) {
  out <- percentiles
  for (column in c("minimum", PERCENTILE_COLUMNS, "maximum")) {
    out[[column]] <- number_text(percentiles[[column]])
  }
  out$prumer <- ifelse(is.na(percentiles$prumer), "–", dua_decimal(percentiles$prumer))
  out$n <- number_text(percentiles$n)
  out
}

log_bin_sql <- function(column) sprintf("floor(log10(%s) * 10) / 10", column)

length_histogram <- function(con, tables) {
  out <- per_table(con, tables, function(metrics) sprintf("
    SELECT %s AS kos, count(*) AS n
    FROM %s
    WHERE chars > 0
    GROUP BY kos
    ORDER BY kos
  ", log_bin_sql("chars"), metrics))
  as_numeric_columns(out, c("kos", "n"))
}

marker_summary <- function(con, tables) {
  out <- per_table(con, tables, function(metrics) sprintf("
    SELECT
      count(*) AS zprav,
      coalesce(sum(markers), 0) AS znacek,
      count(*) FILTER (WHERE markers > 0) AS se_znackou,
      coalesce(sum(longer_runs), 0) AS delsi_useky
    FROM %s
  ", metrics))
  as_numeric_columns(out, c("zprav", "znacek", "se_znackou", "delsi_useky"))
}

residual_masks <- function(con, tables) {
  selects <- vapply(seq_along(RESIDUAL_PATTERNS), function(index) {
    sprintf(
      "SELECT %s AS vzor, count(*) FILTER (WHERE %s > 0) AS zprav, coalesce(sum(%s), 0) AS vyskytu FROM %%s",
      DBI::dbQuoteString(con, names(RESIDUAL_PATTERNS)[index]),
      RESIDUAL_COLUMNS[index], RESIDUAL_COLUMNS[index]
    )
  }, character(1))
  out <- per_table(con, tables, function(metrics) {
    paste(sprintf(selects, metrics), collapse = " UNION ALL ")
  })
  as_numeric_columns(out, c("zprav", "vyskytu"))
}

note_quality <- function(con, tables) {
  out <- per_table(con, tables, function(metrics) sprintf("
    SELECT
      count(*) AS zprav,
      count(*) FILTER (WHERE coalesce(chars, 0) = 0) AS prazdne,
      count(*) FILTER (WHERE chars > 0 AND chars < %d) AS pod_100,
      count(*) FILTER (WHERE chars >= %d AND chars <= %d) AS od_100_do_500,
      count(*) FILTER (WHERE chars > %d) AS nad_50000,
      count(fingerprint) - count(DISTINCT fingerprint) AS duplicity
    FROM %s
  ", SHORT_LIMIT, SHORT_LIMIT, MEDIUM_LIMIT, LONG_LIMIT, metrics))
  as_numeric_columns(
    out, c("zprav", "prazdne", "pod_100", "od_100_do_500", "nad_50000", "duplicity")
  )
}

age_masks <- function(con, table = "discharge") {
  metrics <- DBI::dbQuoteIdentifier(con, metrics_for(con, table))
  row <- as.list(DBI::dbGetQuery(con, sprintf("
    SELECT
      count(*) AS zprav,
      count(*) FILTER (WHERE coalesce(age_masks, 0) = 0) AS bez_masky,
      count(*) FILTER (WHERE age_masks = 1) AS jedna,
      count(*) FILTER (WHERE age_masks > 1) AS vice,
      count(*) FILTER (WHERE age_anchor) AS kotva,
      count(*) FILTER (WHERE age_numeric) AS ciselny
    FROM %s
  ", metrics)))
  lapply(row, as.numeric)
}

marker_means <- function(markers, residual) {
  if (markers$zprav == 0) {
    return(list(presne = NA_real_, delsi = NA_real_, zbytkove = NA_real_))
  }
  list(
    presne = markers$znacek / markers$zprav,
    delsi = markers$delsi_useky / markers$zprav,
    zbytkove = sum(residual$vyskytu) / markers$zprav
  )
}

marker_reference_verdict <- function(presne, zbytkove, reference = REFERENCE_MARKERS) {
  if (is.na(presne) || is.na(zbytkove)) {
    return("nelze")
  }
  near <- function(value) abs(value - reference) <= REFERENCE_TOLERANCE * reference
  if (near(presne)) {
    return("presne")
  }
  if (near(presne + zbytkove)) "se_zbytkovymi" else "rozdil"
}

age_verdict <- function(age) {
  if (age$ciselny == 0) "bez_ciselneho" else "ciselny"
}

residual_verdict <- function(residual) {
  if (sum(residual$vyskytu) == 0) "zadne" else "jsou"
}

mean_text <- function(x) {
  ifelse(is.na(x), "–", dua_decimal(x))
}

percentile_cell <- function(shown, table, metric, column) {
  row <- shown[shown$tabulka == table & shown$metrika == metric, , drop = FALSE]
  if (nrow(row) == 0L) "–" else row[[column]][1]
}

PERCENTILE_LABELS <- c(
  p01 = "1. percentil", p05 = "5. percentil", p10 = "10. percentil", p25 = "25. percentil",
  p50 = "Medián", p75 = "75. percentil", p90 = "90. percentil", p95 = "95. percentil",
  p99 = "99. percentil"
)

percentile_matrix <- function(shown, columns) {
  cells <- lapply(seq_len(nrow(columns)), function(index) {
    table <- columns$tabulka[index]
    metric <- columns$metrika[index]
    c(
      percentile_cell(shown, table, metric, "n"),
      percentile_cell(shown, table, metric, "prumer"),
      percentile_cell(shown, table, metric, "minimum"),
      vapply(
        PERCENTILE_COLUMNS,
        function(column) percentile_cell(shown, table, metric, column),
        character(1)
      ),
      percentile_cell(shown, table, metric, "maximum")
    )
  })
  out <- data.frame(
    ukazatel = c(
      "Zpráv", "Průměr", "Minimum", unname(PERCENTILE_LABELS[PERCENTILE_COLUMNS]), "Maximum"
    ),
    stringsAsFactors = FALSE
  )
  for (index in seq_along(cells)) out[[paste0("v", index)]] <- cells[[index]]
  out
}

histogram_shares <- function(hist) {
  totals <- tapply(hist$n, hist$tabulka, sum)
  data.frame(
    tabulka = hist$tabulka,
    popisek = format_number(signif(10^hist$kos, 3)),
    dolni = hist$kos,
    podil = hist$n / as.numeric(totals[hist$tabulka]),
    stringsAsFactors = FALSE
  )
}

age_table <- function(age) {
  counts <- c(age$zprav, age$bez_masky, age$jedna, age$vice, age$kotva, age$ciselny)
  data.frame(
    ukazatel = c(
      "Propouštěcí zprávy celkem", "Bez masky věku", "S jednou maskou věku",
      "S více maskami věku", "S kotvou \u201eis a ___ year old\u201c", "S číselným věkem"
    ),
    n = format_number(counts),
    podil = c("", percent_text(counts[-1], age$zprav)),
    stringsAsFactors = FALSE
  )
}

QUALITY_ROWS <- c("prazdne", "pod_100", "od_100_do_500", "nad_50000", "duplicity")

quality_labels <- function() {
  c(
    "Zprávy celkem", "Prázdný nebo chybějící text",
    sprintf("Kratší než %d znaků", SHORT_LIMIT),
    sprintf("%d až %d znaků", SHORT_LIMIT, MEDIUM_LIMIT),
    sprintf("Delší než %s znaků", format_number(LONG_LIMIT)),
    "Přesné duplicity textu"
  )
}

quality_table <- function(quality, tables = c("discharge", "radiology")) {
  out <- data.frame(ukazatel = quality_labels(), stringsAsFactors = FALSE)
  for (table in tables) {
    row <- quality[quality$tabulka == table, ]
    counts <- c(row$zprav, unlist(row[QUALITY_ROWS], use.names = FALSE))
    out[[paste0("n_", table)]] <- format_number(counts)
    out[[paste0("podil_", table)]] <- c("", percent_text(counts[-1], row$zprav))
  }
  out
}

reference_verdict <- function(markers, residual) {
  means <- marker_means(markers, residual)
  marker_reference_verdict(means$presne, means$zbytkove)
}

marker_table <- function(markers, shown) {
  cell <- function(column) {
    vapply(
      markers$tabulka,
      function(table) percentile_cell(shown, table, "znacky", column),
      character(1),
      USE.NAMES = FALSE
    )
  }
  data.frame(
    tabulka = markers$tabulka,
    zprav = format_number(markers$zprav),
    se_znackou = format_number(markers$se_znackou),
    se_znackou_podil = percent_text(markers$se_znackou, markers$zprav),
    znacek = format_number(markers$znacek),
    delsi_useky = format_number(markers$delsi_useky),
    prumer = cell("prumer"),
    p50 = cell("p50"),
    p95 = cell("p95")
  )
}

masks_table <- function(residual, markers) {
  totals <- markers$zprav[match(residual$tabulka, markers$tabulka)]
  data.frame(
    tabulka = residual$tabulka,
    vzor = residual$vzor,
    zprav = format_number(residual$zprav),
    podil = percent_text(residual$zprav, totals),
    vyskytu = format_number(residual$vyskytu)
  )
}
