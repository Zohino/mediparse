PREVALENCE_BAND <- c(low = 0.05, high = 0.60)
REDUNDANCY_THRESHOLD <- 0.5
ERA_RATIO <- 1.5
HIDDEN_TEXT <- "potlačeno"
CODES_TABLE <- "diagnosis_codes_input"
CANDIDATES_TABLE <- "candidate_notes"
GROUP_KEY <- "substr(code, 1, CASE WHEN version = 9 AND starts_with(code, 'E') THEN 4 ELSE 3 END)"

SYMPTOM_PREFIXES <- list(icd9 = c("78", "79"), icd10 = "R")

CHARLSON_LABELS <- c(
  mi = "Infarkt myokardu",
  chf = "Srdeční selhání",
  pvd = "Onemocnění periferních cév",
  cevd = "Cerebrovaskulární onemocnění",
  dementia = "Demence",
  cpd = "Chronické plicní onemocnění",
  rheumd = "Revmatické onemocnění",
  pud = "Peptický vřed",
  mld = "Mírné onemocnění jater",
  diab = "Diabetes bez komplikací",
  diabwc = "Diabetes s komplikacemi",
  hp = "Hemiplegie nebo paraplegie",
  rend = "Onemocnění ledvin",
  canc = "Zhoubný nádor",
  msld = "Středně těžké až těžké onemocnění jater",
  metacanc = "Metastatický nádor",
  aids = "AIDS/HIV"
)

BAND_LABELS <- c(
  pod = "pod pásmem", v = "v pásmu", nad = "nad pásmem", nelze = "nelze posoudit"
)

candidates_path <- function() config_path("diagnosis_candidates.json")

candidate_labels <- function(keys, candidates) {
  unname(vapply(keys, function(key) candidates[[key]]$label_cs, character(1)))
}

read_candidates <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)$candidates
  lapply(raw, function(entry) {
    list(
      label_cs = entry$label_cs,
      icd9_prefixes = as.character(unlist(entry$icd9_prefixes)),
      icd10_prefixes = as.character(unlist(entry$icd10_prefixes)),
      charlson = as.character(unlist(entry$charlson)),
      kapitola = entry$kapitola,
      keywords_en = as.character(unlist(entry$keywords_en)),
      acronyms_en = as.character(unlist(entry$acronyms_en))
    )
  })
}

version_clause <- function(con, version, prefixes) {
  if (length(prefixes) == 0L) {
    return("false")
  }
  starts <- sprintf("starts_with(code, %s)", DBI::dbQuoteString(con, prefixes))
  sprintf("(version = %d AND (%s))", version, paste(starts, collapse = " OR "))
}

match_sql <- function(con, icd9, icd10) {
  sprintf(
    "coalesce(%s OR %s, false)",
    version_clause(con, 9L, icd9), version_clause(con, 10L, icd10)
  )
}

diagnosis_codes <- function(con) {
  if (CODES_TABLE %in% DBI::dbListTables(con)) {
    return(CODES_TABLE)
  }
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    WITH corpus AS (%s)
    SELECT DISTINCT
      c.note_id,
      c.hadm_id,
      upper(trim(i.icd_code)) AS code,
      TRY_CAST(i.icd_version AS INTEGER) AS version
    FROM corpus c
    JOIN diagnoses_icd i ON i.hadm_id = c.hadm_id
    WHERE i.icd_code IS NOT NULL
  ", CODES_TABLE, input_corpus_sql(con)))
  CODES_TABLE
}

era_summary <- function(con) {
  codes <- diagnosis_codes(con)
  symptom <- match_sql(con, SYMPTOM_PREFIXES$icd9, SYMPTOM_PREFIXES$icd10)
  row <- query_row(con, sprintf("
    WITH corpus AS (%s),
    per_note AS (
      SELECT
        note_id,
        coalesce(bool_or(version = 9), false) AS v9,
        coalesce(bool_or(version = 10), false) AS v10,
        bool_or(%s) AS symptom
      FROM %s
      GROUP BY note_id
    )
    SELECT
      (SELECT count(*) FROM corpus) AS notes,
      count(*) FILTER (WHERE v9 AND NOT v10) AS only9,
      count(*) FILTER (WHERE v10 AND NOT v9) AS only10,
      count(*) FILTER (WHERE v9 AND v10) AS both,
      count(*) FILTER (WHERE v9) AS n9,
      count(*) FILTER (WHERE v10) AS n10,
      count(*) FILTER (WHERE symptom) AS symptom
    FROM per_note
  ", input_corpus_sql(con), symptom, codes))
  out <- lapply(row, as.numeric)
  out$neither <- out$notes - out$only9 - out$only10 - out$both
  out
}

charlson_flags <- function(codes, map) {
  if (nrow(codes) == 0L) {
    return(NULL)
  }
  flags <- suppressMessages(comorbidity::comorbidity(
    x = codes, id = "note_id", code = "code", map = map,
    assign0 = FALSE, labelled = FALSE, tidy.codes = TRUE
  ))
  as.data.frame(flags)
}

charlson_counts <- function(con) {
  codes <- diagnosis_codes(con)
  rows <- DBI::dbGetQuery(con, sprintf("SELECT DISTINCT note_id, code, version FROM %s", codes))
  maps <- c("9" = "charlson_icd9_quan", "10" = "charlson_icd10_quan")
  flagged <- Filter(Negate(is.null), lapply(names(maps), function(version) {
    charlson_flags(rows[rows$version %in% as.integer(version), c("note_id", "code")], maps[[version]])
  }))
  counts <- vapply(names(CHARLSON_LABELS), function(category) {
    ids <- unlist(lapply(flagged, function(flags) flags$note_id[flags[[category]] == 1]))
    as.numeric(length(unique(ids)))
  }, numeric(1))
  data.frame(kategorie = names(CHARLSON_LABELS), n = unname(counts), stringsAsFactors = FALSE)
}

GROUP_SQL <- paste0("
  WITH scoped AS (
    SELECT note_id, code, version, ", GROUP_KEY, " AS grp
    FROM %s WHERE version = %d AND %s
  ),
  full_codes AS (
    SELECT grp, code, count(DISTINCT note_id) AS n_code
    FROM scoped GROUP BY ALL
  ),
  headline AS (
    SELECT grp, code,
      row_number() OVER (PARTITION BY grp ORDER BY n_code DESC, code) AS place
    FROM full_codes
    WHERE n_code >= %d
  ),
  titles AS (
    SELECT upper(trim(icd_code)) AS code, min(long_title) AS title
    FROM d_icd_diagnoses
    WHERE TRY_CAST(icd_version AS INTEGER) = %d
    GROUP BY 1
  ),
  groups AS (
    SELECT grp, count(DISTINCT note_id) AS n FROM scoped GROUP BY 1
  )
  SELECT
    g.grp AS kod,
    t.title AS nazev,
    g.n AS n,
    (SELECT count(DISTINCT note_id) FROM %s WHERE version = %d) AS total
  FROM groups g
  LEFT JOIN headline h ON h.grp = g.grp AND h.place = 1
  LEFT JOIN titles t ON t.code = h.code
")

code_ranking <- function(con, version, prefixes = NULL, top = 20L) {
  codes <- diagnosis_codes(con)
  filter <- if (length(prefixes) == 0L) "true" else version_clause(con, version, prefixes)
  rows <- DBI::dbGetQuery(con, sprintf(
    GROUP_SQL, codes, as.integer(version), filter, DUA_THRESHOLD,
    as.integer(version), codes, as.integer(version)
  ))
  rows$n <- as.numeric(rows$n)
  rows$total <- as.numeric(rows$total)
  rows <- rows[!is.na(dua_binary(rows$n, rows$total)), , drop = FALSE]
  rows <- rows[order(-rows$n, rows$kod), , drop = FALSE]
  rows <- utils::head(rows, top)
  data.frame(
    kod = rows$kod,
    nazev = ifelse(is.na(rows$nazev), UNKNOWN_LABEL, rows$nazev),
    n = rows$n,
    podil = rows$n / rows$total,
    stringsAsFactors = FALSE
  )
}

candidate_notes <- function(con, candidates) {
  codes <- diagnosis_codes(con)
  selects <- vapply(names(candidates), function(key) {
    entry <- candidates[[key]]
    sprintf(
      "SELECT DISTINCT %s AS kandidat, note_id, version FROM %s WHERE %s",
      DBI::dbQuoteString(con, key), codes,
      match_sql(con, entry$icd9_prefixes, entry$icd10_prefixes)
    )
  }, character(1))
  DBI::dbExecute(con, sprintf(
    "CREATE OR REPLACE TEMP TABLE %s AS %s", CANDIDATES_TABLE, paste(selects, collapse = " UNION ALL ")
  ))
  CANDIDATES_TABLE
}

candidate_prevalence <- function(con, candidates) {
  table <- candidate_notes(con, candidates)
  era <- era_summary(con)
  found <- DBI::dbGetQuery(con, sprintf("
    SELECT
      kandidat,
      count(DISTINCT note_id) AS n,
      count(DISTINCT note_id) FILTER (WHERE version = 9) AS n9,
      count(DISTINCT note_id) FILTER (WHERE version = 10) AS n10
    FROM %s GROUP BY kandidat
  ", table))
  index <- match(names(candidates), found$kandidat)
  value <- function(column) {
    out <- as.numeric(found[[column]][index])
    out[is.na(out)] <- 0
    out
  }
  data.frame(
    kandidat = names(candidates),
    n = value("n"), n9 = value("n9"), n10 = value("n10"),
    celkem = era$notes, celkem9 = era$n9, celkem10 = era$n10,
    stringsAsFactors = FALSE
  )
}

suppress_prevalence <- function(prevalence) {
  out <- prevalence
  out$n <- dua_binary(prevalence$n, prevalence$celkem)
  out$n9 <- dua_binary(prevalence$n9, prevalence$celkem9)
  out$n10 <- dua_binary(prevalence$n10, prevalence$celkem10)
  both <- !is.na(out$n9) & !is.na(out$n10)
  out$n9[!both] <- NA_real_
  out$n10[!both] <- NA_real_
  out
}

cooccurrence <- function(con, candidates) {
  table <- candidate_notes(con, candidates)
  keys <- names(candidates)
  totals <- DBI::dbGetQuery(con, sprintf(
    "SELECT kandidat, count(DISTINCT note_id) AS n_i FROM %s GROUP BY kandidat", table
  ))
  pairs <- DBI::dbGetQuery(con, sprintf("
    SELECT a.kandidat AS radek, b.kandidat AS sloupec, count(DISTINCT a.note_id) AS n_ij
    FROM %s a JOIN %s b ON a.note_id = b.note_id AND a.kandidat <> b.kandidat
    GROUP BY 1, 2
  ", table, table))
  grid <- expand.grid(radek = keys, sloupec = keys, stringsAsFactors = FALSE)
  grid <- grid[grid$radek != grid$sloupec, , drop = FALSE]
  at <- match(paste(grid$radek, grid$sloupec), paste(pairs$radek, pairs$sloupec))
  grid$n_ij <- ifelse(is.na(at), 0, as.numeric(pairs$n_ij[at]))
  grid$n_i <- ifelse(
    is.na(match(grid$radek, totals$kandidat)), 0,
    as.numeric(totals$n_i[match(grid$radek, totals$kandidat)])
  )
  grid$n_j <- ifelse(
    is.na(match(grid$sloupec, totals$kandidat)), 0,
    as.numeric(totals$n_i[match(grid$sloupec, totals$kandidat)])
  )
  rownames(grid) <- NULL
  grid
}

cooccurrence_cells <- function(co) {
  out <- co
  visible <- !is.na(dua_binary(co$n_ij, co$n_i)) & !is.na(dua_binary(co$n_ij, co$n_j))
  out$n_ij <- ifelse(visible, co$n_ij, NA_real_)
  out$podil <- ifelse(visible, co$n_ij / co$n_i, NA_real_)
  out
}

band_verdict <- function(share) {
  ifelse(
    is.na(share), "nelze",
    ifelse(share < PREVALENCE_BAND[["low"]], "pod",
      ifelse(share > PREVALENCE_BAND[["high"]], "nad", "v")
    )
  )
}

candidate_verdicts <- function(published) {
  band_verdict(published$n / published$celkem)
}

preselection <- function(published) {
  verdicts <- candidate_verdicts(published)
  if (length(verdicts) == 0L || all(verdicts == "nelze")) {
    return(list(kind = "nelze"))
  }
  keys <- function(verdict) published$kandidat[verdicts == verdict]
  list(
    kind = "vyber", in_band = keys("v"), below = keys("pod"), above = keys("nad"),
    unknown = keys("nelze")
  )
}

reconcile_charlson <- function(charlson, prevalence, candidates) {
  hide_category <- character()
  hide_candidate <- character()
  for (key in intersect(names(candidates), prevalence$kandidat)) {
    categories <- intersect(candidates[[key]]$charlson, charlson$kategorie)
    n_candidate <- prevalence$n[prevalence$kandidat == key]
    if (length(categories) == 0L || is.na(n_candidate)) next
    n_category <- charlson$n[match(categories, charlson$kategorie)]
    if (anyNA(n_category)) next
    differences <- c(n_category - n_candidate, sum(n_category) - n_candidate)
    if (any(abs(differences) < DUA_THRESHOLD)) {
      hide_candidate <- c(hide_candidate, key)
      hide_category <- c(hide_category, categories)
    }
  }
  charlson$n[charlson$kategorie %in% hide_category] <- NA_real_
  hidden <- prevalence$kandidat %in% hide_candidate
  prevalence$n[hidden] <- NA_real_
  prevalence$n9[hidden] <- NA_real_
  prevalence$n10[hidden] <- NA_real_
  list(charlson = charlson, prevalence = prevalence)
}

uncovered_charlson <- function(counts, candidates, total) {
  known <- counts[!is.na(counts$n), , drop = FALSE]
  if (nrow(known) == 0L || total <= 0) {
    return(list(kind = "nelze"))
  }
  in_band <- known$kategorie[band_verdict(known$n / total) == "v"]
  covered <- unlist(lapply(candidates, function(entry) entry$charlson))
  list(kind = "audit", categories = setdiff(in_band, covered))
}

redundant_pairs <- function(cells) {
  known <- cells[!is.na(cells$podil), , drop = FALSE]
  hidden <- sum(is.na(cells$podil))
  if (nrow(known) == 0L) {
    return(list(kind = "nelze"))
  }
  pairs <- known[known$podil >= REDUNDANCY_THRESHOLD, c("radek", "sloupec", "podil"), drop = FALSE]
  pairs <- pairs[order(-pairs$podil, pairs$radek, pairs$sloupec), , drop = FALSE]
  rownames(pairs) <- NULL
  list(kind = "dvojice", pairs = pairs, hidden = hidden)
}

era_gaps <- function(published) {
  share9 <- published$n9 / published$celkem9
  share10 <- published$n10 / published$celkem10
  known <- !is.na(share9) & !is.na(share10)
  if (!any(known)) {
    return(list(kind = "nelze"))
  }
  gap <- known & pmax(share9, share10) > ERA_RATIO * pmin(share9, share10)
  gap[is.na(gap)] <- FALSE
  list(
    kind = "rozdily",
    gaps = data.frame(
      kandidat = published$kandidat[gap], podil9 = share9[gap], podil10 = share10[gap],
      stringsAsFactors = FALSE
    ),
    unknown = published$kandidat[!known]
  )
}

symptom_share <- function(era) {
  n <- dua_binary(era$symptom, era$notes)
  if (is.na(n)) {
    return(list(kind = "nelze"))
  }
  list(kind = "share", n = n, total = era$notes, share = n / era$notes)
}

hidden_number <- function(n) {
  ifelse(is.na(n), HIDDEN_TEXT, number_text(n))
}

hidden_percent <- function(n, total) {
  total <- rep_len(total, length(n))
  out <- rep("–", length(n))
  known <- !is.na(n) & !is.na(total)
  out[known] <- percent_text(n[known], total[known])
  out
}

era_table <- function(era, symptom) {
  symptom_n <- if (symptom$kind == "share") symptom$n else NA_real_
  data.frame(
    ukazatel = c(
      "Zprávy vstupního korpusu", "Jen kódy ICD-9", "Jen kódy ICD-10", "Kódy obou verzí",
      "Bez kódu s platnou verzí", "Zprávy s kódem kapitoly R (780–799)"
    ),
    hodnota = c(
      number_text(c(era$notes, era$only9, era$only10, era$both, era$neither)),
      hidden_number(symptom_n)
    ),
    podil = c(
      percent_text(c(era$notes, era$only9, era$only10, era$both, era$neither), era$notes),
      hidden_percent(symptom_n, era$notes)
    ),
    stringsAsFactors = FALSE
  )
}

charlson_published <- function(counts, total) {
  out <- counts
  out$n <- dua_binary(counts$n, total)
  out$popisek <- unname(CHARLSON_LABELS[counts$kategorie])
  out
}

charlson_table <- function(published, total) {
  data.frame(
    kategorie = published$popisek,
    n = hidden_number(published$n),
    podil = hidden_percent(published$n, total),
    stringsAsFactors = FALSE
  )
}

ranking_table <- function(ranking) {
  data.frame(
    kod = ranking$kod,
    nazev = ranking$nazev,
    n = number_text(ranking$n),
    podil = percent_text(ranking$podil, 1),
    stringsAsFactors = FALSE
  )
}

candidate_table <- function(published, candidates) {
  verdict <- candidate_verdicts(published)
  data.frame(
    kandidat = candidate_labels(published$kandidat, candidates),
    n = hidden_number(published$n),
    korpus = hidden_percent(published$n, published$celkem),
    icd9 = hidden_percent(published$n9, published$celkem9),
    icd10 = hidden_percent(published$n10, published$celkem10),
    pasmo = unname(BAND_LABELS[verdict]),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}
