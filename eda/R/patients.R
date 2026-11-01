query_row <- function(con, sql) {
  as.list(DBI::dbGetQuery(con, sql))
}

patient_summary <- function(con, table = "discharge") {
  source <- DBI::dbQuoteIdentifier(con, table)
  query_row(con, sprintf("
    WITH per_patient AS (
      SELECT subject_id, count(*) AS notes FROM %s GROUP BY subject_id
    )
    SELECT
      (SELECT count(*) FROM per_patient) AS patients,
      (SELECT count(DISTINCT hadm_id) FROM %s) AS admissions,
      coalesce((SELECT sum(notes) FROM per_patient), 0) AS notes,
      (SELECT count(*) FROM per_patient WHERE notes > 1) AS multi_note_patients
  ", source, source))
}

capped_distribution <- function(con, sql, cap) {
  rows <- DBI::dbGetQuery(con, sprintf("
    WITH per_unit AS (%s)
    SELECT least(notes, %d) AS bucket, count(*) AS n
    FROM per_unit
    GROUP BY bucket
    ORDER BY bucket
  ", sql, as.integer(cap)))
  data.frame(
    kos = ifelse(rows$bucket < cap, as.character(rows$bucket), paste0(cap, "+")),
    n = as.numeric(rows$n),
    stringsAsFactors = FALSE
  )
}

notes_per_patient <- function(con, cap = 10) {
  capped_distribution(
    con, "SELECT count(*) AS notes FROM discharge GROUP BY subject_id", cap
  )
}

notes_per_admission <- function(con, cap = 3) {
  capped_distribution(
    con,
    "SELECT count(*) AS notes FROM discharge WHERE hadm_id IS NOT NULL GROUP BY hadm_id",
    cap
  )
}

hadm_coverage <- function(con, tables = c("discharge", "radiology")) {
  rows <- lapply(tables, function(table) {
    out <- DBI::dbGetQuery(con, sprintf("
      WITH flagged AS (
        SELECT
          t.hadm_id IS NULL AS no_hadm,
          t.hadm_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM diagnoses_icd i WHERE i.hadm_id = t.hadm_id
          ) AS has_icd
        FROM %s t
      )
      SELECT
        count(*) AS zprav,
        count(*) FILTER (WHERE no_hadm) AS bez_hadm,
        count(*) FILTER (WHERE NOT no_hadm AND NOT has_icd) AS bez_icd,
        count(*) FILTER (WHERE has_icd) AS s_icd
      FROM flagged
    ", DBI::dbQuoteIdentifier(con, table)))
    cbind(tabulka = table, out, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  out[-1] <- lapply(out[-1], as.numeric)
  out
}

coverage_verdict <- function(coverage) {
  pair <- coverage[match(c("discharge", "radiology"), coverage$tabulka), ]
  if (anyNA(pair$tabulka) || any(pair$zprav == 0)) {
    return("nelze")
  }
  share <- pair$s_icd / pair$zprav
  if (share[1] >= share[2]) "discharge" else "radiology"
}

ICD_ERAS <- c("jen ICD-9", "jen ICD-10", "obě éry", "bez kódu")

ERA_CASE <- "
  CASE
    WHEN v9 AND v10 THEN 'obě éry'
    WHEN v9 THEN 'jen ICD-9'
    WHEN v10 THEN 'jen ICD-10'
    ELSE 'bez kódu'
  END
"

ICD_ERA_CTE <- "
  WITH admission_subject AS (
    SELECT hadm_id, min(subject_id) AS subject_id
    FROM discharge
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id
  ),
  admission_era AS (
    SELECT
      a.hadm_id,
      a.subject_id,
      coalesce(bool_or(i.icd_version = '9'), false) AS v9,
      coalesce(bool_or(i.icd_version = '10'), false) AS v10
    FROM admission_subject a
    LEFT JOIN diagnoses_icd i ON i.hadm_id = a.hadm_id
    GROUP BY a.hadm_id, a.subject_id
  )
"

era_order <- function(era) match(era, ICD_ERAS)

icd_era <- function(con) {
  out <- DBI::dbGetQuery(con, paste0(ICD_ERA_CTE, sprintf("
    SELECT %s AS era, count(*) AS n FROM admission_era GROUP BY era
  ", ERA_CASE)))
  out <- out[order(era_order(out$era)), , drop = FALSE]
  out$n <- as.numeric(out$n)
  rownames(out) <- NULL
  out
}

icd_era_by_group <- function(con) {
  out <- DBI::dbGetQuery(con, paste0(ICD_ERA_CTE, sprintf("
    SELECT
      coalesce(p.anchor_year_group, 'neznámé') AS obdobi,
      %s AS era,
      count(*) AS n
    FROM admission_era e
    LEFT JOIN patients p ON p.subject_id = e.subject_id
    GROUP BY obdobi, era
  ", ERA_CASE)))
  out <- out[order(out$obdobi == "neznámé", out$obdobi, era_order(out$era)), , drop = FALSE]
  out$n <- as.numeric(out$n)
  rownames(out) <- NULL
  out
}

era_transition <- function(era_group) {
  known <- era_group[
    era_group$obdobi != "neznámé" & era_group$n > 0,
    , drop = FALSE
  ]
  periods_9 <- unique(known$obdobi[known$era == "jen ICD-9"])
  periods_10 <- unique(known$obdobi[known$era == "jen ICD-10"])
  both <- intersect(periods_9, periods_10)
  kind <- if (length(both) > 0) {
    "overlap"
  } else if (length(periods_9) > 0 && length(periods_10) > 0) {
    "separated"
  } else {
    "none"
  }
  list(kind = kind, periods_9 = periods_9, periods_10 = periods_10, both = both)
}
