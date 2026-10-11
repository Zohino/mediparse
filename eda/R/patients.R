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

COVERAGE_GROUPS <- c("bez_hadm", "bez_icd", "s_icd")

coverage_mask <- function(coverage) {
  rows <- lapply(seq_len(nrow(coverage)), function(row) {
    dua_hidden(unlist(coverage[row, COVERAGE_GROUPS], use.names = FALSE))
  })
  mask <- matrix(
    unlist(rows), ncol = length(COVERAGE_GROUPS), byrow = TRUE,
    dimnames = list(NULL, COVERAGE_GROUPS)
  )
  as.data.frame(mask)
}

coverage_verdict <- function(coverage) {
  pair <- coverage[match(c("discharge", "radiology"), coverage$tabulka), ]
  if (anyNA(pair$tabulka)) {
    return("nelze")
  }
  if (any(pair$zprav == 0) || any(coverage_mask(pair)$s_icd)) {
    return("nelze")
  }
  share <- pair$s_icd / pair$zprav
  if (share[1] >= share[2]) "discharge" else "radiology"
}
