OTHER_VALUE <- "jiné"
OTHER_RACE <- "jiná"
MASKED_AGE <- 91L
AGE_CAP <- 90L
DEMOGRAPHIC_TABLE <- "demographic_notes"

AGE_BANDS <- c(
  "do 29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–89", "90+", UNKNOWN_LABEL
)

GENDER_LABELS <- c("ženy", "muži", OTHER_VALUE, UNKNOWN_LABEL)

ATTRIBUTES <- list(
  list(
    name = "Pohlaví", column = "gender", per = "subject_id",
    levels = GENDER_LABELS
  ),
  list(name = "Věkové pásmo", column = "age_band", per = "note_id", levels = AGE_BANDS),
  list(
    name = "Pojištění", column = "insurance", per = "note_id",
    levels = c("Medicare", "Medicaid", "soukromé", "jiné pojištění", OTHER_VALUE, UNKNOWN_LABEL)
  ),
  list(
    name = "Rodinný stav", column = "marital", per = "note_id",
    levels = c(
      "v manželství", "svobodní", "ovdovělí", "rozvedení", OTHER_VALUE, UNKNOWN_LABEL
    )
  ),
  list(
    name = "Rasa", column = "race", per = "note_id",
    levels = c("bílá", "černá", "hispánská", "asijská", OTHER_RACE, UNKNOWN_LABEL)
  ),
  list(
    name = "Jazyk", column = "language", per = "note_id",
    levels = c("angličtina", "jiný jazyk", UNKNOWN_LABEL)
  )
)

NON_GROUP_LABELS <- c(OTHER_VALUE, OTHER_RACE, UNKNOWN_LABEL)

DEMOGRAPHIC_SQL <- "
  CREATE TEMP TABLE %s AS
  WITH corpus AS (%s)
  SELECT
    c.note_id,
    c.subject_id,
    CASE
      WHEN p.gender IS NULL THEN '%s'
      WHEN p.gender = 'F' THEN 'ženy'
      WHEN p.gender = 'M' THEN 'muži'
      ELSE '%s'
    END AS gender,
    CASE
      WHEN coalesce(TRY_CAST(p.anchor_age AS INTEGER) = %d, false) THEN %d
      WHEN TRY_CAST(p.anchor_age AS INTEGER) IS NULL
        OR TRY_CAST(a.admittime AS TIMESTAMP) IS NULL
        OR TRY_CAST(p.anchor_year AS INTEGER) IS NULL THEN NULL
      ELSE least(
        TRY_CAST(p.anchor_age AS INTEGER)
          + year(TRY_CAST(a.admittime AS TIMESTAMP))
          - TRY_CAST(p.anchor_year AS INTEGER),
        %d
      )
    END AS age,
    coalesce(TRY_CAST(p.anchor_age AS INTEGER) = %d, false) AS age_masked,
    CASE
      WHEN a.insurance IS NULL THEN '%s'
      WHEN a.insurance = 'Medicare' THEN 'Medicare'
      WHEN a.insurance = 'Medicaid' THEN 'Medicaid'
      WHEN a.insurance = 'Private' THEN 'soukromé'
      WHEN a.insurance = 'Other' THEN 'jiné pojištění'
      ELSE '%s'
    END AS insurance,
    CASE
      WHEN a.marital_status IS NULL THEN '%s'
      WHEN a.marital_status = 'MARRIED' THEN 'v manželství'
      WHEN a.marital_status = 'SINGLE' THEN 'svobodní'
      WHEN a.marital_status = 'WIDOWED' THEN 'ovdovělí'
      WHEN a.marital_status = 'DIVORCED' THEN 'rozvedení'
      ELSE '%s'
    END AS marital,
    CASE
      WHEN a.race IS NULL THEN '%s'
      WHEN a.race LIKE 'WHITE%%' THEN 'bílá'
      WHEN a.race LIKE 'BLACK%%' THEN 'černá'
      WHEN a.race LIKE 'HISPANIC%%' OR a.race = 'SOUTH AMERICAN' THEN 'hispánská'
      WHEN a.race LIKE 'ASIAN%%' THEN 'asijská'
      WHEN a.race IN ('UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER')
        THEN '%s'
      ELSE '%s'
    END AS race,
    CASE
      WHEN a.language IS NULL OR a.language = '?' THEN '%s'
      WHEN a.language = 'English' THEN 'angličtina'
      ELSE 'jiný jazyk'
    END AS language,
    a.hadm_id IS NOT NULL AS has_admission,
    p.subject_id IS NOT NULL AS has_patient
  FROM corpus c
  LEFT JOIN patients p ON p.subject_id = c.subject_id
  LEFT JOIN admissions a ON a.hadm_id = c.hadm_id
"

demographic_notes <- function(con) {
  if (DEMOGRAPHIC_TABLE %in% DBI::dbListTables(con)) {
    return(DEMOGRAPHIC_TABLE)
  }
  DBI::dbExecute(con, sprintf(
    DEMOGRAPHIC_SQL,
    DEMOGRAPHIC_TABLE, input_corpus_sql(con),
    UNKNOWN_LABEL, OTHER_VALUE,
    MASKED_AGE, AGE_CAP, AGE_CAP, MASKED_AGE,
    UNKNOWN_LABEL, OTHER_VALUE,
    UNKNOWN_LABEL, OTHER_VALUE,
    UNKNOWN_LABEL, UNKNOWN_LABEL, OTHER_RACE,
    UNKNOWN_LABEL
  ))
  DEMOGRAPHIC_TABLE
}

age_band <- function(age) {
  out <- rep(UNKNOWN_LABEL, length(age))
  known <- !is.na(age)
  decade <- pmin(floor(age[known] / 10) * 10, AGE_CAP)
  out[known] <- ifelse(
    decade >= AGE_CAP, "90+",
    ifelse(decade < 30, "do 29", sprintf("%d–%d", as.integer(decade), as.integer(decade) + 9L))
  )
  out
}

demographic_summary <- function(con) {
  source_table <- demographic_notes(con)
  row <- query_row(con, sprintf("
    SELECT
      count(DISTINCT subject_id) AS patients,
      count(*) AS notes,
      quantile_cont(age, 0.25) AS p25,
      quantile_cont(age, 0.5) AS p50,
      quantile_cont(age, 0.75) AS p75,
      count(*) FILTER (WHERE NOT has_admission) AS no_admission,
      count(*) FILTER (WHERE NOT has_patient) AS no_patient,
      count(*) FILTER (WHERE age_masked) AS masked
    FROM %s
  ", source_table))
  lapply(row, as.numeric)
}

attribute_counts <- function(con) {
  source_table <- demographic_notes(con)
  notes <- DBI::dbGetQuery(con, sprintf("
    SELECT age, subject_id, note_id, gender, insurance, marital, race, language FROM %s
  ", source_table))
  notes$age_band <- age_band(notes$age)
  parts <- lapply(ATTRIBUTES, function(attribute) {
    unit <- unique(notes[c(attribute$per, attribute$column)])
    counts <- table(factor(unit[[attribute$column]], levels = attribute$levels))
    counts <- counts[counts > 0]
    data.frame(
      atribut = rep(attribute$name, length(counts)),
      kategorie = names(counts),
      n = as.numeric(counts),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  out
}

age_gender_counts <- function(con) {
  source_table <- demographic_notes(con)
  rows <- DBI::dbGetQuery(con, sprintf("
    SELECT gender AS pohlavi, age, count(*) AS n FROM %s GROUP BY gender, age
  ", source_table))
  if (nrow(rows) == 0L) {
    return(data.frame(
      pohlavi = character(), pasmo = character(), n = numeric(), stringsAsFactors = FALSE
    ))
  }
  rows$pasmo <- age_band(rows$age)
  grouped <- stats::aggregate(n ~ pohlavi + pasmo, data = rows, FUN = sum)
  grouped <- grouped[
    order(match(grouped$pohlavi, GENDER_LABELS), match(grouped$pasmo, AGE_BANDS)),
    c("pohlavi", "pasmo", "n"),
    drop = FALSE
  ]
  grouped$n <- as.numeric(grouped$n)
  rownames(grouped) <- NULL
  grouped
}

majority_gender <- function(published) {
  gender <- published[published$atribut == "Pohlaví", , drop = FALSE]
  known <- gender[gender$kategorie %in% c("ženy", "muži"), , drop = FALSE]
  if (nrow(known) < 2L) {
    return(list(kind = "nelze"))
  }
  top <- known[which.max(known$n), ]
  list(
    kind = "majority", label = top$kategorie, n = top$n, total = sum(known$n),
    share = top$n / sum(known$n)
  )
}

largest_race <- function(published, total) {
  race <- published[published$atribut == "Rasa", , drop = FALSE]
  groups <- race[!race$kategorie %in% c(NON_GROUP_LABELS, DUA_OTHER), , drop = FALSE]
  if (nrow(groups) == 0L) {
    return(list(kind = "nelze"))
  }
  top <- groups[which.max(groups$n), ]
  list(kind = "group", label = top$kategorie, n = top$n, total = total, share = top$n / total)
}

band90_count <- function(published) {
  band <- published$n[published$atribut == "Věkové pásmo" & published$kategorie == "90+"]
  if (length(band) == 0L) NA_real_ else band
}

masked_text <- function(masked, band90) {
  safe <- !is.na(band90) & masked >= DUA_THRESHOLD & band90 - masked >= DUA_THRESHOLD
  ifelse(safe, number_text(masked), sprintf("méně než %d", DUA_THRESHOLD))
}

drawable_cells <- function(age_gender) {
  by_gender <- dua_suppress(age_gender, by = "pohlavi")
  by_band <- dua_suppress(age_gender, by = "pasmo")
  key <- function(cells) paste(cells$pohlavi, cells$pasmo)
  visible_band <- key(by_band)[by_band$pohlavi != DUA_OTHER & by_band$pasmo != DUA_OTHER]
  kept <- by_gender[
    by_gender$pasmo != DUA_OTHER & key(by_gender) %in% visible_band, , drop = FALSE
  ]
  rownames(kept) <- NULL
  kept
}

characteristics_table <- function(published, summary) {
  denominator <- ifelse(published$atribut == "Pohlaví", summary$patients, summary$notes)
  first <- !duplicated(published$atribut)
  data.frame(
    atribut = ifelse(first, published$atribut, ""),
    kategorie = published$kategorie,
    n = number_text(published$n),
    podil = percent_text(published$n, denominator),
    stringsAsFactors = FALSE
  )
}

summary_table <- function(summary) {
  quartiles <- if (is.na(summary$p50)) {
    "–"
  } else {
    sprintf(
      "%s (%s–%s)", mean_text(summary$p50), mean_text(summary$p25), mean_text(summary$p75)
    )
  }
  data.frame(
    ukazatel = c(
      "Pacienti", "Zprávy", "Medián věku při přijetí (kvartily)",
      "Zprávy bez záznamu přijetí", "Zprávy bez záznamu pacienta"
    ),
    hodnota = c(
      number_text(summary$patients), number_text(summary$notes), quartiles,
      number_text(summary$no_admission), number_text(summary$no_patient)
    ),
    stringsAsFactors = FALSE
  )
}
