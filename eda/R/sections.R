section_headers_path <- function() config_path("section_headers.json")

read_section_headers <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  keys <- names(raw$sections)
  rows <- lapply(keys, function(key) {
    entry <- raw$sections[[key]]
    synonyms <- as.character(unlist(entry$synonyms))
    if (length(synonyms) == 0L || is.null(entry$label_cs) || !nzchar(entry$label_cs)) {
      stop(sprintf("Sekce %s v lexikonu potřebuje label_cs a aspoň jedno synonymum.", key))
    }
    data.frame(
      sekce = rep(key, length(synonyms)),
      label_cs = rep(entry$label_cs, length(synonyms)),
      synonymum = synonyms,
      stringsAsFactors = FALSE
    )
  })
  sections <- do.call(rbind, rows)
  list(sections = sections, preamble = as.character(unlist(raw$preamble)))
}

# Číslice ve výrazu nejsou, aby "10:30" nebyl hlavička; odsazení se připouští, protože hlavičky bývají odsazené.
HEADER_PATTERN <- "(?m)^[ \\t]*([A-Z][A-Za-z][A-Za-z \\-/&()'.]{0,47}):"
MIN_HEADER_CHARS <- 3L
MIN_COVERAGE <- 0.01
RANKING_SIZE <- 40L
CANDIDATE_COVERAGE <- 0.5
REPEAT_SHARE <- 0.05
HEADERS_TABLE <- "note_headers_input"
LEXICON_TABLE <- "header_lexicon_input"
KIND_SECTION <- "sekce"
KIND_PREAMBLE <- "preamble"
KIND_OUTSIDE <- "mimo"
KIND_LABELS <- c(
  sekce = "sekce lexikonu", preamble = "administrativní pole", mimo = "mimo lexikon"
)

# Pravidlo shody hlavičky se synonymem má protějšek v src/mediparse/domain/section_headers.py
# (normalized, HEADER_PATTERN); změna pravidla se dělá v obou.
normalized_sql <- function(column) {
  sprintf("rtrim(regexp_replace(lower(%s), '\\s+', ' ', 'g'), ' .')", column)
}

publishable_floor <- function(total) {
  pmax(DUA_THRESHOLD, ceiling(MIN_COVERAGE * total))
}

note_headers <- function(con) {
  if (HEADERS_TABLE %in% DBI::dbListTables(con)) {
    return(HEADERS_TABLE)
  }
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    WITH corpus AS (%s),
    found AS (
      SELECT c.note_id, unnest(regexp_extract_all(d.text, %s, 1)) AS header
      FROM corpus c JOIN discharge d ON d.note_id = c.note_id
    ),
    normalized AS (SELECT note_id, header, %s AS norm FROM found)
    SELECT note_id, header, norm FROM normalized WHERE length(norm) >= %d
  ", HEADERS_TABLE, input_corpus_sql(con), DBI::dbQuoteString(con, HEADER_PATTERN),
    normalized_sql("header"), MIN_HEADER_CHARS))
  HEADERS_TABLE
}

header_lexicon <- function(con, lexicon) {
  sections <- lexicon$sections
  preamble <- lexicon$preamble
  rows <- data.frame(
    poradi = seq_len(nrow(sections) + length(preamble)),
    druh = c(rep(KIND_SECTION, nrow(sections)), rep(KIND_PREAMBLE, length(preamble))),
    sekce = c(sections$sekce, rep(NA_character_, length(preamble))),
    label_cs = c(sections$label_cs, rep(NA_character_, length(preamble))),
    synonymum = c(sections$synonymum, preamble),
    stringsAsFactors = FALSE
  )
  DBI::dbWriteTable(con, LEXICON_TABLE, rows, temporary = TRUE, overwrite = TRUE)
  collisions <- DBI::dbGetQuery(con, sprintf("
    SELECT string_agg(synonymum, ', ' ORDER BY poradi) AS synonyma
    FROM %s GROUP BY %s HAVING count(*) > 1
  ", LEXICON_TABLE, normalized_sql("synonymum")))
  if (nrow(collisions) > 0L) {
    stop(sprintf(
      "Lexikon má synonyma, která se po normalizaci shodují: %s.",
      paste(collisions$synonyma, collapse = "; ")
    ))
  }
  LEXICON_TABLE
}

ranked_headers <- function(con, lexicon, minimum, kind = NULL, limit = NULL) {
  headers <- note_headers(con)
  lexicon_table <- header_lexicon(con, lexicon)
  total <- corpus_size(con)
  out <- DBI::dbGetQuery(con, sprintf("
    WITH spelled AS (
      SELECT norm, header, count(DISTINCT note_id) AS used
      FROM %1$s GROUP BY norm, header
    ),
    display AS (
      SELECT norm, header,
        row_number() OVER (PARTITION BY norm ORDER BY used DESC, header) AS place
      FROM spelled
    ),
    coverage AS (
      SELECT norm, count(DISTINCT note_id) AS n FROM %1$s GROUP BY norm
    ),
    known AS (
      SELECT %3$s AS norm, any_value(druh) AS druh, any_value(label_cs) AS label_cs
      FROM %2$s GROUP BY 1
    ),
    ranked AS (
      SELECT c.norm, d.header AS hlavicka, c.n,
        coalesce(k.druh, '%4$s') AS druh, k.label_cs
      FROM coverage c
      JOIN display d ON d.norm = c.norm AND d.place = 1
      LEFT JOIN known k ON k.norm = c.norm
    )
    SELECT * FROM ranked
    WHERE n >= %5$s AND (%6$s)
    ORDER BY n DESC, norm
    %7$s
  ", headers, lexicon_table, normalized_sql("synonymum"), KIND_OUTSIDE,
    format(minimum(total), scientific = FALSE),
    if (is.null(kind)) "true" else sprintf("druh = '%s'", kind),
    if (is.null(limit)) "" else sprintf("LIMIT %d", limit)))
  shown <- ifelse(out$druh == KIND_SECTION, out$label_cs, unname(KIND_LABELS[out$druh]))
  data.frame(
    norm = out$norm,
    hlavicka = out$hlavicka,
    n = as.numeric(out$n),
    celkem = rep(total, nrow(out)),
    druh = out$druh,
    zarazeni = shown,
    stringsAsFactors = FALSE
  )
}

header_ranking <- function(con, lexicon) {
  ranked_headers(con, lexicon, publishable_floor, limit = RANKING_SIZE)
}

header_candidates <- function(con, lexicon) {
  at_least_half <- function(total) pmax(publishable_floor(total), CANDIDATE_COVERAGE * total)
  ranked_headers(con, lexicon, at_least_half, kind = KIND_OUTSIDE)
}

header_totals <- function(con, lexicon) {
  headers <- note_headers(con)
  lexicon_table <- header_lexicon(con, lexicon)
  total <- corpus_size(con)
  row <- DBI::dbGetQuery(con, sprintf("
    SELECT
      (SELECT count(DISTINCT norm) FROM %1$s) AS unikatni,
      (SELECT count(DISTINCT note_id) FROM %1$s) AS s_hlavickou,
      (SELECT count(DISTINCT h.note_id) FROM %1$s h
        JOIN %2$s l ON l.druh = '%4$s' AND %3$s = h.norm) AS s_sekci
  ", headers, lexicon_table, normalized_sql("l.synonymum"), KIND_SECTION))
  list(
    celkem = total,
    unikatni = as.numeric(row$unikatni),
    s_sekci = as.numeric(row$s_sekci),
    bez_hlavicky = total - as.numeric(row$s_hlavickou)
  )
}

lexicon_coverage <- function(con, lexicon, keys) {
  headers <- note_headers(con)
  lexicon_table <- header_lexicon(con, lexicon)
  total <- corpus_size(con)
  qualified <- function(prefix) paste(sprintf("%s.%s", prefix, keys), collapse = ", ")
  same_keys <- paste(sprintf("r.%1$s = f.%1$s", keys), collapse = " AND ")
  out <- DBI::dbGetQuery(con, sprintf("
    WITH per_note AS (
      SELECT h.note_id, %1$s, count(*) AS vyskytu
      FROM %2$s h JOIN %3$s l ON l.druh = '%4$s' AND %5$s = h.norm
      GROUP BY h.note_id, %1$s
    ),
    found AS (
      SELECT %6$s, count(*) AS n, count(*) FILTER (WHERE vyskytu >= 2) AS opakovani
      FROM per_note GROUP BY %6$s
    ),
    listed AS (
      SELECT %7$s, any_value(l.label_cs) AS label_cs, min(l.poradi) AS poradi
      FROM %3$s l WHERE l.druh = '%4$s' GROUP BY %7$s
    )
    SELECT %8$s, r.label_cs, coalesce(f.n, 0) AS n, coalesce(f.opakovani, 0) AS opakovani
    FROM listed r LEFT JOIN found f ON %9$s
    ORDER BY r.poradi
  ", qualified("l"), headers, lexicon_table, KIND_SECTION, normalized_sql("l.synonymum"),
    paste(keys, collapse = ", "), qualified("l"), qualified("r"), same_keys))
  out[c("n", "opakovani")] <- lapply(out[c("n", "opakovani")], as.numeric)
  out$celkem <- rep(total, nrow(out))
  out
}

section_coverage <- function(con, lexicon) {
  out <- lexicon_coverage(con, lexicon, "sekce")
  out[c("sekce", "label_cs", "n", "opakovani", "celkem")]
}

synonym_coverage <- function(con, lexicon) {
  out <- lexicon_coverage(con, lexicon, c("sekce", "synonymum"))
  out[c("sekce", "synonymum", "n", "opakovani", "celkem")]
}

SYNONYM_FORMS <- c("synonymum", "synonyma", "synonym")
UNDECIDABLE_TEXT <- "nelze posoudit, protože vstupní korpus je prázdný."

header_table <- function(ranking) {
  data.frame(
    hlavicka = ranking$hlavicka,
    zarazeni = ranking$zarazeni,
    n = number_text(ranking$n),
    podil = percent_text(ranking$n, ranking$celkem),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

ranking_bars <- function(ranking) {
  data.frame(
    hlavicka = factor(ranking$hlavicka, levels = rev(ranking$hlavicka)),
    podil = ranking$n / ranking$celkem,
    druh = factor(KIND_LABELS[ranking$druh], levels = KIND_LABELS),
    stringsAsFactors = FALSE
  )
}

section_table <- function(coverage) {
  data.frame(
    sekce = coverage$sekce,
    label_cs = coverage$label_cs,
    pokryti = percent_text(coverage$n, coverage$celkem),
    opakovani = percent_text(coverage$opakovani, coverage$celkem),
    stringsAsFactors = FALSE
  )
}

synonym_table <- function(coverage) {
  data.frame(
    sekce = coverage$sekce,
    synonymum = coverage$synonymum,
    pokryti = percent_text(coverage$n, coverage$celkem),
    opakovani = percent_text(coverage$opakovani, coverage$celkem),
    stringsAsFactors = FALSE
  )
}

coverage_threshold_text <- function(share) paste0(round(100 * share), NBSP, "%")

candidates_text <- function(candidates, totals) {
  if (totals$celkem == 0) {
    return(paste("Kandidáty na doplnění lexikonu", UNDECIDABLE_TEXT))
  }
  sprintf(
    "Kandidáti na doplnění lexikonu (hlavičky mimo lexikon s pokrytím aspoň %s): %s.",
    coverage_threshold_text(CANDIDATE_COVERAGE),
    join_labels(sprintf("`%s` (%s)", candidates$hlavicka, percent_label(candidates$n, candidates$celkem)))
  )
}

dead_synonyms_text <- function(synonyms, totals) {
  if (totals$celkem == 0) {
    return(paste("Mrtvá synonyma", UNDECIDABLE_TEXT))
  }
  dead <- synonyms$synonymum[synonyms$n == 0]
  if (length(dead) == 0L) {
    return("Lexikon neobsahuje žádné mrtvé synonymum.")
  }
  sprintf(
    "Bez jediné zprávy %s %s: %s.",
    plural_form(length(dead), c("je", "jsou", "je")),
    count_phrase(length(dead), SYNONYM_FORMS), join_labels(sprintf("`%s`", dead))
  )
}

repeated_synonyms_text <- function(synonyms, totals) {
  if (totals$celkem == 0) {
    return(paste("Opakování synonym", UNDECIDABLE_TEXT))
  }
  share <- synonyms$opakovani / synonyms$celkem
  repeated <- synonyms[share >= REPEAT_SHARE, , drop = FALSE]
  if (nrow(repeated) == 0L) {
    return(sprintf(
      "Žádné synonymum se neopakuje ve zprávách od %s korpusu.", coverage_threshold_text(REPEAT_SHARE)
    ))
  }
  sprintf(
    "V aspoň %s zpráv %s %s: %s.",
    coverage_threshold_text(REPEAT_SHARE),
    plural_form(nrow(repeated), c("se opakuje", "se opakují", "se opakuje")),
    count_phrase(nrow(repeated), SYNONYM_FORMS),
    join_labels(sprintf("`%s` (%s)", repeated$synonymum, percent_label(repeated$opakovani, repeated$celkem)))
  )
}

coverage_text <- function(totals) {
  if (totals$celkem == 0) {
    return(paste("Pokrytí lexikonu", UNDECIDABLE_TEXT))
  }
  sprintf(
    "Aspoň jednu sekci lexikonu má %s zpráv korpusu, žádnou hlavičku %s. Korpus obsahuje %s.",
    percent_label(totals$s_sekci, totals$celkem),
    percent_label(totals$bez_hlavicky, totals$celkem),
    count_phrase(totals$unikatni, c("unikátní hlavičku", "unikátní hlavičky", "unikátních hlaviček"))
  )
}
