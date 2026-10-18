section_headers_path <- function() config_path("section_headers.json")

read_section_headers <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  keys <- names(raw$sections)
  rows <- lapply(keys, function(key) {
    entry <- raw$sections[[key]]
    synonyms <- as.character(unlist(entry$synonyms))
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
PREAMBLE_CLASS <- "preamble"
OUTSIDE_CLASS <- "mimo lexikon"

# Jediná kopie pravidla shody hlavičky se synonymem; parser sekcí ho převezme doslova.
normalized_sql <- function(column) {
  sprintf("trim(regexp_replace(lower(%s), '\\s+', ' ', 'g'), ' .')", column)
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
    )
    SELECT note_id, header, %s AS norm FROM found WHERE length(header) >= %d
  ", HEADERS_TABLE, input_corpus_sql(con), DBI::dbQuoteString(con, HEADER_PATTERN),
    normalized_sql("header"), MIN_HEADER_CHARS))
  HEADERS_TABLE
}

corpus_size <- function(con) {
  as.numeric(DBI::dbGetQuery(con, sprintf("SELECT count(*) AS n FROM (%s)", input_corpus_sql(con)))$n)
}

header_lexicon <- function(con, lexicon) {
  sections <- lexicon$sections
  preamble <- lexicon$preamble
  rows <- data.frame(
    druh = c(rep("sekce", nrow(sections)), rep("preamble", length(preamble))),
    sekce = c(sections$sekce, rep(NA_character_, length(preamble))),
    label_cs = c(sections$label_cs, rep(NA_character_, length(preamble))),
    synonymum = c(sections$synonymum, preamble),
    stringsAsFactors = FALSE
  )
  DBI::dbWriteTable(con, LEXICON_TABLE, rows, temporary = TRUE, overwrite = TRUE)
  LEXICON_TABLE
}

header_ranking <- function(con, lexicon) {
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
    )
    SELECT c.norm, d.header AS hlavicka, c.n,
      CASE k.druh
        WHEN 'sekce' THEN k.label_cs
        WHEN 'preamble' THEN '%4$s'
        ELSE '%5$s'
      END AS zarazeni
    FROM coverage c
    JOIN display d ON d.norm = c.norm AND d.place = 1
    LEFT JOIN known k ON k.norm = c.norm
    WHERE c.n >= %6$s
    ORDER BY c.n DESC, c.norm
    LIMIT %7$d
  ", headers, lexicon_table, normalized_sql("synonymum"), PREAMBLE_CLASS, OUTSIDE_CLASS,
    format(publishable_floor(total), scientific = FALSE), RANKING_SIZE))
  data.frame(
    norm = out$norm,
    hlavicka = out$hlavicka,
    n = as.numeric(out$n),
    celkem = rep(total, nrow(out)),
    zarazeni = out$zarazeni,
    stringsAsFactors = FALSE
  )
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
        JOIN %2$s l ON l.druh = 'sekce' AND %3$s = h.norm) AS s_sekci
  ", headers, lexicon_table, normalized_sql("l.synonymum")))
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
  grouping <- paste(sprintf("l.%s", keys), collapse = ", ")
  found <- DBI::dbGetQuery(con, sprintf("
    WITH per_note AS (
      SELECT h.note_id, %1$s, count(*) AS vyskytu
      FROM %2$s h JOIN %3$s l ON l.druh = 'sekce' AND %4$s = h.norm
      GROUP BY h.note_id, %1$s
    )
    SELECT %5$s, count(*) AS n, count(*) FILTER (WHERE vyskytu >= 2) AS opakovani
    FROM per_note GROUP BY %5$s
  ", grouping, headers, lexicon_table, normalized_sql("l.synonymum"), paste(keys, collapse = ", ")))
  key_of <- function(df) do.call(paste, c(unname(as.list(df[keys])), sep = "\r"))
  rows <- unique(lexicon$sections[c(keys, "label_cs")])
  at <- match(key_of(rows), key_of(found))
  count <- function(column) {
    out <- as.numeric(found[[column]][at])
    out[is.na(out)] <- 0
    out
  }
  data.frame(
    rows[keys], label_cs = rows$label_cs,
    n = count("n"), opakovani = count("opakovani"), celkem = rep(total, nrow(rows)),
    stringsAsFactors = FALSE, row.names = NULL
  )
}

section_coverage <- function(con, lexicon) {
  out <- lexicon_coverage(con, lexicon, "sekce")
  out[c("sekce", "label_cs", "n", "opakovani", "celkem")]
}

synonym_coverage <- function(con, lexicon) {
  out <- lexicon_coverage(con, lexicon, c("sekce", "synonymum"))
  out[c("sekce", "synonymum", "n", "opakovani", "celkem")]
}
