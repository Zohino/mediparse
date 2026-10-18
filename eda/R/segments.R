SECTIONS_TABLE <- "note_sections_input"
SECTION_COLUMNS <- c(
  note_id = "VARCHAR", position = "INTEGER", section = "VARCHAR",
  span_chars = "INTEGER", body_chars = "INTEGER"
)

sections_module <- function() reticulate::import("mediparse.entrypoints.eda_sections")

# Texty a identifikátory musí do Pythonu jako nepojmenovaný seznam: pojmenovaný vektor
# reticulate převede na slovník, duplicitní jméno (dvě shodné zprávy) se v něm sloučí
# a zip() pak spadne na nestejných délkách; vektor délky 1 by se zase stal řetězcem.
section_rows <- function(module, lexicon_path, chunk) {
  path <- reticulate::import("pathlib")$Path(lexicon_path)
  rows <- module$note_sections(
    path, as.list(unname(chunk$note_id)), as.list(unname(chunk$text))
  )
  # Prázdný výsledek by reticulate vrátil jako data.frame bez sloupců, proto typy skládáme sami.
  data.frame(
    note_id = as.character(unlist(rows$note_id)),
    position = as.integer(unlist(rows$position)),
    section = as.character(unlist(rows$section)),
    span_chars = as.integer(unlist(rows$span_chars)),
    body_chars = as.integer(unlist(rows$body_chars)),
    stringsAsFactors = FALSE
  )
}

segment_corpus <- function(con, module, lexicon_path) {
  rows_sql <- sprintf(
    "SELECT c.note_id, d.text FROM (%s) c JOIN discharge d ON d.note_id = c.note_id",
    input_corpus_sql(con)
  )
  batched_table(
    con, SECTIONS_TABLE, SECTION_COLUMNS, rows_sql,
    function(chunk) section_rows(module, lexicon_path, chunk)
  )
}
