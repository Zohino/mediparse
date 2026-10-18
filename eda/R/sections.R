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
