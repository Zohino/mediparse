text_mentions_path <- function() config_path("text_mentions.json")

read_text_mentions <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  list(
    negation = list(
      cues = as.character(unlist(raw$negation$cues)),
      window_chars = as.integer(raw$negation$window_chars)
    ),
    symptoms = list(keywords_en = as.character(unlist(raw$symptoms$keywords_en)))
  )
}
