# Zrcadlo MIN_CELL v src/mediparse/domain/dua.py; změna se dělá v obou souborech.
DUA_THRESHOLD <- 10L
UNKNOWN_LABEL <- "neuvedeno"
NBSP <- "\u00a0"

format_number <- function(x) {
  format(x, big.mark = NBSP, scientific = FALSE, trim = TRUE)
}

dua_decimal <- function(x) {
  format(round(x, 1), nsmall = 1, decimal.mark = ",", trim = TRUE)
}

number_text <- function(x) {
  ifelse(is.na(x), "–", format_number(x))
}

percent_text <- function(n, total) {
  total <- rep_len(total, length(n))
  out <- rep("–", length(n))
  known <- total > 0
  out[known] <- dua_decimal(100 * n[known] / total[known])
  # Nenulový počet zaokrouhlený na 0,0 by vedle čísla zpráv vypadal jako rozpor (22 zpráv, 0,0 %).
  out[known & n > 0 & out == dua_decimal(0)] <- paste0("<", NBSP, dua_decimal(0.1))
  out
}

percent_label <- function(n, total) {
  share <- percent_text(n, total)
  ifelse(share == "–", share, paste0(share, NBSP, "%"))
}

share_table <- function(df) {
  out <- df
  out$n <- number_text(df$n)
  out$podil <- percent_text(df$n, sum(df$n))
  out
}

dua_binary <- function(n, total) {
  total <- rep_len(total, length(n))
  hidden <- is.na(n) | is.na(total) | n < DUA_THRESHOLD | total - n < DUA_THRESHOLD
  ifelse(hidden, NA_real_, as.numeric(n))
}
