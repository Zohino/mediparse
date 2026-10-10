DUA_THRESHOLD <- 10L
NBSP <- "\u00a0"
DUA_OTHER <- sprintf("ostatní (n < %d)", DUA_THRESHOLD)

dua_count <- function(n) {
  ifelse(
    n < DUA_THRESHOLD,
    sprintf("<%s%d", NBSP, DUA_THRESHOLD),
    format(n, big.mark = NBSP, scientific = FALSE, trim = TRUE)
  )
}

dua_table <- function(df) {
  if (!"n" %in% names(df)) {
    stop("Tabulka musí být agregovaná: chybí sloupec n s počtem záznamů.")
  }
  if (!is.numeric(df$n) || any(df$n != round(df$n))) {
    stop("Sloupec n musí být celočíselný.")
  }
  labels <- setdiff(names(df), "n")
  if (!all(vapply(df[labels], function(column) is.character(column) || is.factor(column), logical(1)))) {
    stop("Popisky smí být jen text nebo faktor: další číselný sloupec by nešel potlačit.")
  }
  small <- df$n < DUA_THRESHOLD
  kept <- df[!small, , drop = FALSE]
  out <- as.data.frame(lapply(kept[labels], as.character), stringsAsFactors = FALSE)
  counts <- kept$n
  if (any(small)) {
    other <- as.data.frame(
      lapply(kept[labels], function(column) DUA_OTHER),
      stringsAsFactors = FALSE
    )
    out <- rbind(out, other)
    counts <- c(counts, sum(df$n[small]))
  }
  out$n <- dua_count(counts)
  out
}
