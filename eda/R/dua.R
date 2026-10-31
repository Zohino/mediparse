# Zrcadlo MIN_CELL v src/mediparse/domain/dua.py; změna se dělá v obou souborech.
DUA_THRESHOLD <- 10L
UNKNOWN_LABEL <- "neuvedeno"
NBSP <- "\u00a0"
DUA_OTHER <- sprintf("ostatní (n < %d)", DUA_THRESHOLD)

format_number <- function(x) {
  format(x, big.mark = NBSP, scientific = FALSE, trim = TRUE)
}

dua_hidden <- function(n) {
  hidden <- n < DUA_THRESHOLD
  total <- sum(n[hidden])
  if (total == 0 || total >= DUA_THRESHOLD) {
    return(hidden)
  }
  kept <- which(!hidden)
  for (index in kept[order(n[kept])]) {
    hidden[index] <- TRUE
    total <- total + n[index]
    if (total >= DUA_THRESHOLD) break
  }
  hidden
}

suppress_group <- function(df, labels, by) {
  hidden <- dua_hidden(df$n)
  kept <- df[!hidden, , drop = FALSE]
  out <- as.data.frame(lapply(kept[labels], as.character), stringsAsFactors = FALSE)
  counts <- kept$n
  if (any(hidden)) {
    other <- setNames(as.list(rep(DUA_OTHER, length(labels))), labels)
    if (!is.null(by)) other[[by]] <- as.character(df[[by]][1])
    out <- rbind(out, as.data.frame(other, stringsAsFactors = FALSE, check.names = FALSE))
    counts <- c(counts, sum(df$n[hidden]))
  }
  out$n <- counts
  out
}

dua_suppress <- function(df, by = NULL) {
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
  if (!is.null(by) && !by %in% labels) {
    stop("Sloupec by musí být popisek tabulky.")
  }
  if (is.null(by) || nrow(df) == 0L) {
    out <- suppress_group(df, labels, NULL)
  } else {
    groups <- as.character(df[[by]])
    parts <- lapply(unique(groups), function(group) {
      suppress_group(df[groups == group, , drop = FALSE], labels, by)
    })
    out <- do.call(rbind, parts)
  }
  rownames(out) <- NULL
  out
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
