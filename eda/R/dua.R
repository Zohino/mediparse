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

dua_table <- function(df) {
  out <- dua_suppress(df)
  out$n <- dua_count(out$n)
  out
}

dua_decimal <- function(x) {
  format(round(x, 1), nsmall = 1, decimal.mark = ",", trim = TRUE)
}

dua_percent <- function(n, total) {
  total <- rep_len(total, length(n))
  hidden <- n < DUA_THRESHOLD | total == 0
  out <- rep("\u2013", length(n))
  out[!hidden] <- dua_decimal(100 * n[!hidden] / total[!hidden])
  out
}

dua_percent_text <- function(n, total) {
  share <- dua_percent(n, total)
  ifelse(share == "\u2013", share, paste0(share, NBSP, "%"))
}

dua_share <- function(df) {
  out <- dua_suppress(df)
  out$podil <- dua_percent(out$n, sum(df$n))
  out$n <- dua_count(out$n)
  out
}

dua_quantile_ok <- function(n, p) {
  round(n * pmin(p, 1 - p), 9) >= DUA_THRESHOLD
}
