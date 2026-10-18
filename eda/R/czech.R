plural_form <- function(n, forms) {
  decimal_form <- if (length(forms) >= 4L) forms[[4]] else forms[[2]]
  whole <- !is.na(n) & n == round(n)
  out <- rep(forms[[3]], length(n))
  out[whole & n == 1] <- forms[[1]]
  out[whole & n >= 2 & n <= 4] <- forms[[2]]
  out[!whole] <- decimal_form
  out[is.na(n)] <- NA_character_
  out
}

number_phrase <- function(n) {
  vapply(n, function(value) {
    if (is.na(value)) {
      return("–")
    }
    if (value == round(value)) {
      return(format_number(value))
    }
    format(value, big.mark = NBSP, decimal.mark = ",", scientific = FALSE, trim = TRUE)
  }, character(1))
}

count_phrase <- function(n, forms, zero = NULL) {
  out <- paste0(number_phrase(n), NBSP, plural_form(n, forms))
  if (!is.null(zero)) {
    out[!is.na(n) & n == 0] <- zero
  }
  out
}

join_labels <- function(labels) {
  if (length(labels) == 0L) {
    return("žádný")
  }
  if (length(labels) == 1L) {
    return(labels)
  }
  paste0(paste(utils::head(labels, -1L), collapse = ", "), " a ", utils::tail(labels, 1L))
}
