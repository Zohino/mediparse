THESIS_WIDTH_CM <- 16.5
THESIS_FONT <- "Libertinus Serif"
THESIS_FONT_SIZE <- 10

LABEL_PATTERN <- "^[a-z0-9-]+$"

LATEX_SPECIALS <- c(
  "\\" = "\\textbackslash{}",
  "&" = "\\&",
  "%" = "\\%",
  "$" = "\\$",
  "#" = "\\#",
  "_" = "\\_",
  "{" = "\\{",
  "}" = "\\}",
  "~" = "\\textasciitilde{}",
  "^" = "\\textasciicircum{}",
  "<" = "\\textless{}",
  ">" = "\\textgreater{}"
)

export_dir <- function(input = knitr::current_input()) {
  file.path("..", "results", "eda", tools::file_path_sans_ext(input))
}

latex_escape <- function(x) {
  vapply(
    as.character(x),
    function(value) {
      chars <- strsplit(value, "", fixed = TRUE)[[1]]
      known <- chars %in% names(LATEX_SPECIALS)
      chars[known] <- LATEX_SPECIALS[chars[known]]
      paste(chars, collapse = "")
    },
    character(1),
    USE.NAMES = FALSE
  )
}

check_label <- function(label) {
  if (!is.character(label) || length(label) != 1L || !grepl(LABEL_PATTERN, label)) {
    stop("Chunk potřebuje label odpovídající ", LABEL_PATTERN, " (název exportovaného souboru).")
  }
}

check_layout <- function(df, sides, widths) {
  if (length(sides) != ncol(df) || !all(sides %in% c("l", "c", "r"))) {
    stop("align musí mít jeden znak l, c nebo r na sloupec (", ncol(df), ").")
  }
  if (length(widths) == 0L) {
    return(invisible(NULL))
  }
  numbers <- suppressWarnings(as.numeric(unlist(widths)))
  if (length(numbers) != ncol(df) || !all(is.finite(numbers) & numbers > 0)) {
    stop("widths musí být kladná čísla, jedno na sloupec (", ncol(df), ").")
  }
}

text_cells <- function(df) {
  as.data.frame(
    lapply(df, function(column) if (is.character(column)) column else as.character(column)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

column_spec <- function(align, widths) {
  if (length(widths) == 0L) {
    return(align)
  }
  mapply(
    function(side, width) {
      alignment <- switch(
        side,
        l = "\\raggedright",
        r = "\\raggedleft",
        c = "\\centering"
      )
      sprintf(
        ">{%s\\arraybackslash}p{\\dimexpr %s\\linewidth-2\\tabcolsep\\relax}",
        alignment, format(width / 100)
      )
    },
    align, as.numeric(unlist(widths)),
    USE.NAMES = FALSE
  )
}

latex_cells <- function(df, code) {
  out <- lapply(names(df), function(name) {
    cells <- latex_escape(df[[name]])
    if (!name %in% code) {
      return(cells)
    }
    ifelse(df[[name]] == DUA_OTHER, cells, paste0("\\texttt{", cells, "}"))
  })
  names(out) <- names(df)
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

markdown_cells <- function(df, code) {
  out <- df
  for (name in intersect(code, names(df))) {
    out[[name]] <- ifelse(df[[name]] == DUA_OTHER, df[[name]], paste0("`", df[[name]], "`"))
  }
  out
}

export_table <- function(
    df, col.names, align, code = character(),
    label = knitr::opts_current$get("label"),
    widths = knitr::opts_current$get("tbl-colwidths"),
    dir = export_dir()) {
  check_label(label)
  sides <- strsplit(align, "", fixed = TRUE)[[1]]
  check_layout(df, sides, widths)
  df <- text_cells(df)
  tex <- knitr::kable(
    latex_cells(df, code),
    format = "latex",
    booktabs = TRUE,
    linesep = "",
    escape = FALSE,
    col.names = latex_escape(col.names),
    align = column_spec(sides, widths),
    row.names = FALSE
  )
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(trimws(as.character(tex)), file.path(dir, paste0(label, ".tex")), useBytes = TRUE)
  knitr::kable(markdown_cells(df, code), col.names = col.names, align = align, row.names = FALSE)
}

theme_thesis <- function() {
  ggplot2::theme_minimal(base_family = THESIS_FONT, base_size = THESIS_FONT_SIZE)
}

embedded_fonts <- function(path) {
  system2("pdffonts", shQuote(path), stdout = TRUE)
}

check_embedded_font <- function(path) {
  expected <- gsub(" ", "", THESIS_FONT, fixed = TRUE)
  if (any(grepl(expected, embedded_fonts(path), fixed = TRUE))) {
    return(invisible(path))
  }
  unlink(path)
  stop(
    "PDF grafu nevložilo písmo ", THESIS_FONT, ": cairo dosadilo jiné. ",
    "Nainstaluj písmo systémovým balíkem (Fedora: libertinus-fonts, ",
    "Debian: fonts-libertinus), jinde bash scripts/install-libertinus.sh."
  )
}

export_figure <- function(
    plot, height_cm,
    label = knitr::opts_current$get("label"),
    dir = export_dir()) {
  check_label(label)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(dir, paste0(label, ".pdf"))
  ggplot2::ggsave(
    path, plot,
    device = grDevices::cairo_pdf,
    width = THESIS_WIDTH_CM, height = height_cm, units = "cm"
  )
  check_embedded_font(path)
  plot
}

percent_axis <- function(x) {
  paste0(format(100 * x, decimal.mark = ",", trim = TRUE), NBSP, "%")
}
