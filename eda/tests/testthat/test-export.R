sample_table <- function() {
  data.frame(
    tabulka = c("diagnoses_icd", DUA_OTHER),
    popis = c("a & b 100%", "x"),
    n = c(paste0("1", NBSP, "234"), "5"),
    stringsAsFactors = FALSE
  )
}

read_tex <- function(dir, label) {
  paste(readLines(file.path(dir, paste0(label, ".tex")), encoding = "UTF-8"), collapse = "\n")
}

export_sample <- function(dir, label = "tbl-a", ...) {
  export_table(
    sample_table(),
    col.names = c("Tabulka", "Popis", "Záznamů"),
    align = "llr",
    label = label,
    widths = NULL,
    dir = dir,
    ...
  )
}

test_that("export_table zapíše tabular s booktabs bez floatu", {
  dir <- withr::local_tempdir()
  export_sample(dir)
  tex <- read_tex(dir, "tbl-a")

  expect_match(tex, "^\\\\begin\\{tabular\\}")
  expect_match(tex, "\\\\toprule", fixed = FALSE)
  expect_match(tex, "\\\\midrule")
  expect_match(tex, "\\\\bottomrule")
  expect_false(grepl("\\begin{table}", tex, fixed = TRUE))
  expect_false(grepl("\\caption", tex, fixed = TRUE))
  expect_false(grepl("\\label", tex, fixed = TRUE))
})

test_that("latex_escape escapuje speciální znaky a NBSP nechá", {
  expect_equal(
    latex_escape("_ & % # $ { }"),
    "\\_ \\& \\% \\# \\$ \\{ \\}"
  )
  expect_equal(latex_escape(paste0("1", NBSP, "2")), paste0("1", NBSP, "2"))
})

test_that("export_table escapuje buňky i hlavičku a NBSP projde", {
  dir <- withr::local_tempdir()
  export_table(
    data.frame(a = "x_y & 5%", n = paste0("1", NBSP, "234")),
    col.names = c("Sloupec_1 #", "Počet $"),
    align = "lr",
    label = "tbl-b",
    widths = NULL,
    dir = dir
  )
  tex <- read_tex(dir, "tbl-b")

  expect_match(tex, "x\\_y \\& 5\\%", fixed = TRUE)
  expect_match(tex, "Sloupec\\_1 \\#", fixed = TRUE)
  expect_match(tex, "Počet \\$", fixed = TRUE)
  expect_match(tex, paste0("1", NBSP, "234"), fixed = TRUE)
})

test_that("code obalí hodnoty v LaTeXu texttt a v kable backticky, DUA_OTHER nechá", {
  dir <- withr::local_tempdir()
  md <- export_sample(dir, code = "tabulka")
  tex <- read_tex(dir, "tbl-a")
  md_text <- paste(as.character(md), collapse = "\n")

  expect_match(tex, "\\texttt{diagnoses\\_icd}", fixed = TRUE)
  expect_false(grepl("texttt{ostatní", tex, fixed = TRUE))
  expect_match(md_text, "`diagnoses_icd`", fixed = TRUE)
  expect_false(grepl(paste0("`", DUA_OTHER), md_text, fixed = TRUE))
})

test_that("widths vytvoří p sloupce podle zarovnání, bez widths prosté l/r", {
  dir <- withr::local_tempdir()
  export_table(
    sample_table(),
    col.names = c("Tabulka", "Popis", "Záznamů"),
    align = "llr",
    label = "tbl-w",
    widths = c(22, 60, 18),
    dir = dir
  )
  tex <- read_tex(dir, "tbl-w")

  expect_match(tex, "p{\\dimexpr 0.22\\linewidth-2\\tabcolsep\\relax}", fixed = TRUE)
  expect_match(tex, "p{\\dimexpr 0.6\\linewidth-2\\tabcolsep\\relax}", fixed = TRUE)
  expect_match(tex, ">{\\raggedright\\arraybackslash}p{\\dimexpr 0.22", fixed = TRUE)
  expect_match(tex, ">{\\raggedleft\\arraybackslash}p{\\dimexpr 0.18", fixed = TRUE)

  export_sample(dir, label = "tbl-plain")
  expect_match(read_tex(dir, "tbl-plain"), "\\begin{tabular}{llr}", fixed = TRUE)
})

test_that("návratová kable má stejné řádky jako .tex", {
  dir <- withr::local_tempdir()
  md <- export_sample(dir)
  tex <- read_tex(dir, "tbl-a")

  expect_s3_class(md, "knitr_kable")
  expect_equal(length(md), 2 + nrow(sample_table()))
  expect_equal(lengths(regmatches(tex, gregexpr("\\\\\\\\", tex))), 1 + nrow(sample_table()))
})

test_that("chybějící nebo neplatný label je chyba a dir se založí", {
  dir <- file.path(withr::local_tempdir(), "a", "b")

  expect_error(export_sample(dir, label = NULL), "label")
  expect_error(export_sample(dir, label = "Tbl_A"), "label")
  expect_false(dir.exists(dir))
  export_sample(dir)
  expect_true(file.exists(file.path(dir, "tbl-a.tex")))
})

test_that("export_dir odvodí adresář z názvu vstupu", {
  expect_equal(export_dir("1-dataset.rmarkdown"), "../results/eda/1-dataset")
})


test_that("export_table nevkládá \\addlinespace po pěti řádcích", {
  dir <- withr::local_tempdir()
  df <- data.frame(a = letters[1:7], stringsAsFactors = FALSE)
  export_table(df, col.names = "A", align = "l", label = "tbl-l", widths = NULL, dir = dir)

  expect_false(grepl("addlinespace", read_tex(dir, "tbl-l"), fixed = TRUE))
})

test_that("export_table odmítne nesouladné zarovnání a šířky", {
  dir <- withr::local_tempdir()
  df <- data.frame(a = "x", b = "y")
  run <- function(align = "ll", widths = NULL) {
    export_table(df, col.names = c("A", "B"), align = align, label = "tbl-e", widths = widths, dir = dir)
  }

  expect_error(run(align = "l"), "align")
  expect_error(run(align = "llr"), "align")
  expect_error(run(align = "lx"), "align")
  expect_error(run(widths = 100), "widths")
  expect_error(run(widths = c(50, 50, 10)), "widths")
  expect_error(run(widths = c(50, NA)), "widths")
  expect_error(run(widths = c(50, 0)), "widths")
  expect_error(run(widths = c(50, -5)), "widths")
  expect_error(run(widths = c("50", "auto")), "widths")
  expect_error(run(widths = c(50, Inf)), "widths")
})

test_that("číselné sloupce jsou v .tex i v návratové kable stejné", {
  dir <- withr::local_tempdir()
  md <- export_table(
    data.frame(x = c(0.5, 0.25, 1234567)),
    col.names = "X", align = "r", label = "tbl-n", widths = NULL, dir = dir
  )
  tex <- read_tex(dir, "tbl-n")
  md_text <- paste(as.character(md), collapse = "\n")

  for (cell in as.character(c(0.5, 0.25, 1234567))) {
    expect_match(tex, cell, fixed = TRUE)
    expect_match(md_text, cell, fixed = TRUE)
  }
})

sample_plot <- function(theme = theme_thesis()) {
  ggplot2::ggplot(
    data.frame(kategorie = c("Přijetí", "Žluťoučký"), n = c(3, 5)),
    ggplot2::aes(kategorie, n)
  ) +
    ggplot2::geom_col() +
    theme
}

test_that("theme_thesis používá písmo a velikost práce", {
  theme <- theme_thesis()

  expect_equal(theme$text$family, THESIS_FONT)
  expect_equal(theme$text$size, THESIS_FONT_SIZE)
})

test_that("export_figure zapíše PDF o šířce sazby s vloženým písmem a vrátí graf", {
  dir <- withr::local_tempdir()
  plot <- sample_plot()
  returned <- export_figure(plot, height_cm = 8, label = "fig-a", dir = dir)
  path <- file.path(dir, "fig-a.pdf")
  info <- system2("pdfinfo", path, stdout = TRUE)
  box <- grep("Page size", info, value = TRUE)
  width <- as.numeric(strsplit(trimws(sub("Page size:", "", box)), " +")[[1]][1])
  fonts <- system2("pdffonts", path, stdout = TRUE)

  expect_identical(returned, plot)
  expect_equal(width, THESIS_WIDTH_CM / 2.54 * 72, tolerance = 1 / 467.7)
  expect_true(any(grepl("LibertinusSerif", fonts)))
})

test_that("export_figure bez písma práce skončí chybou s názvem balíku", {
  dir <- withr::local_tempdir()

  expect_error(
    export_figure(sample_plot(ggplot2::theme_minimal(base_family = "Neexistující písmo")), height_cm = 8, label = "fig-b", dir = dir),
    "install-libertinus.sh"
  )
})

test_that("export_figure odmítne neplatný label", {
  expect_error(
    export_figure(sample_plot(), height_cm = 8, label = "Fig_A", dir = withr::local_tempdir()),
    "label"
  )
})

test_that("latex_escape escapuje < a > a nechá * a [ ], které v textu nic neznamenají", {
  expect_equal(latex_escape("<…>"), "\\textless{}…\\textgreater{}")
  expect_equal(latex_escape("[**…**]"), "[**…**]")
  expect_equal(latex_escape("___"), "\\_\\_\\_")
})

test_that("export_table s code obalí popisky vzorů a escapuje je i v hlavičce", {
  dir <- withr::local_tempdir()
  md <- export_table(
    data.frame(vzor = c("___", "[**…**]", "<…>", "XXX"), n = c("1", "2", "3", "4")),
    col.names = c("Vzor <a>", "Počet"),
    align = "lr",
    code = "vzor",
    label = "tbl-vzory",
    widths = NULL,
    dir = dir
  )
  tex <- read_tex(dir, "tbl-vzory")

  expect_match(tex, "\\texttt{\\_\\_\\_}", fixed = TRUE)
  expect_match(tex, "\\texttt{[**…**]}", fixed = TRUE)
  expect_match(tex, "\\texttt{\\textless{}…\\textgreater{}}", fixed = TRUE)
  expect_match(tex, "Vzor \\textless{}a\\textgreater{}", fixed = TRUE)
  expect_match(paste(as.character(md), collapse = "\n"), "`___`", fixed = TRUE)
})

test_that("percent_axis formátuje podíl s desetinnou čárkou a pevnou mezerou před procentem", {
  expect_equal(percent_axis(c(0, 0.5, 1)), paste0(c("0", "50", "100"), NBSP, "%"))
  expect_equal(percent_axis(0.125), paste0("12,5", NBSP, "%"))
})

lightness <- function(colour) {
  grDevices::convertColor(t(grDevices::col2rgb(colour) / 255), "sRGB", "Lab")[, 1]
}

greyscale <- function(colour) {
  drop(c(0.299, 0.587, 0.114) %*% (grDevices::col2rgb(colour) / 255))
}

layer_values <- function(plot, aesthetic) {
  unique(ggplot2::ggplot_build(plot)$data[[1]][[aesthetic]])
}

test_that("kategoriální paleta je z Okabe–Ito a v šedi se barvy nesplynou", {
  expect_gte(length(PALETTE_CATEGORICAL), max(length(ICD_ERAS) + 1L, length(GENDER_LABELS)))
  expect_true(all(PALETTE_CATEGORICAL %in% grDevices::palette.colors(palette = "Okabe-Ito")))
  gaps <- stats::dist(greyscale(c(PALETTE_CATEGORICAL, "white")))
  expect_gte(min(gaps), 0.12)
})

test_that("theme_thesis barví sloupce, výplně a čáry kategoriální paletou", {
  df <- data.frame(x = c("a", "b", "c", "d"), y = 1:4)

  single <- ggplot2::ggplot(df, ggplot2::aes(x, y)) + ggplot2::geom_col() + theme_thesis()
  filled <- ggplot2::ggplot(df, ggplot2::aes(x, y, fill = x)) + ggplot2::geom_col() + theme_thesis()
  lined <- ggplot2::ggplot(df, ggplot2::aes(x, y, colour = x, group = x)) + ggplot2::geom_point() + theme_thesis()

  expect_equal(layer_values(single, "fill"), PALETTE_CATEGORICAL[[1]])
  expect_equal(layer_values(filled, "fill"), PALETTE_CATEGORICAL[1:4])
  expect_equal(layer_values(lined, "colour"), PALETTE_CATEGORICAL[1:4])
})

test_that("scale_fill_sequential tmavne s hodnotou, nese černý text a NA nechá bílé", {
  df <- data.frame(x = c("a", "b", "c"), y = "r", podil = c(0, 1, NA))
  plot <- ggplot2::ggplot(df, ggplot2::aes(x, y, fill = podil)) +
    ggplot2::geom_tile() +
    scale_fill_sequential(limits = c(0, 1)) +
    theme_thesis()
  fills <- ggplot2::ggplot_build(plot)$data[[1]]$fill

  expect_gt(lightness(fills[[1]]), lightness(fills[[2]]))
  expect_gte(lightness(fills[[2]]), 49)
  expect_equal(fills[[3]], "white")
})
