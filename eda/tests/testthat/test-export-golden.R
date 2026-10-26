golden_dir <- "../../../tests/fixtures/thesis_table"

golden_bytes <- function(path) {
  readBin(path, "raw", n = file.info(path)$size)
}

golden_frame <- function(case) {
  columns <- lapply(seq_along(case$header), function(i) {
    vapply(case$rows, function(row) row[[i]], character(1))
  })
  names(columns) <- unlist(case$header)
  as.data.frame(columns, stringsAsFactors = FALSE, check.names = FALSE)
}

test_that("export tabulek odpovídá sdíleným golden souborům bajt po bajtu", {
  cases <- jsonlite::read_json(file.path(golden_dir, "cases.json"))
  dir <- withr::local_tempdir()
  for (case in cases) {
    label <- paste0("tbl-", case$name)
    md <- export_table(
      golden_frame(case), col.names = unlist(case$header), align = case$align,
      code = unlist(case$code), widths = unlist(case$widths), label = label, dir = dir
    )
    expect_identical(
      golden_bytes(file.path(dir, paste0(label, ".tex"))),
      golden_bytes(file.path(golden_dir, paste0(case$name, ".tex"))),
      info = case$name
    )
    expect_identical(
      charToRaw(enc2utf8(paste(as.character(md), collapse = "\n"))),
      golden_bytes(file.path(golden_dir, paste0(case$name, ".md"))),
      info = case$name
    )
  }
})

test_that("provenance odpovídá sdílenému golden souboru", {
  details <- jsonlite::read_json(file.path(golden_dir, "provenance.json"))
  dir <- withr::local_tempdir()
  export_table(
    provenance_table(details), col.names = c("Položka", "Hodnota"), align = "ll",
    label = "tbl-provenance", widths = c(28, 72), dir = dir
  )
  expect_identical(
    golden_bytes(file.path(dir, "tbl-provenance.tex")),
    golden_bytes(file.path(golden_dir, "provenance.tex"))
  )
})
