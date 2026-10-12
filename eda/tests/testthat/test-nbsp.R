FILTER <- normalizePath(test_path("..", "..", "_filters", "nbsp.lua"), mustWork = FALSE)

typst_of <- function(markdown) {
  skip_if(Sys.which("quarto") == "", "quarto není v PATH")
  withr::local_dir(withr::local_tempdir())
  writeLines(markdown, "in.md")
  system2(
    "quarto",
    c("pandoc", "-f", "markdown", "-t", "typst", "--lua-filter", shQuote(FILTER), "in.md"),
    stdout = TRUE, stderr = FALSE
  )
}

test_that("filtr spojí jednopísmenné předložky a spojky s dalším slovem", {
  out <- paste(typst_of("Byl v nemocnici a doma. K tomu (v domě) i o tom."), collapse = " ")

  expect_match(out, "v~nemocnici", fixed = TRUE)
  expect_match(out, "a~doma", fixed = TRUE)
  expect_match(out, "K~tomu", fixed = TRUE)
  expect_match(out, "(v~domě)", fixed = TRUE)
  expect_match(out, "i~o~tom", fixed = TRUE)
})

test_that("filtr spojí předložku po levé horní uvozovce", {
  out <- paste(typst_of("Byl „v nemocnici“ doma."), collapse = " ")

  expect_match(out, "„v~nemocnici", fixed = TRUE)
})

test_that("filtr spojí číslo s dalším slovem a znakem procenta", {
  out <- paste(typst_of("Je 5 zmínek, 12 345 zpráv a 5 % a 1,5 zmínky."), collapse = " ")

  expect_match(out, "5~zmínek", fixed = TRUE)
  expect_match(out, "12~345~zpráv", fixed = TRUE)
  expect_match(out, "5~%", fixed = TRUE)
  expect_match(out, "1,5~zmínky", fixed = TRUE)
})

test_that("filtr spojí i přes zalomení řádku", {
  out <- paste(typst_of("Byl v\nnemocnici a\ndoma 5\nzmínek."), collapse = " ")

  expect_match(out, "v~nemocnici", fixed = TRUE)
  expect_match(out, "a~doma", fixed = TRUE)
  expect_match(out, "5~zmínek", fixed = TRUE)
})

test_that("filtr nechá víceznakové slovo a kód beze změny", {
  out <- paste(typst_of("Pak dům je `x v y` hotový."), collapse = " ")

  expect_match(out, "Pak dům je", fixed = TRUE)
  expect_match(out, "`x v y`", fixed = TRUE)
  expect_false(grepl("Pak~", out, fixed = TRUE))
})
