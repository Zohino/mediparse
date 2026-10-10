test_that("dua_count potlačí hodnoty pod prahem", {
  expect_equal(dua_count(9L), "<\u00a010")
  expect_equal(dua_count(10L), "10")
  expect_equal(dua_count(0L), "<\u00a010")
})

test_that("dua_count odděluje tisíce nezlomitelnou mezerou a zvládne vektor", {
  expect_equal(dua_count(c(1234567L, 5L)), c("1\u00a0234\u00a0567", "<\u00a010"))
})

test_that("dua_table sloučí řádky pod prahem a jejich popisky zmizí", {
  df <- data.frame(
    kategorie = c("velká", "malá A", "malá B", "střední"),
    n = c(100L, 4L, 3L, 20L)
  )
  out <- dua_table(df)

  expect_equal(out$kategorie, c("velká", "střední", "ostatní (n < 10)"))
  expect_equal(out$n, c("100", "20", "<\u00a010"))
  expect_false(any(grepl("malá", unlist(out))))
})

test_that("dua_table ukáže součet sloučených řádků, když práh přesáhne", {
  df <- data.frame(kategorie = c("velká", "a", "b", "c"), n = c(100L, 6L, 5L, 4L))
  out <- dua_table(df)

  expect_equal(out$n, c("100", "15"))
  expect_equal(out$kategorie, c("velká", "ostatní (n < 10)"))
})

test_that("dua_table bez sloupce n skončí chybou", {
  expect_error(dua_table(data.frame(text = c("a", "b"))), "n")
})

test_that("dua_table odmítne neceločíselné n", {
  expect_error(dua_table(data.frame(kategorie = "a", n = 1.5)), "celočíseln")
})

test_that("dua_table přijme celočíselné n typu double", {
  out <- dua_table(data.frame(kategorie = "a", n = 42))

  expect_equal(out$n, "42")
})

test_that("dua_table nemění tabulku, kde je vše na prahu nebo nad ním", {
  df <- data.frame(kategorie = c("a", "b"), n = c(10L, 25L))
  out <- dua_table(df)

  expect_equal(out$kategorie, c("a", "b"))
  expect_equal(out$n, c("10", "25"))
})

test_that("dua_table zvládne prázdnou tabulku", {
  out <- dua_table(data.frame(kategorie = character(), n = integer()))

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kategorie", "n"))
})

test_that("dua_table odmítne další číselný sloupec vedle n", {
  df <- data.frame(kategorie = c("a", "b"), rok = c(2001L, 2002L), n = c(5L, 50L))

  expect_error(dua_table(df), "číselný")
})

test_that("dua_table přijme faktor jako popisek", {
  out <- dua_table(data.frame(kategorie = factor(c("a", "b")), n = c(50L, 60L)))

  expect_equal(out$kategorie, c("a", "b"))
})
