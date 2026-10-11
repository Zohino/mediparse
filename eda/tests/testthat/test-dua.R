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

  expect_equal(out$kategorie, c("velká", "ostatní (n < 10)"))
  expect_equal(out$n, c("100", "27"))
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

test_that("dua_suppress vrací číselné n a sloučí malé kategorie do jednoho řádku", {
  df <- data.frame(kategorie = c("velká", "a", "b"), n = c(100L, 6L, 5L))
  out <- dua_suppress(df)

  expect_equal(out$kategorie, c("velká", "ostatní (n < 10)"))
  expect_equal(out$n, c(100, 11))
  expect_true(is.numeric(out$n))
})

test_that("dua_suppress bez malých kategorií řádek nepřidá", {
  df <- data.frame(kategorie = c("a", "b"), n = c(10L, 25L))
  out <- dua_suppress(df)

  expect_equal(out$kategorie, c("a", "b"))
  expect_equal(out$n, c(10, 25))
})

test_that("dua_suppress odmítne další číselný sloupec", {
  df <- data.frame(kategorie = "a", rok = 2001L, n = 5L)

  expect_error(dua_suppress(df), "číselný")
})

test_that("dua_percent vrátí procento s jedním desetinným místem a čárkou", {
  expect_equal(dua_percent(25, 200), "12,5")
  expect_equal(dua_percent(10, 30), "33,3")
  expect_equal(dua_percent(50, 50), "100,0")
})

test_that("dua_percent potlačí čitatel pod prahem", {
  expect_equal(dua_percent(9, 1000), "–")
  expect_equal(dua_percent(0, 1000), "–")
})

test_that("dua_percent pro nulový jmenovatel vrátí pomlčku, ne NaN", {
  expect_equal(dua_percent(0, 0), "–")
  expect_equal(dua_percent(15, 0), "–")
})

test_that("dua_percent zvládne vektor", {
  expect_equal(dua_percent(c(5, 20, 20), c(100, 0, 40)), c("–", "–", "50,0"))
})

test_that("dua_percent přijme skalární jmenovatel pro vektor čitatelů", {
  expect_equal(dua_percent(c(5, 20), 100), c("–", "20,0"))
})

test_that("dua_share přidá podíl z celku před potlačením", {
  df <- data.frame(kategorie = c("velká", "malá", "střední"), n = c(80L, 4L, 16L))
  out <- dua_share(df)

  expect_named(out, c("kategorie", "n", "podil"))
  expect_equal(out$kategorie, c("velká", "ostatní (n < 10)"))
  expect_equal(out$n, c("80", "20"))
  expect_equal(out$podil, c("80,0", "20,0"))
})

test_that("dua_share zveřejní podíl sloučeného řádku, když práh přesáhne", {
  df <- data.frame(kategorie = c("velká", "a", "b"), n = c(90L, 6L, 5L))

  expect_equal(dua_share(df)$podil, c("89,1", "10,9"))
})

test_that("dua_share zvládne prázdnou tabulku", {
  out <- dua_share(data.frame(kategorie = character(), n = integer()))

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kategorie", "n", "podil"))
})

test_that("dua_hidden skryje n pod prahem", {
  expect_equal(dua_hidden(c(100, 200, 50)), c(FALSE, FALSE, FALSE))
  expect_equal(dua_hidden(c(0, 100, 200)), c(TRUE, FALSE, FALSE))
})

test_that("dua_hidden doplní nejmenší zachovanou buňku, když skryté dají 1 až 9", {
  expect_equal(dua_hidden(c(100, 4, 50)), c(FALSE, TRUE, TRUE))
  expect_equal(dua_hidden(c(100, 3, 4)), c(TRUE, TRUE, TRUE))
  expect_equal(dua_hidden(c(5, 5, 100)), c(TRUE, TRUE, FALSE))
})

test_that("dua_hidden skryje všechno, když ani doplnění nedosáhne prahu", {
  expect_equal(dua_hidden(c(4, 3)), c(TRUE, TRUE))
  expect_equal(dua_hidden(integer()), logical())
})

test_that("po skrytí je součet skrytých buněk 0 nebo aspoň 10", {
  set.seed(1)
  for (i in 1:200) {
    n <- sample(c(0:12, 50, 400), sample(1:8, 1), replace = TRUE)
    hidden <- dua_hidden(n)
    expect_true(all(hidden[n < 10]))
    expect_true(sum(n[hidden]) == 0 || sum(n[hidden]) >= 10 || all(hidden))
  }
})

test_that("jediná malá kategorie nedá řádek ostatní s n pod prahem", {
  df <- data.frame(kategorie = c("velká", "střední", "malá"), n = c(100L, 40L, 4L))
  out <- dua_suppress(df)

  expect_equal(out$kategorie, c("velká", "ostatní (n < 10)"))
  expect_equal(out$n, c(100, 44))
})

test_that("dua_suppress s by potlačuje uvnitř každé skupiny zvlášť", {
  df <- data.frame(
    skupina = c("A", "A", "A", "B", "B", "B"),
    kategorie = c("x", "y", "z", "x", "y", "z"),
    n = c(100L, 4L, 50L, 60L, 70L, 80L)
  )
  out <- dua_suppress(df, by = "skupina")

  expect_equal(out$skupina, c("A", "A", "B", "B", "B"))
  expect_equal(out$kategorie, c("x", "ostatní (n < 10)", "x", "y", "z"))
  expect_equal(out$n, c(100, 54, 60, 70, 80))
})

test_that("dua_suppress s by zvládne prázdnou tabulku a neznámý sloupec", {
  empty <- data.frame(skupina = character(), kategorie = character(), n = integer())

  expect_equal(nrow(dua_suppress(empty, by = "skupina")), 0L)
  expect_error(dua_suppress(data.frame(k = "a", n = 50L), by = "nic"), "by")
})

test_that("dua_decimal zaokrouhlí na jedno desetinné místo s čárkou", {
  expect_equal(dua_decimal(12.46), "12,5")
  expect_equal(dua_decimal(100), "100,0")
  expect_equal(dua_decimal(c(1, 2.25)), c("1,0", "2,2"))
})

test_that("dua_percent_text přidá nezlomitelnou mezeru a znak procenta", {
  expect_equal(dua_percent_text(25, 200), "12,5 %")
  expect_equal(dua_percent_text(c(5, 20), 100), c("–", "20,0 %"))
  expect_equal(dua_percent_text(15, 0), "–")
})

test_that("format_number odděluje tisíce nezlomitelnou mezerou a dua_count ho používá", {
  expect_equal(format_number(1234567), paste0("1", NBSP, "234", NBSP, "567"))
  expect_equal(dua_count(1234), format_number(1234))
})

test_that("number_text oddělí tisíce a místo NA vrátí pomlčku", {
  expect_equal(number_text(c(1234567, 5, NA)), c("1 234 567", "5", "–"))
})

test_that("percent_text zveřejní i malý čitatel a pro nulový jmenovatel vrátí pomlčku", {
  expect_equal(percent_text(c(3, 25, 15), c(1000, 200, 0)), c("0,3", "12,5", "–"))
  expect_equal(percent_text(c(0, 20), 100), c("0,0", "20,0"))
})
