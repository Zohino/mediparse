test_that("dua_decimal zaokrouhlí na jedno desetinné místo s čárkou", {
  expect_equal(dua_decimal(12.46), "12,5")
  expect_equal(dua_decimal(100), "100,0")
  expect_equal(dua_decimal(c(1, 2.25)), c("1,0", "2,2"))
})

test_that("format_number odděluje tisíce nezlomitelnou mezerou", {
  expect_equal(format_number(1234567), paste0("1", NBSP, "234", NBSP, "567"))
})

test_that("number_text oddělí tisíce a místo NA vrátí pomlčku", {
  expect_equal(number_text(c(1234567, 5, NA)), c("1\u00a0234\u00a0567", "5", "–"))
})

test_that("percent_text zveřejní i malý čitatel a pro nulový jmenovatel vrátí pomlčku", {
  expect_equal(percent_text(c(3, 25, 15), c(1000, 200, 0)), c("0,3", "12,5", "–"))
  expect_equal(percent_text(c(0, 20), 100), c("0,0", "20,0"))
})

test_that("percent_text nenulový podíl pod 0,05 % neukáže jako nulu", {
  expect_equal(percent_text(c(22, 0, 2), c(331604, 331604, 1000)), c("<\u00a00,1", "0,0", "0,2"))
  expect_equal(percent_label(22, 331604), "<\u00a00,1\u00a0%")
})

test_that("percent_label přidá znak procenta, pomlčku nechá", {
  expect_equal(percent_label(c(3, 25, 15), c(1000, 200, 0)), c("0,3\u00a0%", "12,5\u00a0%", "–"))
})

test_that("share_table zveřejní všechny řádky s počtem a podílem z celku", {
  out <- share_table(data.frame(kos = c("1", "2", "3+"), n = c(1990, 6, 4)))

  expect_named(out, c("kos", "n", "podil"))
  expect_equal(out$kos, c("1", "2", "3+"))
  expect_equal(out$n, c("1\u00a0990", "6", "4"))
  expect_equal(out$podil, c("99,5", "0,3", "0,2"))
})

test_that("share_table zvládne prázdnou tabulku", {
  out <- share_table(data.frame(kos = character(), n = numeric()))

  expect_equal(nrow(out), 0L)
  expect_named(out, c("kos", "n", "podil"))
})

test_that("percent_text vrátí pomlčku pro neznámý čitatel i jmenovatel", {
  expect_equal(percent_text(c(NA, 1), c(10, NA)), c("–", "–"))
  expect_equal(percent_text(1, 4), "25,0")
})
