FORMS <- c("zmínka", "zmínky", "zmínek")

test_that("plural_form vybere tvar podle českého skloňování po číslovce", {
  expect_equal(plural_form(1, FORMS), "zmínka")
  expect_equal(plural_form(c(2, 3, 4), FORMS), rep("zmínky", 3L))
  expect_equal(
    plural_form(c(0, 5, 11, 12, 21, 22, 100, 1001), FORMS),
    rep("zmínek", 8L)
  )
})

test_that("plural_form dá desetinnému číslu tvar few, nebo čtvrtý tvar", {
  expect_equal(plural_form(1.5, FORMS), "zmínky")
  expect_equal(plural_form(1.5, c("den", "dny", "dnů", "dne")), "dne")
  expect_equal(plural_form(c(1, 2, 5), c("den", "dny", "dnů", "dne")), c("den", "dny", "dnů"))
})

test_that("plural_form je vektorový", {
  expect_equal(plural_form(c(1, 2, 5, 1.5), FORMS), c("zmínka", "zmínky", "zmínek", "zmínky"))
})

test_that("count_phrase spojí číslo s tvarem nedělitelnou mezerou", {
  expect_equal(count_phrase(5, FORMS), "5 zmínek")
  expect_equal(count_phrase(1, FORMS), "1 zmínka")
  expect_equal(count_phrase(12345, FORMS), "12 345 zmínek")
  expect_equal(count_phrase(1.5, FORMS), "1,5 zmínky")
})

test_that("count_phrase formátuje po prvcích a pro nulu vrátí daný text", {
  expect_equal(count_phrase(c(1, 1.5), FORMS), c("1 zmínka", "1,5 zmínky"))
  expect_equal(count_phrase(c(0, 3), FORMS, zero = "žádná zmínka"), c("žádná zmínka", "3 zmínky"))
  expect_equal(count_phrase(0, FORMS), "0 zmínek")
})

test_that("join_labels spojí popisky česky a prázdný seznam pojmenuje", {
  expect_equal(join_labels(character()), "žádný")
  expect_equal(join_labels("a"), "a")
  expect_equal(join_labels(c("a", "b", "c")), "a, b a c")
})
