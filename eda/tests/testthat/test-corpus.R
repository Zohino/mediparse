test_that("katalog tabulek má jedinečná jména a neprázdné popisy", {
  catalog <- mimic_catalog()

  expect_false(anyDuplicated(catalog$tabulka) > 0)
  expect_true(all(nzchar(catalog$popis)))
  expect_true(all(nzchar(catalog$zdroj)))
})

test_that("katalog obsahuje tabulku mimic-iv-bhc jako třetí zdroj", {
  catalog <- mimic_catalog()
  row <- catalog[catalog$tabulka == "mimic-iv-bhc", ]

  expect_equal(nrow(catalog), 13L)
  expect_equal(row$zdroj, MIMIC_BHC)
  expect_equal(length(unique(catalog$zdroj)), 3L)
})

test_that("manifest_text převede velikost na MB a spočte sloupce", {
  shapes <- list(a = list(columns = list("x", "y"), records = 5))
  sizes <- list(a = 2600000)
  out <- manifest_text("a", shapes, sizes)

  expect_equal(out$sloupce, "2")
  expect_equal(out$mb, "3")
  expect_equal(out$records, 5)
})

test_that("manifest_text odděluje tisíce MB nezlomitelnou mezerou", {
  shapes <- list(a = list(columns = list("x"), records = 1))
  sizes <- list(a = 1139e6)

  expect_equal(manifest_text("a", shapes, sizes)$mb, "1\u00a0139")
})

test_that("provenance_table překládá klíče na české popisky a dirty na ano/ne", {
  details <- list(commit = "abc", dirty = "false", python = "3.13", r = "R 4.5")
  out <- provenance_table(details)

  expect_equal(out$Položka, c("Revize kódu (commit)", "Necommitnuté změny", "Python", "R"))
  expect_equal(out$Hodnota, c("abc", "ne", "3.13", "R 4.5"))
})

test_that("provenance_table pojmenuje otisky CSV", {
  out <- provenance_table(list(`csv_sha256 discharge.csv.gz` = "x"))

  expect_equal(out$Položka, "Otisk discharge.csv.gz (SHA-256)")
})

test_that("provenance_table rozdělí 64znakový otisk do skupin po 16 znacích", {
  sha <- paste(rep(c("a", "b", "c", "d"), each = 16), collapse = "")
  out <- provenance_table(list(pixi_lock_sha256 = sha))

  expect_equal(out$Hodnota, paste(strsplit(sha, "(?<=.{16})", perl = TRUE)[[1]], collapse = " "))
})
