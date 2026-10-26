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
