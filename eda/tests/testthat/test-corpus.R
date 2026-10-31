test_that("katalog odpovídá veřejné konfiguraci tabulek", {
  tables <- jsonlite::read_json("../../../config/mimic_tables.json")$mimic_tables
  files <- vapply(tables, function(table) basename(table$url), character(1))
  expect_setequal(mimic_catalog()$tabulka, sub("\\.csv(\\.gz)?$", "", files))
})

test_that("katalog tabulek má jedinečná jména a neprázdné popisy", {
  catalog <- mimic_catalog()

  expect_false(anyDuplicated(catalog$tabulka) > 0)
  expect_true(all(nzchar(catalog$popis)))
  expect_true(all(nzchar(catalog$zdroj)))
})

test_that("katalog obsahuje tabulku mimic-iv-bhc jako třetí zdroj", {
  catalog <- mimic_catalog()
  row <- catalog[catalog$tabulka == "mimic-iv-bhc", ]

  expect_equal(row$zdroj, "MIMIC-IV-Ext-BHC")
  expect_equal(length(unique(catalog$zdroj)), 3L)
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


synthetic_manifest <- function(env = parent.frame()) {
  path <- withr::local_tempfile(.local_envir = env)
  tables <- list(
    list(file = "discharge.csv.gz", bytes = 1139203149, records = 331793,
         columns = as.list(letters[1:8])),
    list(file = "radiology.csv.gz", bytes = 2600000, records = 100,
         columns = list("text")),
    list(file = "mimic-iv-bhc.csv", bytes = 2726465013, records = 331793,
         columns = list("note_id", "target"))
  )
  jsonlite::write_json(list(tables = tables), path, auto_unbox = TRUE, digits = NA)
  manifest_tables(path)
}

manifest_counts <- function(manifest) {
  data.frame(tabulka = manifest$tabulka, n = manifest$records)
}

test_that("JSON manifest zachová velké velikosti a přehled je ručně zaokrouhlí na MB", {
  manifest <- synthetic_manifest()
  out <- corpus_overview(manifest_counts(manifest), manifest)

  expect_type(manifest$bytes, "double")
  expect_type(manifest$records, "double")
  expect_type(manifest$sloupce, "integer")
  expect_equal(out$table$tabulka, c("discharge", "mimic-iv-bhc", "radiology"))
  expect_equal(out$table$mb, c("1\u00a0139", "2\u00a0726", "3"))
  expect_equal(out$table$sloupce[1], "8")
  expect_equal(out$table$n[1], "331\u00a0793")
  expect_equal(out$tables, "3")
  expect_equal(out$discharge, "331\u00a0793")
  expect_equal(out$radiology, "100")
})

test_that("přehled odmítne tabulky mimo katalog s uspořádanými jmény", {
  manifest <- synthetic_manifest()
  extra <- manifest[rep(1, 2), ]
  extra$tabulka <- c("foo", "bar")
  manifest <- rbind(manifest, extra)
  expect_error(corpus_overview(manifest_counts(manifest), manifest),
               "Tabulky mimo katalog: bar, foo.", fixed = TRUE)
  manifest <- manifest[manifest$tabulka != "bar", ]
  expect_error(corpus_overview(manifest_counts(manifest), manifest),
               "Tabulky mimo katalog: foo.", fixed = TRUE)
})

test_that("přehled odmítne i tabulku mimo katalog pouze v manifestu", {
  manifest <- synthetic_manifest()
  counts <- manifest_counts(manifest)
  extra <- manifest[1, ]
  extra$tabulka <- "foo"
  expect_error(corpus_overview(counts, rbind(manifest, extra)),
               "Tabulky mimo katalog: foo.", fixed = TRUE)
})

test_that("přehled odmítne rozdílné množiny tabulek a chybějící povinné tabulky", {
  manifest <- synthetic_manifest()
  counts <- manifest_counts(manifest)
  expect_error(corpus_overview(counts[counts$tabulka != "radiology", ], manifest),
               "Tabulky v manifestu validace, ale ne v databázi: radiology.", fixed = TRUE)
  expect_error(corpus_overview(counts, manifest[manifest$tabulka != "radiology", ]),
               "Tabulky v databázi, ale ne v manifestu validace: radiology.", fixed = TRUE)
  manifest <- manifest[manifest$tabulka != "radiology", ]
  expect_error(corpus_overview(manifest_counts(manifest), manifest),
               "Povinné tabulky chybí v databázi: radiology.", fixed = TRUE)
})

test_that("shoda počtů potvrzuje úplnost a shodné maximum nemá největší tabulku", {
  manifest <- synthetic_manifest()
  counts <- manifest_counts(manifest)
  tied <- c("discharge", "mimic-iv-bhc")
  counts$n[counts$tabulka %in% tied] <- 200
  manifest$records[manifest$tabulka %in% tied] <- 200
  out <- corpus_overview(counts, manifest)
  expect_match(out$records, "odpovídají počtům v manifestu validace\\.$")
  expect_match(out$summary, "^Počty záznamů u všech tabulek odpovídají")
  expect_false(startsWith(out$records, "Nejvíce"))
})

test_that("rozdíl počtů odmítá úplnost a jedinečné maximum pojmenuje tabulku", {
  manifest <- synthetic_manifest()
  counts <- manifest_counts(manifest)
  counts$n[counts$tabulka == "discharge"] <- 331794
  counts$n[counts$tabulka == "radiology"] <- 99
  out <- corpus_overview(counts, manifest)
  expect_equal(out$records, paste0(
    "Nejvíce záznamů má tabulka discharge. ",
    "Počty záznamů v databázi neodpovídají počtům v manifestu validace."
  ))
  expect_equal(out$summary, paste0(
    "Počty záznamů se liší od manifestu validace u tabulek: discharge, radiology. ",
    "Data se proto nesmějí považovat za úplná a před dalším použitím je nutné rozdíl vysvětlit."
  ))
})

test_that("manifest hlásí chybějící pole i soubor", {
  for (field in c("file", "bytes", "records", "columns")) {
    path <- tempfile()
    entry <- list(file = "discharge.csv.gz", bytes = 1, records = 1, columns = list("text"))
    entry[[field]] <- NULL
    jsonlite::write_json(list(tables = list(entry)), path, auto_unbox = TRUE, digits = NA)
    file <- if (field == "file") path else "discharge.csv.gz"
    expect_error(manifest_tables(path), paste0("Soubor ", file), fixed = TRUE)
    expect_error(manifest_tables(path), paste0("chybí pole ", field), fixed = TRUE)
    unlink(path)
  }
})


manifest_entry <- function(...) {
  entry <- list(file = "discharge.csv.gz", bytes = 1, records = 1, columns = list("text"))
  changes <- list(...)
  entry[names(changes)] <- changes
  entry
}

read_test_manifest <- function(value) {
  path <- withr::local_tempfile()
  jsonlite::write_json(value, path, auto_unbox = TRUE, digits = NA)
  manifest_tables(path)
}

expect_invalid_manifest_field <- function(field, value) {
  entry <- manifest_entry()
  entry[field] <- list(value)
  expect_error(read_test_manifest(list(tables = list(entry))),
               paste0("Soubor discharge.csv.gz: neplatné pole ", field), fixed = TRUE)
}

test_that("manifest přijme nulové velikosti i počty", {
  out <- read_test_manifest(list(tables = list(manifest_entry(bytes = 0, records = 0))))
  expect_equal(out[c("bytes", "records")], data.frame(bytes = 0, records = 0))
})

test_that("manifest odmítne chybějící tables, neplatné hodnoty a duplicitní tabulky", {
  expect_error(read_test_manifest(list()), "Soubor .*: chybí pole tables\\.")
  for (field in c("bytes", "records")) {
    expect_invalid_manifest_field(field, -1)
    expect_invalid_manifest_field(field, "1")
    expect_invalid_manifest_field(field, TRUE)
    expect_invalid_manifest_field(field, list(1, 2))
  }
  expect_invalid_manifest_field("columns", "text")
  expect_invalid_manifest_field("columns", list(1))
  expect_invalid_manifest_field("columns", list("text", FALSE))
  expect_error(read_test_manifest(list(tables = list(manifest_entry(), manifest_entry(file = "discharge.csv")))), "Soubor discharge.csv: duplicitní jméno tabulky discharge.", fixed = TRUE)
})


test_that("pořadí jmen v chybě nezávisí na kolaci locale", {
  withr::local_collate("C.UTF-8")
  manifest <- synthetic_manifest()
  extra <- manifest[rep(1, 2), ]
  extra$tabulka <- c("a", "Z")
  manifest <- rbind(manifest, extra)
  expect_error(corpus_overview(manifest_counts(manifest), manifest),
               "Tabulky mimo katalog: Z, a.", fixed = TRUE)
})
