MIMIC_IV <- "MIMIC-IV (hosp)"
MIMIC_NOTE <- "MIMIC-IV-Note (note)"
MIMIC_BHC <- "MIMIC-IV-Ext-BHC"

mimic_catalog <- function() {
  data.frame(
    tabulka = c(
      "discharge", "radiology", "diagnoses_icd", "d_icd_diagnoses",
      "procedures_icd", "d_icd_procedures", "patients", "admissions",
      "services", "prescriptions", "pharmacy", "omr", "mimic-iv-bhc"
    ),
    zdroj = c(
      MIMIC_NOTE, MIMIC_NOTE, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV,
      MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_BHC
    ),
    popis = c(
      "Propouštěcí zprávy; primární korpus práce.",
      "Radiologické zprávy; doplňkový korpus.",
      "Diagnózy ICD přiřazené hospitalizacím; zdroj štítků klasifikace.",
      "Číselník diagnóz ICD; názvy ke kódům.",
      "Výkony ICD přiřazené hospitalizacím; kontext zprávy.",
      "Číselník výkonů ICD; názvy ke kódům.",
      "Pacienti: pohlaví, věk a rok ukotvení; podklad dělení dat po pacientech.",
      "Hospitalizace: čas přijetí a propuštění, typ přijetí; vazba zpráv na pacienty.",
      "Služby, na kterých pacient během hospitalizace ležel; kontext zprávy.",
      "Předpisy léků; kontext zprávy.",
      "Lékárenské záznamy o lécích; kontext zprávy.",
      "Ambulantní měření (například výška, hmotnost, tlak); kontext zprávy.",
      "Brief Hospital Course vytažený regexem autorů datasetu; reference validace parseru sekcí."
    ),
    stringsAsFactors = FALSE
  )
}

manifest_tables <- function(path) {
  tables <- jsonlite::read_json(path)$tables
  rows <- lapply(seq_along(tables), function(index) {
    entry <- tables[[index]]
    file <- if (is.null(entry$file)) paste0(path, " [", index, "]") else entry$file
    for (field in c("file", "bytes", "records", "columns")) {
      if (is.null(entry[[field]])) {
        stop(sprintf("Soubor %s: chybí pole %s.", file, field))
      }
    }
    data.frame(
      tabulka = sub("\\.csv(\\.gz)?$", "", entry$file),
      bytes = as.double(entry$bytes),
      records = as.double(entry$records),
      sloupce = length(entry$columns)
    )
  })
  do.call(rbind, rows)
}

refuse_tables <- function(names, source) {
  if (length(names) > 0L) {
    stop(paste0(source, ": ", paste(sort(names), collapse = ", "), "."))
  }
}

corpus_overview <- function(counts, manifest) {
  catalog <- mimic_catalog()
  database <- counts$tabulka
  validated <- manifest$tabulka
  refuse_tables(setdiff(union(database, validated), catalog$tabulka), "Tabulky mimo katalog")
  refuse_tables(setdiff(validated, database), "Tabulky v manifestu validace, ale ne v databázi")
  refuse_tables(setdiff(database, validated), "Tabulky v databázi, ale ne v manifestu validace")
  refuse_tables(setdiff(c("discharge", "radiology"), database), "Povinné tabulky chybí v databázi")

  counts <- counts[order(counts$tabulka), ]
  names <- counts$tabulka
  manifest <- manifest[match(names, manifest$tabulka), ]
  table <- catalog[match(names, catalog$tabulka), ]
  table$n <- format_number(counts$n)
  table$sloupce <- format_number(manifest$sloupce)
  table$mb <- format_number(round(manifest$bytes / 1e6))
  differences <- names[counts$n != manifest$records]
  largest <- names[counts$n == max(counts$n)]
  prefix <- if (length(largest) == 1L) paste0("Nejvíce záznamů má tabulka ", largest, ". ") else ""
  verb <- if (length(differences) == 0L) "odpovídají" else "neodpovídají"
  summary <- if (length(differences) > 0L) {
    paste0(
      "Počty záznamů se liší od manifestu validace u tabulek: ",
      paste(differences, collapse = ", "),
      ". Data se proto nesmějí považovat za úplná a před dalším použitím je nutné rozdíl vysvětlit."
    )
  } else {
    paste0(
      "Počty záznamů u všech tabulek odpovídají manifestu validace. ",
      "Data jsou proto úplná a další části analýzy mohou stavět na parquetu bez další kontroly počtu záznamů."
    )
  }
  list(
    table = table,
    tables = format_number(nrow(counts)),
    discharge = format_number(counts$n[names == "discharge"]),
    radiology = format_number(counts$n[names == "radiology"]),
    records = paste0(prefix, "Počty záznamů v databázi ", verb, " počtům v manifestu validace."),
    summary = summary
  )
}

PROVENANCE_LABELS <- c(
  commit = "Revize kódu (commit)",
  dirty = "Necommitnuté změny",
  pixi_lock_sha256 = "Otisk pixi.lock (SHA-256)",
  data_source = "Zdroj dat",
  python = "Python",
  duckdb = "DuckDB",
  r = "R"
)

provenance_table <- function(details) {
  keys <- names(details)
  labels <- ifelse(
    keys %in% names(PROVENANCE_LABELS),
    PROVENANCE_LABELS[keys],
    sub("^csv_sha256 (.*)$", "Otisk \\1 (SHA-256)", keys)
  )
  values <- unlist(details, use.names = FALSE)
  values[keys == "dirty"] <- ifelse(values[keys == "dirty"] == "true", "ano", "ne")
  digest <- grepl("^[0-9a-f]{64}$", values)
  values[digest] <- gsub("(.{16})(?=.)", "\\1 ", values[digest], perl = TRUE)
  data.frame(Položka = unname(labels), Hodnota = values)
}
