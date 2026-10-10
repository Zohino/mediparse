MIMIC_IV <- "MIMIC-IV (hosp)"
MIMIC_NOTE <- "MIMIC-IV-Note (note)"

mimic_catalog <- function() {
  data.frame(
    tabulka = c(
      "discharge", "radiology", "diagnoses_icd", "d_icd_diagnoses",
      "procedures_icd", "d_icd_procedures", "patients", "admissions",
      "services", "prescriptions", "pharmacy", "omr"
    ),
    zdroj = c(
      MIMIC_NOTE, MIMIC_NOTE, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV,
      MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV, MIMIC_IV
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
      "Ambulantní měření (například výška, hmotnost, tlak); kontext zprávy."
    ),
    stringsAsFactors = FALSE
  )
}

manifest_text <- function(tables, shapes, sizes) {
  data.frame(
    tabulka = tables,
    sloupce = vapply(
      tables, function(name) as.character(length(shapes[[name]]$columns)), character(1)
    ),
    mb = vapply(
      tables,
      function(name) format(round(sizes[[name]] / 1e6), big.mark = NBSP, trim = TRUE),
      character(1)
    ),
    records = vapply(tables, function(name) shapes[[name]]$records, numeric(1)),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

code_names <- function(x) {
  ifelse(x == DUA_OTHER, x, paste0("`", x, "`"))
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
