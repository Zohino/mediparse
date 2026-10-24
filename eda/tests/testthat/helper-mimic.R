EMPTY_TABLES <- list(
  discharge = c("note_id", "subject_id", "hadm_id", "text"),
  radiology = c("note_id", "subject_id", "hadm_id", "text"),
  diagnoses_icd = c("hadm_id", "icd_code", "icd_version"),
  `mimic-iv-bhc` = c("note_id", "target"),
  d_icd_diagnoses = c("icd_code", "icd_version", "long_title"),
  patients = c("subject_id", "anchor_year_group", "gender", "anchor_age", "anchor_year"),
  admissions = c(
    "hadm_id", "subject_id", "admittime", "insurance", "language", "marital_status", "race"
  )
)

text_frame <- function(rows, columns) {
  if (is.null(rows)) {
    return(as.data.frame(
      setNames(rep(list(character()), length(columns)), columns),
      stringsAsFactors = FALSE
    ))
  }
  frame <- as.data.frame(rows, stringsAsFactors = FALSE)
  for (column in setdiff(columns, names(frame))) {
    frame[[column]] <- rep(NA_character_, nrow(frame))
  }
  frame
}

mimic_con <- function(..., env = parent.frame()) {
  given <- list(...)
  dir <- file.path(withr::local_tempdir(.local_envir = env), "parquet")
  dir.create(dir)
  writeLines('{"prefix":"t","rows":1}', file.path(dir, "synthetic.json"))
  writer <- DBI::dbConnect(duckdb::duckdb())
  for (name in names(EMPTY_TABLES)) {
    frame <- text_frame(given[[name]], EMPTY_TABLES[[name]])
    DBI::dbWriteTable(writer, name, frame)
    DBI::dbExecute(writer, sprintf(
      "COPY %s TO '%s' (FORMAT parquet)",
      DBI::dbQuoteIdentifier(writer, name), file.path(dir, paste0(name, ".parquet"))
    ))
  }
  DBI::dbDisconnect(writer, shutdown = TRUE)
  withr::local_envvar(MEDIPARSE_PARQUET = dir, .local_envir = env)
  con <- eda_connect()
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE), envir = env)
  con
}

text_notes <- function(texts, prefix = "n") {
  data.frame(
    note_id = paste0(prefix, seq_along(texts)),
    subject_id = as.character(seq_along(texts)),
    hadm_id = as.character(seq_along(texts)),
    text = texts,
    stringsAsFactors = FALSE
  )
}

corpus_con <- function(texts, codes = rep("Z99", length(texts)), env = parent.frame()) {
  hadm <- paste0("h", seq_along(texts))
  padding <- paste0(strrep("x", SHORT_LIMIT), " ", hadm, "\n")
  notes <- text_notes(paste0(padding, texts))
  notes$hadm_id <- hadm
  rows <- data.frame(hadm_id = hadm, icd_code = codes, icd_version = "10", stringsAsFactors = FALSE)
  mimic_con(discharge = notes, diagnoses_icd = rows, env = env)
}
