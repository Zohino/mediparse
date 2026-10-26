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
