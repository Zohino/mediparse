# Konzument má záměrně vlastní deklaraci nezávislou na SCHEMA v Pythonu;
# sdílená fixture zachytí změnu producenta, kterou by export schématu skryl.
METRICS_COLUMNS <- c(
  row_id = "VARCHAR", diagnosis = "VARCHAR", test_notes = "BIGINT",
  test_positives = "BIGINT", precision = "DOUBLE", recall = "DOUBLE",
  f1 = "DOUBLE", accuracy = "DOUBLE", roc_auc = "DOUBLE", pr_auc = "DOUBLE"
)

read_metrics <- function(path) read_contract_table(path, METRICS_COLUMNS)
