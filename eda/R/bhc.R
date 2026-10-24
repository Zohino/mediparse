BHC_TABLE <- "mimic-iv-bhc"
BHC_SECTION <- "hospital_course"

# `\s` v RE2 (DuckDB) je jen ASCII, kdežto Pythonovo `\s` nad `str` pokrývá Unicode, a skript
# clin-bhc-summ čistí v Pythonu. Výčet se shoduje s Pythonovým `re` přes všechny code pointy
# (NBSP, U+2028 nebo U+000B by se se `\s` neslily a daly falešnou neshodu).
BHC_WHITESPACE <- paste0(
  "[\\t-\\r\\x{1c}-\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}",
  "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}]"
)

# Čtyři zápisy hlavičky, které čištění odstraňuje všude v textu, s rozlišením velikosti písmen
# a v tomto pořadí; "Brief hospital course:" se neodstraní.
BHC_HEADERS <- c(
  "Brief Hospital Course:\n", "BRIEF HOSPITAL COURSE:\n", "BRIEF HOSPITAL COURSE", "Brief Hospital Course"
)

# Čištění targetu ze skriptu clin-bhc-summ (utils/mimic_iv_bhc_preprocessing.py) krok za krokem:
# sloučit mezery, odstranit hlavičky, zalomení na mezeru, sloučit bílé znaky.
bhc_clean_sql <- function(column) {
  cleaned <- sprintf("regexp_replace(%s, ' +', ' ', 'g')", column)
  for (header in BHC_HEADERS) {
    literal <- gsub("\n", "' || chr(10) || '", sprintf("'%s'", header), fixed = TRUE)
    cleaned <- sprintf("replace(%s, %s, '')", cleaned, literal)
  }
  cleaned <- sprintf("replace(%s, chr(10), ' ')", cleaned)
  sprintf("regexp_replace(%s, '%s+', ' ', 'g')", cleaned, BHC_WHITESPACE)
}

BHC_PAIRS_TABLE <- "bhc_pairs_input"

# Pořadí je pořadí kontrol v SQL (CASE bere první splněnou větev), takže kategorie jsou
# disjunktní a vyčerpávající; tabulka i graf se řadí podle něj.
BHC_CATEGORIES <- c(
  nenalezena = "Sekce nenalezena",
  shoda = "Shoda",
  konec_driv = "Náš konec dřív",
  konec_pozdeji = "Náš konec později",
  zacatek_driv = "Náš začátek dřív",
  zacatek_pozdeji = "Náš začátek později",
  jina = "Jiná neshoda"
)
BHC_LENGTH_CATEGORIES <- c("konec_driv", "konec_pozdeji", "zacatek_driv", "zacatek_pozdeji")

quoted_list <- function(con, values) paste(DBI::dbQuoteString(con, values), collapse = ", ")

check_bhc_table <- function(con) {
  if (!BHC_TABLE %in% DBI::dbListTables(con)) {
    stop(sprintf(
      "V databázi chybí tabulka %s; stáhni ji a převeď přes `snakemake parquet_mimic`.", BHC_TABLE
    ))
  }
  row <- DBI::dbGetQuery(con, sprintf(
    "SELECT count(*) - count(DISTINCT note_id) AS duplicity, count(*) - count(target) AS bez_targetu FROM %s",
    DBI::dbQuoteIdentifier(con, BHC_TABLE)
  ))
  if (row$duplicity > 0) {
    stop(sprintf("Tabulka %s má duplicitní nebo chybějící note_id; jmenovatel by byl zkreslený.", BHC_TABLE))
  }
  if (row$bez_targetu > 0) {
    stop(sprintf("Tabulka %s má NULL v target; jmenovatel by byl zkreslený.", BHC_TABLE))
  }
}

# Do tabulky jde jen kategorie a rozdíl délky po zprávách, ne texty: agregát nad texty
# 270 tisíc zpráv došel paměti už v S27a3.
pair_bhc_sql <- function(con) {
  sprintf("
    CREATE OR REPLACE TEMP TABLE %1$s AS
    WITH ours AS (
      SELECT s.note_id,
        trim(%2$s) AS ours
      FROM %3$s s JOIN discharge d ON d.note_id = s.note_id
      WHERE s.section = %4$s
    ),
    theirs AS (
      SELECT note_id, trim(target) AS target FROM %5$s
    ),
    pairs AS (
      SELECT t.note_id, t.target, o.ours, o.note_id IS NULL AS missing
      FROM theirs t
      JOIN (SELECT DISTINCT note_id FROM %3$s) c ON c.note_id = t.note_id
      LEFT JOIN ours o ON o.note_id = t.note_id
    ),
    judged AS (
      SELECT note_id, ours, target,
        CASE
          WHEN missing THEN 'nenalezena'
          WHEN ours = target THEN 'shoda'
          WHEN starts_with(target, ours) THEN 'konec_driv'
          WHEN starts_with(ours, target) THEN 'konec_pozdeji'
          WHEN contains(ours, target) THEN 'zacatek_driv'
          WHEN contains(target, ours) THEN 'zacatek_pozdeji'
          ELSE 'jina'
        END AS kategorie,
        trim(%6$s) <> target AS necisty
      FROM pairs
    )
    SELECT note_id, kategorie, necisty,
      CASE WHEN kategorie IN (%7$s) THEN abs(length(ours) - length(target)) END AS rozdil
    FROM judged
  ", BHC_PAIRS_TABLE,
    bhc_clean_sql("substring(d.text, s.body_end - s.span_chars + 1, s.span_chars)"),
    SECTIONS_TABLE, DBI::dbQuoteString(con, BHC_SECTION),
    DBI::dbQuoteIdentifier(con, BHC_TABLE), bhc_clean_sql("target"),
    quoted_list(con, BHC_LENGTH_CATEGORIES))
}

# Porovnává se úsek hlavička + tělo sekce `hospital_course` (začátek úseku je body_end - span_chars)
# s targetem BHC. Čistí se jen náš úsek, protože target už čištěním prošel; druhé čištění by slité
# řádky ("Brief Hospital" + "Course:") odstranilo. Že je target pevným bodem čištění, ověřuje
# `necisty_target`.
bhc_agreement <- function(con) {
  check_bhc_table(con)
  DBI::dbExecute(con, pair_bhc_sql(con))
  found <- DBI::dbGetQuery(con, sprintf("
    SELECT kategorie, count(*) AS n, quantile_disc(rozdil, 0.5) AS median_rozdilu
    FROM %s GROUP BY kategorie
  ", BHC_PAIRS_TABLE))
  found <- as_numeric_columns(found, c("n", "median_rozdilu"))
  place <- match(names(BHC_CATEGORIES), found$kategorie)
  kategorie <- data.frame(
    kategorie = names(BHC_CATEGORIES),
    popisek = unname(BHC_CATEGORIES),
    n = ifelse(is.na(place), 0, found$n[place]),
    median_rozdilu = found$median_rozdilu[place],
    stringsAsFactors = FALSE
  )
  counts <- DBI::dbGetQuery(con, sprintf("
    SELECT
      (SELECT count(*) FROM %1$s) AS sparovano,
      (SELECT count(*) FROM (SELECT DISTINCT note_id FROM %2$s) c
        WHERE c.note_id NOT IN (SELECT note_id FROM %1$s)) AS jen_korpus,
      (SELECT count(*) FROM %3$s WHERE note_id NOT IN (SELECT DISTINCT note_id FROM %2$s)) AS jen_bhc,
      (SELECT count(*) FROM %1$s WHERE necisty) AS necisty_target
  ", BHC_PAIRS_TABLE, SECTIONS_TABLE, DBI::dbQuoteIdentifier(con, BHC_TABLE)))
  c(lapply(counts, as.numeric), list(kategorie = kategorie))
}
