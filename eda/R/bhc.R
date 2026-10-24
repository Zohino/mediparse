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
