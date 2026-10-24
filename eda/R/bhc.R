BHC_TABLE <- "mimic-iv-bhc"
BHC_SECTION <- "hospital_course"

# `\s` v RE2 (DuckDB) je jen ASCII, kdežto Pythonovo `\s` nad `str` pokrývá Unicode, a skript
# clin-bhc-summ čistí v Pythonu. Výčet se shoduje s Pythonovým `re` přes všechny code pointy
# (NBSP, U+2028 nebo U+000B by se se `\s` neslily a daly falešnou neshodu).
# Výčet bez obyčejné mezery je v samostatné konstantě, ze které se odvozuje i kontrola targetu.
BHC_NON_SPACE_WHITESPACE <- paste0(
  "\\t-\\r\\x{1c}-\\x{1f}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}",
  "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
)
BHC_WHITESPACE <- paste0("[", BHC_NON_SPACE_WHITESPACE, " ]")
BHC_UNCLEANED_PATTERN <- paste0("  |[", BHC_NON_SPACE_WHITESPACE, "]")

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
  cela_sekce = "Reference obsahuje celou naši následující sekci",
  konec_driv = "Náš konec dřív",
  konec_pozdeji = "Náš konec později",
  zacatek_driv = "Náš začátek dřív",
  zacatek_pozdeji = "Náš začátek později",
  jina = "Jiná neshoda"
)
BHC_LENGTH_CATEGORIES <- c(
  "cela_sekce", "konec_driv", "konec_pozdeji", "zacatek_driv", "zacatek_pozdeji"
)
# Kategorie, kde je náš úsek vlastní prefix targetu; otázka, čí hlavička ho uřízla, platí pro obě.
BHC_CUT_CATEGORIES <- c("cela_sekce", "konec_driv")

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
# Následující část se čistí a ořezává stejně jako náš úsek, protože target prošel týmž čištěním.
# `other` se z následující části vylučuje: má prázdnou hlavičku i tělo a `starts_with(x, '')` je
# vždy pravda, takže by do `cela_sekce` spadla každá zpráva s administrativním polem za BHC.
# Chybí-li následující část (BHC je poslední), je `next_part` NULL a větev propadne.
pair_bhc_sql <- function(con, other) {
  sprintf("
    CREATE OR REPLACE TEMP TABLE %1$s AS
    WITH ours AS (
      SELECT s.note_id,
        trim(%2$s) AS ours,
        trim(%8$s) AS next_part
      FROM %3$s s JOIN discharge d ON d.note_id = s.note_id
      LEFT JOIN %3$s n ON n.note_id = s.note_id AND n.position = s.position + 1 AND n.section <> %9$s
      WHERE s.section = %4$s
    ),
    theirs AS (
      SELECT note_id, target AS raw, trim(target) AS target FROM %5$s
    ),
    pairs AS (
      SELECT t.note_id, t.raw, t.target, o.ours, o.next_part, o.note_id IS NULL AS missing
      FROM theirs t
      JOIN (SELECT DISTINCT note_id FROM %3$s) c ON c.note_id = t.note_id
      LEFT JOIN ours o ON o.note_id = t.note_id
    ),
    judged AS (
      SELECT note_id, ours, target,
        regexp_matches(raw, %6$s) AS necisty,
        CASE
          WHEN missing THEN 'nenalezena'
          WHEN ours = target THEN 'shoda'
          WHEN starts_with(target, ours)
            AND starts_with(ltrim(substr(target, length(ours) + 1)), next_part) THEN 'cela_sekce'
          WHEN starts_with(target, ours) THEN 'konec_driv'
          WHEN starts_with(ours, target) THEN 'konec_pozdeji'
          WHEN contains(ours, target) THEN 'zacatek_driv'
          WHEN contains(target, ours) THEN 'zacatek_pozdeji'
          ELSE 'jina'
        END AS kategorie
      FROM pairs
    )
    SELECT note_id, kategorie, necisty,
      CASE WHEN kategorie IN (%7$s) THEN abs(length(ours) - length(target)) END AS rozdil
    FROM judged
  ", BHC_PAIRS_TABLE,
    bhc_clean_sql(part_text_sql()),
    SECTIONS_TABLE, DBI::dbQuoteString(con, BHC_SECTION),
    DBI::dbQuoteIdentifier(con, BHC_TABLE), DBI::dbQuoteString(con, BHC_UNCLEANED_PATTERN),
    paste(DBI::dbQuoteString(con, BHC_LENGTH_CATEGORIES), collapse = ", "),
    bhc_clean_sql(part_text_sql(alias = "n")), DBI::dbQuoteString(con, other))
}

# Porovnává se úsek hlavička + tělo sekce `hospital_course` s targetem BHC. Čistí se jen náš úsek,
# protože target už čištěním prošel; druhé čištění by slité řádky ("Brief Hospital" + "Course:")
# odstranilo. `necisty_target` počítá jen mezi spárovanými zprávami targety, které mají dvě mezery
# za sebou nebo jiný bílý znak než mezeru, tedy ty, které nemohly projít kroky 1, 3 a 4 čištění.
# Krok 2 (odstranění hlaviček) takhle ověřit nejde.
bhc_agreement <- function(con, other) {
  check_bhc_table(con)
  DBI::dbExecute(con, pair_bhc_sql(con, other))
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

# Šířky se opakují v `tbl-colwidths` chunku sešitu (Quarto je čte z hlavičky chunku), shodu hlídá test.
BHC_LAYOUTS <- list(
  `tbl-bhc` = list(
    col.names = c("Kategorie", "Zprávy", "Podíl (%)", "Medián rozdílu délky"),
    align = "lrrr", widths = c(40, 20, 18, 22)
  ),
  `tbl-bhc-konec` = list(
    col.names = c("Sekce", "Popisek", "Zprávy", "Podíl (%)"),
    align = "llrr", widths = c(28, 40, 16, 16)
  )
)

# Řazení podle BHC_CATEGORIES, ne podle pořadí z databáze; tabulka a graf mají tak stejné pořadí.
ordered_categories <- function(agreement) {
  categories <- agreement$kategorie
  categories[match(names(BHC_CATEGORIES), categories$kategorie), , drop = FALSE]
}

bhc_table <- function(agreement) {
  categories <- ordered_categories(agreement)
  data.frame(
    kategorie = categories$popisek,
    n = number_text(categories$n),
    podil = percent_text(categories$n, agreement$sparovano),
    median = number_text(categories$median_rozdilu),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

bhc_plot <- function(agreement) {
  categories <- ordered_categories(agreement)
  bars <- data.frame(
    popisek = factor(categories$popisek, levels = categories$popisek),
    podil = if (agreement$sparovano > 0) categories$n / agreement$sparovano else rep(0, nrow(categories))
  )
  ggplot2::ggplot(bars, ggplot2::aes(podil, popisek)) +
    ggplot2::geom_col() +
    ggplot2::scale_y_discrete(limits = rev) +
    ggplot2::scale_x_continuous(labels = percent_axis) +
    ggplot2::labs(x = "Podíl spárovaných zpráv", y = NULL) +
    theme_thesis()
}

bhc_mismatch_phrase <- function(categories) {
  mismatches <- categories[categories$kategorie != "shoda", , drop = FALSE]
  paste(
    sprintf("%s: %s", mismatches$popisek, count_phrase(mismatches$n, NOTE_FORMS)),
    collapse = "; "
  )
}

bhc_text <- function(agreement) {
  outside <- sprintf(
    "Mimo průnik zůstávají zprávy jen v BHC (%s) a jen v korpusu (%s); nejsou to chyby parseru, protože BHC vyřadil zprávy bez sekce Sex a s BHC kratší než 100 znaků a korpus filtruje podle ICD, délky a duplicit.",
    count_phrase(agreement$jen_bhc, NOTE_FORMS), count_phrase(agreement$jen_korpus, NOTE_FORMS)
  )
  if (agreement$sparovano == 0) {
    return(paste(
      "Shodu s MIMIC-IV-Ext-BHC nelze posoudit, protože žádná zpráva korpusu nemá pár v MIMIC-IV-Ext-BHC.",
      outside
    ))
  }
  categories <- ordered_categories(agreement)
  matched <- categories$n[categories$kategorie == "shoda"]
  sprintf(
    "Počet spárovaných zpráv: %s. Úsek `%s` se shoduje s targetem BHC u %s spárovaných zpráv (%s). Neshody: %s. %s Target čištěním zjevně neprošel (dvě mezery za sebou nebo jiný bílý znak než mezera) u %s spárovaných zpráv (%s).",
    number_text(agreement$sparovano), BHC_SECTION,
    percent_label(matched, agreement$sparovano), count_phrase(matched, NOTE_FORMS),
    bhc_mismatch_phrase(categories), outside,
    percent_label(agreement$necisty_target, agreement$sparovano),
    count_phrase(agreement$necisty_target, NOTE_FORMS)
  )
}

# Část `other` za sekcí začíná jen na administrativním poli lexikonu (split_sections), text před
# první hlavičkou za sekcí ležet nemůže; popisek proto jmenuje pole.
BHC_FIELD_LABEL <- "Administrativní pole"

BHC_NO_NEXT <- "bez_nasledujici"
BHC_NO_NEXT_LABEL <- "Bez následující části"

check_pairs_table <- function(con) {
  if (!BHC_PAIRS_TABLE %in% DBI::dbListTables(con)) {
    stop("Tabulka párů BHC chybí; rozklad konce dřív se volá až po bhc_agreement.")
  }
}

# Zprávy, kde je náš úsek vlastním prefixem targetu, s částí, která následuje hned za
# `hospital_course` (LEFT JOIN: BHC může být poslední částí zprávy). Sdílí ji rozklad podle
# sekce i rozklad příčin, takže oba mají tutéž populaci. Z následující části se bere jen to,
# co dotazy potřebují: sekce, hlavička a konec těla.
cut_course_sql <- function(con) {
  sprintf("
    SELECT s.note_id, s.position AS course_position, d.text,
      n.section AS next_section, n.position AS next_position, n.body_end AS next_end,
      %5$s AS next_header
    FROM %1$s s
    JOIN %2$s p ON p.note_id = s.note_id
    JOIN discharge d ON d.note_id = s.note_id
    LEFT JOIN %1$s n ON n.note_id = s.note_id AND n.position = s.position + 1
    WHERE s.section = %3$s AND p.kategorie IN (%4$s)
  ", SECTIONS_TABLE, BHC_PAIRS_TABLE, DBI::dbQuoteString(con, BHC_SECTION),
    paste(DBI::dbQuoteString(con, BHC_CUT_CATEGORIES), collapse = ", "),
    header_text_sql("d.text", "n"))
}

# Která část zprávy následuje hned za `hospital_course` u zpráv, kde náš úsek končí dřív než
# target, tedy čí hlavička úsek uřízla. Popisky jsou z lexikonu, ne z dat, takže se nic nepotlačuje.
# Čte tabulku párů, kterou vytvoří `bhc_agreement`.
bhc_cut_sections <- function(con, lexicon) {
  check_pairs_table(con)
  found <- DBI::dbGetQuery(con, sprintf("
    SELECT coalesce(next_section, %2$s) AS sekce, count(*) AS n FROM (%1$s) GROUP BY 1
  ", cut_course_sql(con), DBI::dbQuoteString(con, BHC_NO_NEXT)))
  found$n <- as.numeric(found$n)
  keys <- lexicon_keys(lexicon)
  diagnostic <- unique(lexicon$sections$sekce[lexicon$sections$diagnostic])
  place <- match(found$sekce, keys$sekce)
  # Pole `other` a chybějící následující část nemají místo v lexikonu a řadí se na konec.
  rank <- ifelse(is.na(place), nrow(keys) + 1 + (found$sekce == BHC_NO_NEXT), keys$poradi[place])
  popisek <- ifelse(is.na(place), BHC_FIELD_LABEL, keys$label_cs[place])
  out <- data.frame(
    sekce = found$sekce,
    popisek = ifelse(found$sekce == BHC_NO_NEXT, BHC_NO_NEXT_LABEL, popisek),
    n = found$n,
    diagnosticka = found$sekce %in% diagnostic,
    celkem = rep(sum(found$n), nrow(found)),
    stringsAsFactors = FALSE
  )
  out <- out[order(-out$n, rank), , drop = FALSE]
  rownames(out) <- NULL
  out
}

bhc_cut_table <- function(cut) {
  data.frame(
    sekce = cut$sekce,
    popisek = cut$popisek,
    n = number_text(cut$n),
    podil = percent_text(cut$n, cut$celkem),
    stringsAsFactors = FALSE
  )
}

bhc_cut_text <- function(cut) {
  if (nrow(cut) == 0) {
    return("Kterou hlavičkou náš úsek končí dřív než BHC, nelze posoudit, protože v kategorii „náš konec dřív“ není žádná zpráva.")
  }
  total <- cut$celkem[[1]]
  top <- utils::head(cut, 3)
  cut_by_diagnosis <- sum(cut$n[cut$diagnosticka])
  sprintf(
    "Náš úsek končí dřív než BHC nejčastěji na hlavičce těchto částí: %s. Uřízne ho diagnostická sekce u %s z nich (%s); strip D2 u těchto zpráv odstraní spolu s ní i text, který BHC počítá do Brief Hospital Course.",
    paste(sprintf(
      "%s (%s, %s)", top$popisek, count_phrase(top$n, NOTE_FORMS), percent_label(top$n, total)
    ), collapse = "; "),
    percent_label(cut_by_diagnosis, total), count_phrase(cut_by_diagnosis, NOTE_FORMS)
  )
}
