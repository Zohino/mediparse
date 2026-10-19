# Token je maximální úsek písmen a-z délky aspoň 2 po převodu na malá písmena. Minimální délka je
# jako v `token_pattern` sklearn, ale sklearn tokenizuje `\\w\\w+`, takže ponechá číslice a `_`;
# tady maska `___`, čísla, časy a dávkování vypadnou. Pravidlo žije jen tady.
TOKEN_SQL <- "regexp_extract_all(lower(%s), '[a-z]{2,}')"
NGRAM_SIZES <- 1:3
NGRAM_RANKING <- 15L
SECTION_NGRAMS <- 5L
# Fráze šablony je n-gram, který má aspoň polovina zpráv; vlastní konstanta, protože má jiný důvod
# změny než CANDIDATE_COVERAGE (kandidáti na lexikon) a SKELETON_COVERAGE (sestavení kostry).
BOILERPLATE_COVERAGE <- 0.5
STOP_TABLE <- "ngram_stop_input"
FLOOR_TABLE <- "ngram_floor_input"
TOKENS_TABLE <- "ngram_tokens_input"
NGRAM_COLUMNS <- c("sekce", "n", "ngram", "zpravy", "vyskyty", "jmenovatel")

# Stejná stop slova jako `TfidfVectorizer(stop_words = "english")`, tedy i negace "no" a "not".
stop_words <- function() {
  frozen <- reticulate::import("sklearn.feature_extraction.text", convert = FALSE)$ENGLISH_STOP_WORDS
  as.character(unlist(reticulate::import_builtins()$sorted(frozen)))
}

empty_counts <- function() {
  data.frame(
    sekce = character(), n = integer(), ngram = character(),
    zpravy = numeric(), vyskyty = numeric(), jmenovatel = numeric(),
    stringsAsFactors = FALSE
  )
}

# Text těla vyřízne z `discharge` podle offsetů z parseru (code pointy, jako `substring`), takže
# pravidlo hranice sekce zůstává jen v parseru. Stop slova se odfiltrují před skládáním n-gramů,
# stejně jako v `TfidfVectorizer`. `list_filter` drží pořadí slov a běží proudově po řádcích;
# rozbalení přes `unnest` a zpětné `list(w ORDER BY i)` drželo všechna slova korpusu v paměti
# (seřazený agregát DuckDB neodkládá na disk) a nad MIMIC spadlo. Seznam stop slov jde do lambdy
# jako sloupec jednořádkové tabulky; literál se 318 řetězci byl na sondě třikrát pomalejší.
# Tabulka tokenů má jeden řádek na část zprávy a seznam slov v pořadí textu.
ngram_tokens <- function(con, stopwords, sections) {
  write_temp(con, STOP_TABLE, data.frame(w = stopwords, stringsAsFactors = FALSE))
  scope <- if (is.null(sections)) {
    ""
  } else {
    sprintf("WHERE s.section IN (%s)", paste(DBI::dbQuoteString(con, sections), collapse = ", "))
  }
  key <- if (is.null(sections)) "NULL::VARCHAR" else "s.section"
  DBI::dbExecute(con, sprintf("
    CREATE OR REPLACE TEMP TABLE %1$s AS
    WITH stop AS (SELECT coalesce(list(w), []) AS words FROM %6$s)
    SELECT s.note_id, s.position, %2$s AS sekce,
      list_filter(%3$s, w -> NOT list_contains(stop.words, w)) AS t
    FROM %4$s s JOIN discharge d ON d.note_id = s.note_id CROSS JOIN stop
    %5$s
  ", TOKENS_TABLE, key, sprintf(
    TOKEN_SQL, "substring(d.text, s.body_start + 1, s.body_end - s.body_start)"
  ), SECTIONS_TABLE, scope, STOP_TABLE))
  TOKENS_TABLE
}

# Jmenovatel brány: u celku zprávy vstupního korpusu (`segment_corpus` je plní z něj), u sekce
# zprávy, které ji mají.
ngram_denominators <- function(con, sections) {
  if (is.null(sections)) {
    total <- DBI::dbGetQuery(con, sprintf("SELECT count(DISTINCT note_id) AS n FROM %s", SECTIONS_TABLE))$n
    return(data.frame(sekce = NA_character_, jmenovatel = as.numeric(total), stringsAsFactors = FALSE))
  }
  out <- DBI::dbGetQuery(con, sprintf("
    SELECT section AS sekce, count(DISTINCT note_id) AS jmenovatel FROM %s
    WHERE section IN (%s) GROUP BY section
  ", SECTIONS_TABLE, paste(DBI::dbQuoteString(con, sections), collapse = ", ")))
  out$jmenovatel <- as.numeric(out$jmenovatel)
  out
}

ngram_list_sql <- function(size) {
  if (size == 1L) {
    return("t")
  }
  parts <- sprintf("t[i + %d]", seq_len(size) - 1L)
  sprintf("[concat_ws(' ', %s) FOR i IN range(1, len(t) - %d + 2)]", paste(parts, collapse = ", "), size)
}

# Brána DUA je v SQL, takže do R jde jen to, co se smí zveřejnit. `sections = NULL` je celek
# (dokumentová frekvence přes všechny části zprávy), jinak se počítá po zadaných sekcích.
ngram_counts <- function(con, stopwords, sections = NULL, floor_fn = publishable_floor) {
  if (length(sections) == 0L && !is.null(sections)) {
    return(empty_counts())
  }
  tokens <- ngram_tokens(con, stopwords, sections)
  on.exit(
    for (table in c(STOP_TABLE, TOKENS_TABLE, FLOOR_TABLE)) {
      DBI::dbExecute(con, sprintf("DROP TABLE IF EXISTS %s", table))
    },
    add = TRUE
  )
  denominators <- ngram_denominators(con, sections)
  denominators$prah <- as.numeric(floor_fn(denominators$jmenovatel))
  limits <- write_temp(con, FLOOR_TABLE, denominators)
  per_size <- lapply(NGRAM_SIZES, function(size) {
    DBI::dbGetQuery(con, sprintf("
      WITH grams AS (
        SELECT note_id, sekce, unnest(%2$s) AS ngram FROM %1$s
      ),
      per_note AS (
        SELECT note_id, sekce, ngram, count(*) AS vyskyty FROM grams GROUP BY ALL
      )
      SELECT p.sekce, %3$d AS n, p.ngram, count(*) AS zpravy, sum(p.vyskyty) AS vyskyty,
        any_value(f.jmenovatel) AS jmenovatel
      FROM per_note p JOIN %4$s f ON f.sekce IS NOT DISTINCT FROM p.sekce
      GROUP BY p.sekce, p.ngram
      HAVING count(*) >= any_value(f.prah)
    ", tokens, ngram_list_sql(size), size, limits))
  })
  out <- do.call(rbind, c(list(empty_counts()), per_size))
  out$n <- as.integer(out$n)
  out[, NGRAM_COLUMNS]
}

# Řazení jako `ranked_headers` (počet sestupně, pak řetězec), navíc výskyty; `radix` řadí řetězce
# bez ohledu na locale, aby pořadí nezáviselo na stroji.
ngram_ranking <- function(counts, limit) {
  ordered <- counts[order(counts$sekce, counts$n, -counts$zpravy, -counts$vyskyty, counts$ngram, method = "radix"), ]
  group <- paste(ifelse(is.na(ordered$sekce), "", ordered$sekce), ordered$n)
  rank <- stats::ave(seq_len(nrow(ordered)), group, FUN = seq_along)
  rownames(ordered) <- NULL
  ordered[rank <= limit, , drop = FALSE]
}

# Popisky a tvary podle délky n-gramu; klíč je n.
NGRAM_KINDS <- list(
  `1` = list(label = "Tokeny", noun = "token", forms = c("token", "tokeny", "tokenů")),
  `2` = list(label = "Bigramy", noun = "bigram", forms = c("bigram", "bigramy", "bigramů")),
  `3` = list(label = "Trigramy", noun = "trigram", forms = c("trigram", "trigramy", "trigramů"))
)
NGRAM_LABELS <- vapply(NGRAM_KINDS, `[[`, character(1), "label", USE.NAMES = FALSE)

# Rozvržení tabulek n-gramů; šířky se opakují v `tbl-colwidths` chunků sešitu, shodu hlídá test.
NGRAM_LAYOUTS <- list(
  `tbl-ngramy` = list(
    col.names = c("Délka", "N-gram", "Zprávy", "Podíl (%)", "Výskyty"),
    align = "clrrr", widths = c(8, 44, 16, 16, 16)
  ),
  `tbl-ngramy-sekce` = list(
    col.names = c("Sekce", "Délka", "Nejčastější n-gramy (podíl zpráv se sekcí)"),
    align = "lcl", widths = c(22, 8, 70)
  )
)

ngram_table <- function(ranked) {
  data.frame(
    n = as.character(ranked$n),
    ngram = ranked$ngram,
    zpravy = number_text(ranked$zpravy),
    podil = percent_text(ranked$zpravy, ranked$jmenovatel),
    vyskyty = number_text(ranked$vyskyty),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

# Jeden řádek na sekci kostry a n; sekce bez n-gramu nad branou má v buňce pomlčku.
section_ngram_table <- function(ranked, sections) {
  grid <- expand.grid(n = NGRAM_SIZES, sekce = sections, stringsAsFactors = FALSE)
  grid <- grid[order(match(grid$sekce, sections), grid$n), , drop = FALSE]
  cell <- function(section, size) {
    rows <- ranked[ranked$sekce %in% section & ranked$n == size, , drop = FALSE]
    if (nrow(rows) == 0L) {
      return("–")
    }
    paste(
      sprintf("%s (%s)", rows$ngram, percent_label(rows$zpravy, rows$jmenovatel)),
      collapse = "; "
    )
  }
  data.frame(
    sekce = grid$sekce,
    n = as.character(grid$n),
    ngramy = vapply(seq_len(nrow(grid)), function(i) cell(grid$sekce[[i]], grid$n[[i]]), character(1)),
    stringsAsFactors = FALSE
  )
}

ngram_plot <- function(ranked) {
  bars <- ranked
  bars$klic <- paste(bars$n, bars$ngram, sep = "|")
  bars$podil <- ifelse(bars$jmenovatel > 0, bars$zpravy / bars$jmenovatel, NA_real_)
  bars$klic <- factor(bars$klic, levels = rev(unique(bars$klic)))
  bars$druh <- factor(NGRAM_LABELS[bars$n], levels = NGRAM_LABELS)
  # facet_grid nad prázdnou tabulkou spadne, graf bez n-gramů proto zůstane bez panelů.
  panels <- if (nrow(bars) == 0L) NULL else ggplot2::facet_grid(druh ~ ., scales = "free_y", space = "free_y")
  ggplot2::ggplot(bars, ggplot2::aes(podil, klic)) +
    ggplot2::geom_col() +
    panels +
    ggplot2::scale_x_continuous(labels = percent_axis) +
    ggplot2::scale_y_discrete(labels = function(x) sub("^[0-9]+\\|", "", x)) +
    ggplot2::labs(x = "Podíl zpráv", y = NULL) +
    theme_thesis()
}

top_ngram_text <- function(counts, size) {
  rows <- counts[counts$n == size, , drop = FALSE]
  if (nrow(rows) == 0L) {
    return(NULL)
  }
  best <- ngram_ranking(rows, limit = 1L)
  sprintf(
    "%s `%s` (%s zpráv)", NGRAM_KINDS[[as.character(size)]]$noun, best$ngram,
    percent_label(best$zpravy, best$jmenovatel)
  )
}

# Fráze šablony jsou n-gramy, které má aspoň polovina zpráv; podklad pro `max_df` v S40 a D26.
ngram_text <- function(counts) {
  if (nrow(counts) == 0L) {
    return("Žádný n-gram nepřekročil bránu DUA, nejčastější n-gramy nelze uvést.")
  }
  tops <- Filter(Negate(is.null), lapply(NGRAM_SIZES, function(size) top_ngram_text(counts, size)))
  common <- counts$zpravy >= BOILERPLATE_COVERAGE * counts$jmenovatel
  counted <- vapply(NGRAM_SIZES, function(size) {
    count_phrase(sum(common & counts$n == size), NGRAM_KINDS[[as.character(size)]]$forms)
  }, character(1))
  sprintf(
    "Nejčastější je %s. Aspoň %s zpráv má %s; jsou to fráze šablony, které nic nerozlišují.",
    join_labels(unlist(tops)), coverage_threshold_text(BOILERPLATE_COVERAGE),
    join_labels(counted)
  )
}
