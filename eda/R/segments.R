SECTIONS_TABLE <- "note_sections_input"
SECTION_COLUMNS <- c(
  note_id = "VARCHAR", position = "INTEGER", section = "VARCHAR",
  span_chars = "INTEGER", body_chars = "INTEGER"
)

sections_module <- function() reticulate::import("mediparse.entrypoints.eda_sections")

# Texty a identifikátory musí do Pythonu jako nepojmenovaný seznam: pojmenovaný vektor
# reticulate převede na slovník, duplicitní jméno (dvě shodné zprávy) se v něm sloučí
# a zip() pak spadne na nestejných délkách; vektor délky 1 by se zase stal řetězcem.
section_rows <- function(module, lexicon_path, chunk) {
  path <- reticulate::import("pathlib")$Path(lexicon_path)
  rows <- module$note_sections(
    path, as.list(unname(chunk$note_id)), as.list(unname(chunk$text))
  )
  # Prázdný výsledek by reticulate vrátil jako data.frame bez sloupců, proto typy skládáme sami.
  data.frame(
    note_id = as.character(unlist(rows$note_id)),
    position = as.integer(unlist(rows$position)),
    section = as.character(unlist(rows$section)),
    span_chars = as.integer(unlist(rows$span_chars)),
    body_chars = as.integer(unlist(rows$body_chars)),
    stringsAsFactors = FALSE
  )
}

segment_corpus <- function(con, module, lexicon_path) {
  rows_sql <- sprintf(
    "SELECT c.note_id, d.text FROM (%s) c JOIN discharge d ON d.note_id = c.note_id",
    input_corpus_sql(con)
  )
  batched_table(
    con, SECTIONS_TABLE, SECTION_COLUMNS, rows_sql,
    function(chunk) section_rows(module, lexicon_path, chunk)
  )
}

# Kostra jsou sekce, které má většina zpráv; popisná hranice jako CANDIDATE_COVERAGE, ale jiný
# důvod změny (kandidáti na lexikon vs. sestavení kostry), proto vlastní konstanta.
SKELETON_COVERAGE <- 0.5
KEYS_TABLE <- "section_keys_input"
SKELETON_TABLE <- "skeleton_input"
# Sekce, kterou vystřihne ablace D2 (strip S32); S32 má číst týž klíč.
STRIP_SECTION <- "discharge_diagnosis"
OTHER_LABEL <- "Text mimo sekce"
HEADERLESS_LABELS <- c(bez_nadpisu = "Bez nadpisů", se_sekcemi = "Se sekcemi")
SECTION_FORMS <- c("sekce", "sekce", "sekcí")
CHAR_FORMS <- c("znak", "znaky", "znaků")
NOTE_FORMS <- c("zpráva", "zprávy", "zpráv")

# Rozvržení tabulek kapitoly; sešit i test exportu je berou odsud. Šířky se opakují v
# `tbl-colwidths` chunku (Quarto je čte z hlavičky chunku), shodu hlídá test.
TABLE_LAYOUTS <- list(
  `tbl-kostra` = list(
    col.names = c("Sekce", "Popisek", "V kostře", "Pokrytí (%)", "Medián pozice", "IQR pozice"),
    align = "llcrrr", widths = c(24, 30, 10, 12, 12, 12)
  ),
  `tbl-delky` = list(
    col.names = c("Sekce", "Popisek", "Zprávy se sekcí", "Medián", "p75", "p95"),
    align = "llrrrr", widths = c(24, 34, 10, 10, 11, 11)
  ),
  `tbl-bez-nadpisu` = list(
    col.names = c("Skupina", "Zprávy", "Podíl (%)", "Medián délky", "p95 délky"),
    align = "lrrrr", widths = c(30, 15, 15, 20, 20)
  )
)

lexicon_keys <- function(lexicon) {
  sections <- lexicon$sections[!duplicated(lexicon$sections$sekce), , drop = FALSE]
  data.frame(
    sekce = sections$sekce, label_cs = sections$label_cs,
    poradi = seq_len(nrow(sections)), stringsAsFactors = FALSE
  )
}

write_temp <- function(con, name, frame) {
  DBI::dbWriteTable(con, name, frame, temporary = TRUE, overwrite = TRUE)
  name
}

# Pozice je pořadí mezi rozpoznanými sekcemi zprávy, protože `position` z Pythonu počítá i `other`.
# Díky pravidlu prvního výskytu má sekce ve zprávě nejvýš jednu pozici.
# Kostra jde první, pak medián pozice, sestupné pokrytí a nakonec pořadí v lexikonu; pořadí
# nesmí záviset na tom, v jakém pořadí vrátí řádky databáze.
skeleton_order <- function(frame) order(!frame$v_kostre, frame$median, -frame$n, frame$poradi)

section_skeleton <- function(con, lexicon) {
  keys <- write_temp(con, KEYS_TABLE, lexicon_keys(lexicon))
  out <- DBI::dbGetQuery(con, sprintf("
    WITH recognized AS (
      SELECT s.note_id, k.sekce,
        row_number() OVER (PARTITION BY s.note_id ORDER BY s.position) - 1 AS pozice
      FROM %1$s s JOIN %2$s k ON k.sekce = s.section
    ),
    stats AS (
      SELECT sekce, count(*) AS n, quantile_cont(pozice, 0.5) AS median,
        quantile_cont(pozice, 0.25) AS q1, quantile_cont(pozice, 0.75) AS q3
      FROM recognized GROUP BY sekce
    )
    SELECT k.sekce, k.label_cs, k.poradi, coalesce(st.n, 0) AS n,
      (SELECT count(DISTINCT note_id) FROM %1$s) AS celkem,
      st.median, st.q1, st.q3
    FROM %2$s k LEFT JOIN stats st ON st.sekce = k.sekce
    ORDER BY k.poradi
  ", SECTIONS_TABLE, keys))
  out <- as_numeric_columns(out, c("poradi", "n", "celkem", "median", "q1", "q3"))
  out$iqr <- out$q3 - out$q1
  out$v_kostre <- out$celkem > 0 & out$n / out$celkem >= SKELETON_COVERAGE
  out <- out[skeleton_order(out), , drop = FALSE]
  out$kostra <- ifelse(out$v_kostre, cumsum(out$v_kostre), NA_integer_)
  rownames(out) <- NULL
  out
}

skeleton_members <- function(skeleton) skeleton[skeleton$v_kostre, , drop = FALSE]

order_consistency <- function(con, skeleton) {
  ranks <- skeleton_members(skeleton)
  table <- write_temp(con, SKELETON_TABLE, data.frame(
    sekce = ranks$sekce, poradi = as.integer(ranks$kostra), stringsAsFactors = FALSE
  ))
  row <- DBI::dbGetQuery(con, sprintf("
    WITH per_note AS (
      SELECT note_id, count(*) AS kolik, count(*) FILTER (WHERE poradi < predchozi) AS porusenych
      FROM (
        SELECT s.note_id, k.poradi,
          lag(k.poradi) OVER (PARTITION BY s.note_id ORDER BY s.position) AS predchozi
        FROM %s s JOIN %s k ON k.sekce = s.section
      )
      GROUP BY note_id
    )
    SELECT count(*) FILTER (WHERE kolik >= 2) AS posuzovano,
      count(*) FILTER (WHERE kolik >= 2 AND porusenych = 0) AS v_poradi
    FROM per_note
  ", SECTIONS_TABLE, table))
  list(posuzovano = as.numeric(row$posuzovano), v_poradi = as.numeric(row$v_poradi))
}

section_precedence <- function(con, skeleton) {
  ranks <- skeleton_members(skeleton)
  table <- write_temp(con, SKELETON_TABLE, data.frame(
    sekce = ranks$sekce, poradi = as.integer(ranks$kostra), stringsAsFactors = FALSE
  ))
  out <- DBI::dbGetQuery(con, sprintf("
    WITH pairs AS (
      SELECT a.sekce AS radek, b.sekce AS sloupec, a.poradi AS r, b.poradi AS s
      FROM %2$s a JOIN %2$s b ON a.sekce <> b.sekce
    ),
    seen AS (
      SELECT x.section AS radek, y.section AS sloupec,
        count(*) AS n_obou, count(*) FILTER (WHERE x.position < y.position) AS n_pred
      FROM %1$s x JOIN %1$s y ON x.note_id = y.note_id AND x.section <> y.section
      WHERE x.section IN (SELECT sekce FROM %2$s) AND y.section IN (SELECT sekce FROM %2$s)
      GROUP BY x.section, y.section
    )
    SELECT p.radek, p.sloupec, coalesce(s.n_obou, 0) AS n_obou, coalesce(s.n_pred, 0) AS n_pred
    FROM pairs p LEFT JOIN seen s ON s.radek = p.radek AND s.sloupec = p.sloupec
    ORDER BY p.r, p.s
  ", SECTIONS_TABLE, table))
  out <- as_numeric_columns(out, c("n_obou", "n_pred"))
  out$podil <- ifelse(out$n_obou > 0, out$n_pred / out$n_obou, NA_real_)
  out
}

skeleton_table <- function(skeleton) {
  data.frame(
    sekce = skeleton$sekce,
    label_cs = skeleton$label_cs,
    v_kostre = ifelse(skeleton$v_kostre, "ano", "ne"),
    pokryti = percent_text(skeleton$n, skeleton$celkem),
    median = ifelse(is.na(skeleton$median), "–", dua_decimal(skeleton$median)),
    iqr = ifelse(is.na(skeleton$iqr), "–", dua_decimal(skeleton$iqr)),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

skeleton_plot <- function(skeleton) {
  members <- skeleton_members(skeleton)
  members$popisek <- factor(members$label_cs, levels = rev(members$label_cs))
  ggplot2::ggplot(members, ggplot2::aes(median, popisek)) +
    ggplot2::geom_linerange(ggplot2::aes(xmin = q1, xmax = q3), orientation = "y") +
    ggplot2::geom_point() +
    ggplot2::labs(x = "Pozice mezi rozpoznanými sekcemi zprávy (medián a IQR)", y = NULL) +
    theme_thesis()
}

precedence_plot <- function(precedence, skeleton) {
  members <- skeleton_members(skeleton)
  label_of <- function(keys) members$label_cs[match(keys, members$sekce)]
  precedence$radek <- factor(label_of(precedence$radek), levels = rev(members$label_cs))
  precedence$sloupec <- factor(label_of(precedence$sloupec), levels = members$label_cs)
  ggplot2::ggplot(precedence, ggplot2::aes(sloupec, radek, fill = podil)) +
    ggplot2::geom_tile() +
    scale_fill_sequential(limits = c(0, 1), labels = percent_axis) +
    ggplot2::scale_x_discrete(drop = FALSE, position = "top") +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::labs(x = "Sekce následující", y = "Sekce předcházející", fill = "Podíl zpráv") +
    theme_thesis() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 0))
}

skeleton_text <- function(skeleton, consistency) {
  members <- skeleton_members(skeleton)
  if (nrow(skeleton) == 0L || skeleton$celkem[[1]] == 0) {
    return(paste("Kostru šablony", UNDECIDABLE_TEXT))
  }
  threshold <- coverage_threshold_text(SKELETON_COVERAGE)
  if (nrow(members) == 0L) {
    return(sprintf("Žádná sekce lexikonu nemá pokrytí aspoň %s, kostra šablony je prázdná.", threshold))
  }
  order_text <- if (consistency$posuzovano == 0) {
    "Pořadí nelze posoudit, protože žádná zpráva nemá aspoň dvě sekce kostry."
  } else {
    sprintf(
      "Pořadí kostry dodržuje %s ze %s s aspoň dvěma sekcemi kostry.",
      percent_label(consistency$v_poradi, consistency$posuzovano),
      count_phrase(consistency$posuzovano, NOTE_FORMS)
    )
  }
  sprintf(
    "Kostru šablony tvoří %s s pokrytím aspoň %s (podle mediánu pozice): %s. %s",
    count_phrase(nrow(members), SECTION_FORMS), threshold,
    join_labels(sprintf("`%s`", members$sekce)), order_text
  )
}

notes_section_counts_sql <- function(keys) {
  sprintf("
    SELECT s.note_id, count(k.sekce) AS pocet
    FROM %s s LEFT JOIN %s k ON k.sekce = s.section
    GROUP BY s.note_id
  ", SECTIONS_TABLE, keys)
}

sections_per_note <- function(con, lexicon) {
  keys <- write_temp(con, KEYS_TABLE, lexicon_keys(lexicon))
  found <- DBI::dbGetQuery(con, sprintf(
    "SELECT pocet, count(*) AS n FROM (%s) GROUP BY pocet", notes_section_counts_sql(keys)
  ))
  out <- data.frame(pocet = 0:nrow(lexicon_keys(lexicon)))
  out$n <- as.numeric(found$n[match(out$pocet, found$pocet)])
  out$n[is.na(out$n)] <- 0
  out$celkem <- sum(out$n)
  out
}

section_count_percentiles <- function(con, lexicon) {
  keys <- write_temp(con, KEYS_TABLE, lexicon_keys(lexicon))
  row <- DBI::dbGetQuery(con, sprintf("
    SELECT quantile_disc(pocet, 0.10) AS p10, quantile_disc(pocet, 0.50) AS p50,
      quantile_disc(pocet, 0.75) AS p75, quantile_disc(pocet, 0.95) AS p95, max(pocet) AS maximum
    FROM (%s)
  ", notes_section_counts_sql(keys)))
  unlist(lapply(row, as.numeric))
}

sections_per_note_plot <- function(counts) {
  bars <- data.frame(
    pocet = factor(counts$pocet, levels = counts$pocet),
    podil = ifelse(counts$celkem > 0, counts$n / counts$celkem, 0)
  )
  ggplot2::ggplot(bars, ggplot2::aes(pocet, podil)) +
    ggplot2::geom_col() +
    ggplot2::scale_y_continuous(labels = percent_axis) +
    ggplot2::labs(x = "Počet rozpoznaných sekcí lexikonu ve zprávě", y = "Podíl zpráv") +
    theme_thesis()
}

sections_per_note_text <- function(counts, percentiles) {
  if (sum(counts$n) == 0) {
    return(paste("Počet sekcí na zprávu", UNDECIDABLE_TEXT))
  }
  without <- counts$n[counts$pocet == 0]
  sprintf(
    "Počet rozpoznaných sekcí lexikonu na zprávu: medián %s (p10 %s, p75 %s, p95 %s, maximum %s). Žádnou sekci nemá %s (%s).",
    number_text(percentiles[["p50"]]), number_text(percentiles[["p10"]]),
    number_text(percentiles[["p75"]]), number_text(percentiles[["p95"]]),
    number_text(percentiles[["maximum"]]),
    count_phrase(without, NOTE_FORMS), percent_label(without, sum(counts$n))
  )
}

body_lengths <- function(con, skeleton, other) {
  found <- DBI::dbGetQuery(con, sprintf("
    SELECT section AS sekce, count(*) AS n,
      quantile_disc(body_chars, 0.5) AS p50, quantile_disc(body_chars, 0.75) AS p75,
      quantile_disc(body_chars, 0.95) AS p95
    FROM %s GROUP BY section
  ", SECTIONS_TABLE))
  found <- as_numeric_columns(found, c("n", "p50", "p75", "p95"))
  out <- data.frame(
    sekce = c(skeleton$sekce, other),
    label_cs = c(skeleton$label_cs, OTHER_LABEL),
    stringsAsFactors = FALSE
  )
  place <- match(out$sekce, found$sekce)
  out$n <- ifelse(is.na(place), 0, found$n[place])
  for (column in c("p50", "p75", "p95")) out[[column]] <- found[[column]][place]
  out
}

body_length_table <- function(lengths) {
  data.frame(
    sekce = lengths$sekce,
    label_cs = lengths$label_cs,
    n = number_text(lengths$n),
    p50 = number_text(lengths$p50),
    p75 = number_text(lengths$p75),
    p95 = number_text(lengths$p95),
    stringsAsFactors = FALSE
  )
}

# Délka zprávy je součet span_chars, protože round-trip parseru dává součet přes zprávu rovný
# délce textu a text se tak nečte podruhé.
strip_share <- function(con, lexicon) {
  if (!STRIP_SECTION %in% lexicon$sections$sekce) {
    stop(sprintf("Sekce %s pro strip D2 není v lexikonu.", STRIP_SECTION))
  }
  row <- DBI::dbGetQuery(con, sprintf("
    WITH per_note AS (
      SELECT note_id, sum(span_chars) AS delka,
        sum(span_chars) FILTER (WHERE section = %s) AS sekce
      FROM %s GROUP BY note_id
    )
    SELECT count(*) AS celkem, count(sekce) AS se_sekci,
      coalesce(sum(delka), 0) AS text_celkem, coalesce(sum(sekce), 0) AS text_sekce,
      quantile_cont(sekce::DOUBLE / delka, 0.5) AS median_podilu
    FROM per_note
  ", DBI::dbQuoteString(con, STRIP_SECTION), SECTIONS_TABLE))
  out <- lapply(row, as.numeric)
  out
}

strip_share_text <- function(strip) {
  if (strip$celkem == 0) {
    return(paste("Podíl textu odstraněný stripem D2", UNDECIDABLE_TEXT))
  }
  if (strip$se_sekci == 0) {
    return(sprintf(
      "Sekci `%s` má %s korpusu, strip D2 by nic neodstranil.",
      STRIP_SECTION, count_phrase(0, NOTE_FORMS)
    ))
  }
  sprintf(
    "Strip D2 odstraní sekci `%s` u %s korpusu (%s) a tvoří %s jeho textu; ve zprávě se sekcí tvoří v mediánu %s textu zprávy.",
    STRIP_SECTION, count_phrase(strip$se_sekci, NOTE_FORMS),
    percent_label(strip$se_sekci, strip$celkem),
    percent_label(strip$text_sekce, strip$text_celkem),
    percent_label(strip$median_podilu, 1)
  )
}

# Zpráva bez nadpisů má jedinou část, a tou je `other`; `other` se předává z modulu Pythonu,
# aby se klíč nerozešel s doménou.
note_lengths_sql <- function(con, other) {
  sprintf("
    SELECT note_id, sum(span_chars) AS delka,
      CASE WHEN bool_or(section <> %s) THEN 'se_sekcemi' ELSE 'bez_nadpisu' END AS skupina
    FROM %s GROUP BY note_id
  ", DBI::dbQuoteString(con, other), SECTIONS_TABLE)
}

headerless_notes <- function(con, other) {
  found <- DBI::dbGetQuery(con, sprintf("
    SELECT skupina, count(*) AS n,
      quantile_disc(delka, 0.5) AS p50, quantile_disc(delka, 0.95) AS p95
    FROM (%s) GROUP BY skupina
  ", note_lengths_sql(con, other)))
  found <- as_numeric_columns(found, c("n", "p50", "p95"))
  out <- data.frame(skupina = names(HEADERLESS_LABELS), stringsAsFactors = FALSE)
  place <- match(out$skupina, found$skupina)
  out$n <- ifelse(is.na(place), 0, found$n[place])
  out$celkem <- sum(out$n)
  out$p50 <- found$p50[place]
  out$p95 <- found$p95[place]
  out
}

headerless_histogram <- function(con, other) {
  out <- DBI::dbGetQuery(con, sprintf("
    SELECT skupina, %s AS kos, count(*) AS n
    FROM (%s) WHERE delka > 0
    GROUP BY skupina, kos ORDER BY skupina, kos
  ", log_bin_sql("delka"), note_lengths_sql(con, other)))
  as_numeric_columns(out, c("kos", "n"))
}

headerless_table <- function(headerless) {
  data.frame(
    skupina = HEADERLESS_LABELS[headerless$skupina],
    n = number_text(headerless$n),
    podil = percent_text(headerless$n, headerless$celkem),
    p50 = number_text(headerless$p50),
    p95 = number_text(headerless$p95),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

headerless_plot <- function(hist) {
  shares <- if (nrow(hist) == 0L) {
    data.frame(skupina = character(), dolni = numeric(), podil = numeric())
  } else {
    histogram_shares(hist, "skupina")
  }
  shares$skupina <- factor(shares$skupina, levels = names(HEADERLESS_LABELS), labels = HEADERLESS_LABELS)
  ggplot2::ggplot(shares, ggplot2::aes(dolni + 0.05, podil)) +
    ggplot2::geom_col(width = 0.1) +
    ggplot2::facet_wrap(ggplot2::vars(skupina), ncol = 1, drop = FALSE) +
    ggplot2::scale_x_continuous(
      breaks = 0:6, labels = function(x) format_number(10^x), limits = c(0, 6)
    ) +
    ggplot2::scale_y_continuous(labels = percent_axis) +
    ggplot2::labs(x = "Délka zprávy ve znacích (dolní mez koše)", y = "Podíl zpráv skupiny") +
    theme_thesis()
}

headerless_text <- function(headerless) {
  if (headerless$celkem[[1]] == 0) {
    return(paste("Zprávy bez nadpisů", UNDECIDABLE_TEXT))
  }
  without <- headerless[headerless$skupina == "bez_nadpisu", ]
  if (without$n == 0) {
    return("Každá zpráva korpusu má aspoň jednu sekci lexikonu.")
  }
  sectioned <- headerless[headerless$skupina == "se_sekcemi", ]
  length_phrase <- function(x) ifelse(is.na(x), "–", count_phrase(x, CHAR_FORMS))
  sprintf(
    "Bez nadpisů je %s korpusu (%s). Medián jejich délky je %s proti %s ve zprávách se sekcemi, p95 je %s proti %s.",
    count_phrase(without$n, NOTE_FORMS), percent_label(without$n, without$celkem),
    length_phrase(without$p50), length_phrase(sectioned$p50),
    length_phrase(without$p95), length_phrase(sectioned$p95)
  )
}

body_length_text <- function(lengths, other) {
  sections <- lengths[lengths$sekce != other & lengths$n > 0, , drop = FALSE]
  outside <- lengths[lengths$sekce == other, , drop = FALSE]
  if (nrow(sections) == 0L && outside$n == 0) {
    return(paste("Délky těl sekcí", UNDECIDABLE_TEXT))
  }
  outside_text <- if (outside$n == 0) {
    "Žádná zpráva nemá text mimo sekce."
  } else {
    sprintf("Text mimo sekce má medián %s.", count_phrase(outside$p50, CHAR_FORMS))
  }
  if (nrow(sections) == 0L) {
    return(outside_text)
  }
  longest <- sections[which.max(sections$p50), ]
  shortest <- sections[which.min(sections$p50), ]
  sprintf(
    "Nejdelší tělo má v mediánu sekce `%s` (%s), nejkratší sekce `%s` (%s). %s",
    longest$sekce, count_phrase(longest$p50, CHAR_FORMS),
    shortest$sekce, count_phrase(shortest$p50, CHAR_FORMS), outside_text
  )
}
