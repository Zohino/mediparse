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
SECTION_FORMS <- c("sekce", "sekce", "sekcí")
NOTE_FORMS <- c("zpráva", "zprávy", "zpráv")

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
  ", SECTIONS_TABLE, keys))
  out <- as_numeric_columns(out, c("poradi", "n", "celkem", "median", "q1", "q3"))
  out$iqr <- out$q3 - out$q1
  out$v_kostre <- out$celkem > 0 & out$n / out$celkem >= SKELETON_COVERAGE
  placed <- order(!out$v_kostre, out$median, -out$n, out$poradi)
  out <- out[placed, , drop = FALSE]
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
