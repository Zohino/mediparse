text_mentions_path <- function() config_path("text_mentions.json")

read_text_mentions <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  list(
    negation = list(
      cues = as.character(unlist(raw$negation$cues)),
      window_chars = as.integer(raw$negation$window_chars)
    ),
    symptoms = list(keywords_en = as.character(unlist(raw$symptoms$keywords_en)))
  )
}

MENTION_FLOOR <- 0.5
SELECTION_SIZE <- 5L
FLAGS_TABLE <- "mention_flags_input"

phrase_escape <- function(x) {
  gsub(" ", "\\s+", re2_escape(x), fixed = TRUE)
}

re2_escape <- function(x) {
  gsub("([.+*?(){}|^$\\\\\\[\\]])", "\\\\\\1", x, perl = TRUE)
}

mention_pattern <- function(keywords, acronyms) {
  alternatives <- c(
    if (length(keywords) > 0L) sprintf("(?i:%s)", paste(phrase_escape(keywords), collapse = "|")),
    re2_escape(acronyms)
  )
  if (length(alternatives) == 0L) {
    stop("Zmínka potřebuje aspoň jedno klíčové slovo nebo zkratku.")
  }
  sprintf("\\b(?:%s)\\b", paste(alternatives, collapse = "|"))
}

negated_pattern <- function(cues, window, mention) {
  sprintf(
    "(?i:\\b(?:%s)\\b)[^.;\\n]{0,%d}%s",
    paste(phrase_escape(cues), collapse = "|"), as.integer(window), mention
  )
}

mention_terms <- function(candidates, mentions) {
  window <- mentions$negation$window_chars
  cues <- mentions$negation$cues
  term <- function(column, mention) {
    list(
      column = column, mention = mention,
      negated = negated_pattern(cues, window, mention)
    )
  }
  candidate_terms <- lapply(seq_along(candidates), function(index) {
    entry <- candidates[[index]]
    term(
      sprintf("k%03d", index), mention_pattern(entry$keywords_en, entry$acronyms_en)
    )
  })
  symptom_terms <- lapply(seq_along(mentions$symptoms$keywords_en), function(index) {
    term(
      sprintf("p%03d", index),
      mention_pattern(mentions$symptoms$keywords_en[[index]], character())
    )
  })
  c(candidate_terms, symptom_terms)
}

mention_flags <- function(con, candidates, mentions) {
  if (FLAGS_TABLE %in% DBI::dbListTables(con)) {
    return(FLAGS_TABLE)
  }
  quote <- function(x) DBI::dbQuoteString(con, x)
  terms <- mention_terms(candidates, mentions)
  found <- vapply(terms, function(term) {
    sprintf("coalesce(regexp_matches(d.text, %s), false) AS %s_m", quote(term$mention), term$column)
  }, character(1))
  asserted <- vapply(terms, function(term) {
    sprintf(
      "CASE WHEN %s_m THEN regexp_matches(regexp_replace(text, %s, '', 'g'), %s) ELSE false END AS %s_t",
      term$column, quote(term$negated), quote(term$mention), term$column
    )
  }, character(1))
  kept <- unlist(lapply(terms, function(term) paste0(term$column, c("_m", "_t"))))
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE %s AS
    WITH corpus AS (%s),
    found AS (
      SELECT c.note_id, d.text, %s
      FROM corpus c JOIN discharge d ON d.note_id = c.note_id
    )
    SELECT note_id, %s, %s FROM found
  ", FLAGS_TABLE, input_corpus_sql(con), paste(found, collapse = ", "),
    paste(sprintf("%s_m", vapply(terms, `[[`, character(1), "column")), collapse = ", "),
    paste(asserted, collapse = ", ")))
  FLAGS_TABLE
}

count_flags <- function(column, within = NULL) {
  scope <- if (is.null(within)) "" else sprintf(" AND %s", within)
  list(
    zminka = sprintf("count(*) FILTER (WHERE %s_m%s)", column, scope),
    tvrzeni = sprintf("count(*) FILTER (WHERE %s_t%s)", column, scope)
  )
}

mention_agreement <- function(con, candidates, mentions) {
  flags <- mention_flags(con, candidates, mentions)
  notes <- candidate_notes(con, candidates)
  terms <- mention_terms(candidates, mentions)
  selects <- vapply(seq_along(candidates), function(index) {
    key <- DBI::dbQuoteString(con, names(candidates)[[index]])
    column <- terms[[index]]$column
    coded <- sprintf("note_id IN (SELECT note_id FROM %s WHERE kandidat = %s)", notes, key)
    all <- count_flags(column)
    coded_counts <- count_flags(column, coded)
    sprintf("
      SELECT %s AS kandidat,
        (SELECT count(DISTINCT note_id) FROM %s WHERE kandidat = %s) AS icd,
        %s AS zminka, %s AS tvrzeni, %s AS icd_zminka, %s AS icd_tvrzeni,
        count(*) AS celkem
      FROM %s
    ", key, notes, key, all$zminka, all$tvrzeni, coded_counts$zminka, coded_counts$tvrzeni, flags)
  }, character(1))
  out <- DBI::dbGetQuery(con, paste(selects, collapse = " UNION ALL "))
  out[names(out) != "kandidat"] <- lapply(out[names(out) != "kandidat"], as.numeric)
  out$kandidat <- names(candidates)[match(out$kandidat, names(candidates))]
  out[match(names(candidates), out$kandidat), , drop = FALSE]
}

symptom_mentions <- function(con, candidates, mentions) {
  flags <- mention_flags(con, candidates, mentions)
  terms <- mention_terms(candidates, mentions)
  keywords <- mentions$symptoms$keywords_en
  selects <- vapply(seq_along(keywords), function(index) {
    counts <- count_flags(terms[[length(candidates) + index]]$column)
    sprintf(
      "SELECT %s AS priznak, %s AS zminka, %s AS tvrzeni, count(*) AS celkem FROM %s",
      DBI::dbQuoteString(con, keywords[[index]]), counts$zminka, counts$tvrzeni, flags
    )
  }, character(1))
  out <- DBI::dbGetQuery(con, paste(selects, collapse = " UNION ALL "))
  out[c("zminka", "tvrzeni", "celkem")] <- lapply(out[c("zminka", "tvrzeni", "celkem")], as.numeric)
  out[match(keywords, out$priznak), , drop = FALSE]
}

agreement_verdict <- function(share) {
  ifelse(
    is.na(share), "nelze",
    ifelse(share >= MENTION_FLOOR, "splňuje", "nesplňuje")
  )
}

rubric_result <- function(band, agreement) {
  failed <- band %in% c("pod", "nad") | agreement == "nesplňuje"
  unknown <- band == "nelze" | agreement == "nelze"
  ifelse(unknown, "nelze", ifelse(failed, "neprošel", "prošel"))
}

rubric <- function(prevalence, agreement, candidates) {
  keys <- names(candidates)
  at_prevalence <- match(keys, prevalence$kandidat)
  at_agreement <- match(keys, agreement$kandidat)
  n <- as.numeric(prevalence$n[at_prevalence])
  total <- as.numeric(prevalence$celkem[at_prevalence])
  share <- agreement$icd_zminka[at_agreement] / agreement$icd[at_agreement]
  share[!is.finite(share)] <- NA_real_
  band <- band_verdict(n / total)
  verdict <- agreement_verdict(share)
  data.frame(
    kandidat = keys,
    kapitola = vapply(candidates, function(entry) entry$kapitola, character(1)),
    podil = n / total,
    pasmo = band,
    shoda = share,
    verdikt_shody = verdict,
    vysledek = rubric_result(band, verdict),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

chapter_winners <- function(passed) {
  chapters <- unique(passed$kapitola)
  vapply(chapters, function(chapter) {
    inside <- passed[passed$kapitola == chapter, , drop = FALSE]
    inside$kandidat[which.max(inside$shoda)]
  }, character(1), USE.NAMES = FALSE)
}

select_five <- function(rubric) {
  keys <- function(result) rubric$kandidat[rubric$vysledek == result]
  if (nrow(rubric) == 0L || all(rubric$vysledek == "nelze")) {
    return(list(kind = "nelze"))
  }
  passed <- rubric[rubric$vysledek == "prošel", , drop = FALSE]
  winners <- chapter_winners(passed)
  if (length(winners) > SELECTION_SIZE) {
    stop("Pool má víc kapitol ICD-10, než kolik míst má pětice.")
  }
  rest <- passed[!passed$kandidat %in% winners, , drop = FALSE]
  rest <- rest[order(-rest$shoda, match(rest$kandidat, rubric$kandidat)), , drop = FALSE]
  filled <- utils::head(rest$kandidat, max(0L, SELECTION_SIZE - length(winners)))
  list(
    kind = "vyber",
    selected = rubric$kandidat[rubric$kandidat %in% c(winners, filled)],
    filled = rubric$kandidat[rubric$kandidat %in% filled],
    passed = keys("prošel"),
    failed = keys("neprošel"),
    unknown = keys("nelze")
  )
}

rubric_outcome <- function(rubric, selection) {
  outcome <- unname(c(
    "prošel" = "prošel", "neprošel" = "neprošel", "nelze" = "nelze posoudit"
  )[rubric$vysledek])
  outcome[rubric$kandidat %in% selection$selected] <- "vybrán"
  outcome[rubric$kandidat %in% selection$filled] <- "vybrán doplněním"
  outcome
}

rubric_table <- function(rubric, selection, candidates) {
  outcome <- if (selection$kind == "vyber") {
    rubric_outcome(rubric, selection)
  } else {
    rep("nelze posoudit", nrow(rubric))
  }
  data.frame(
    kandidat = candidate_labels(rubric$kandidat, candidates),
    kapitola = rubric$kapitola,
    pasmo = unname(BAND_LABELS[rubric$pasmo]),
    shoda = percent_text(rubric$shoda, 1),
    vysledek = outcome,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

agreement_table <- function(agreement, candidates) {
  data.frame(
    kandidat = candidate_labels(agreement$kandidat, candidates),
    icd = number_text(agreement$icd),
    zminka_icd = percent_text(agreement$icd_zminka, agreement$icd),
    tvrzeni_icd = percent_text(agreement$icd_tvrzeni, agreement$icd),
    icd_zminka = percent_text(agreement$icd_zminka, agreement$zminka),
    icd_tvrzeni = percent_text(agreement$icd_tvrzeni, agreement$tvrzeni),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

agreement_points <- function(agreement, candidates) {
  labels <- candidate_labels(names(candidates), candidates)
  metrics <- c(zminka = "P(zmínka | ICD+)", tvrzeni = "P(tvrzení | ICD+)")
  data.frame(
    kandidat = factor(
      rep(candidate_labels(agreement$kandidat, candidates), 2L),
      levels = rev(labels)
    ),
    metrika = factor(rep(unname(metrics), each = nrow(agreement)), levels = unname(metrics)),
    podil = c(agreement$icd_zminka / agreement$icd, agreement$icd_tvrzeni / agreement$icd),
    stringsAsFactors = FALSE
  )
}

symptom_table <- function(symptoms) {
  data.frame(
    priznak = symptoms$priznak,
    zminka = number_text(symptoms$zminka),
    podil = percent_text(symptoms$zminka, symptoms$celkem),
    bez_negace = percent_text(symptoms$tvrzeni, symptoms$celkem),
    stringsAsFactors = FALSE
  )
}

symptom_bars <- function(symptoms) {
  shown <- symptoms[order(-symptoms$zminka, symptoms$priznak), , drop = FALSE]
  data.frame(
    priznak = factor(shown$priznak, levels = rev(shown$priznak)),
    podil = shown$zminka / shown$celkem,
    bez_negace = shown$tvrzeni / shown$celkem,
    stringsAsFactors = FALSE
  )
}

labelled_keys <- function(keys, candidates) {
  join_labels(sprintf("%s (`%s`)", candidate_labels(keys, candidates), keys))
}

selection_text <- function(selection, candidates) {
  if (selection$kind == "nelze") {
    return(paste(
      "Výběr pětice nelze posoudit: u žádného kandidáta není rubrika definovaná."
    ))
  }
  if (length(selection$selected) == 0L) {
    return("Žádný kandidát rubriku nesplnil, pětici nelze vybrat a rozhodne následující úloha.")
  }
  parts <- sprintf(
    "Rubrika vybrala %s: %s.",
    count_phrase(length(selection$selected), c("diagnózu", "diagnózy", "diagnóz")),
    labelled_keys(selection$selected, candidates)
  )
  if (length(selection$filled) > 0L) {
    parts <- c(parts, sprintf(
      "Doplněním prázdných kapitol vznikla část výběru: %s.", labelled_keys(selection$filled, candidates)
    ))
  }
  if (length(selection$selected) < SELECTION_SIZE) {
    parts <- c(parts, "Pětice není úplná, o zbytku rozhodne následující úloha.")
  }
  paste(parts, collapse = " ")
}

mention_floor_text <- function() {
  paste0(round(100 * MENTION_FLOOR), NBSP, "%")
}

noise_text <- function(agreement, candidates) {
  share <- agreement$icd_zminka / agreement$zminka
  known <- !is.na(share)
  if (!any(known)) {
    return("Riziko šumu v negativní třídě nelze posoudit: žádný kandidát nemá zprávu se zmínkou.")
  }
  low <- agreement$kandidat[known & share < MENTION_FLOOR]
  if (length(low) == 0L) {
    return(sprintf(
      "Žádný kandidát nemá P(ICD+ | zmínka) pod %s, riziko šumu v negativní třídě je tedy nízké.",
      mention_floor_text()
    ))
  }
  sprintf(
    "Riziko šumu v negativní třídě se týká %s s P(ICD+ | zmínka) pod %s: %s. %s",
    count_phrase(length(low), c("kandidáta", "kandidátů", "kandidátů")), mention_floor_text(),
    join_labels(candidate_labels(low, candidates)),
    "U diagnóz s nízkou hodnotou nemusí být zpráva bez kódu, ale se zmínkou, opravdu negativní."
  )
}

symptom_text <- function(symptoms) {
  if (nrow(symptoms) == 0L) {
    return("V lexikonu nejsou žádné příznaky.")
  }
  top <- symptoms[which.max(symptoms$zminka), ]
  sprintf(
    "Nejčastější je %s, zmínku má %s zpráv korpusu.",
    top$priznak, percent_label(top$zminka, top$celkem)
  )
}

EXPORT_DIGITS <- 4L

share_or_na <- function(numerator, denominator) {
  value <- round(numerator / denominator, EXPORT_DIGITS)
  ifelse(is.finite(value), value, NA_real_)
}

synthetic_parameters <- function(selection, prevalence, agreement, cells, candidates) {
  keys <- if (selection$kind == "vyber") names(candidates)[names(candidates) %in% selection$selected] else character()
  named <- function(values) {
    if (length(keys) == 0L) {
      return(structure(list(), names = character()))
    }
    stats::setNames(as.list(values), keys)
  }
  at_prevalence <- match(keys, prevalence$kandidat)
  at_agreement <- match(keys, agreement$kandidat)
  conditional <- lapply(keys, function(row) {
    inside <- cells[cells$radek == row, , drop = FALSE]
    shares <- vapply(keys, function(column) {
      if (column == row) return(1)
      at <- inside$sloupec == column
      if (sum(at) == 1L) share_or_na(inside$n_ij[at], inside$n_i[at]) else NA_real_
    }, numeric(1))
    stats::setNames(as.list(unname(shares)), keys)
  })
  mentions <- lapply(seq_along(keys), function(index) {
    row <- agreement[at_agreement[[index]], ]
    entry <- candidates[[keys[[index]]]]
    list(
      given_positive = share_or_na(row$icd_zminka, row$icd),
      given_negative = share_or_na(row$zminka - row$icd_zminka, row$celkem - row$icd),
      keywords = I(c(entry$keywords_en, entry$acronyms_en))
    )
  })
  list(
    selection = I(keys),
    prevalence = named(share_or_na(prevalence$n[at_prevalence], prevalence$celkem[at_prevalence])),
    conditional = named(conditional),
    mentions = named(mentions)
  )
}
