LEXICON_PATH <- function() section_headers_path()
TEXT_WITH_SECTIONS <- "Name: ___\n\nAllergies:\nnone\n\nChief Complaint:\npain\n"

test_that("section_rows přijme dvě shodné zprávy pod různými note_id", {
  module <- sections_module()
  chunk <- list(
    note_id = c(k = "n1", k = "n2"),
    text = c(k = TEXT_WITH_SECTIONS, k = TEXT_WITH_SECTIONS)
  )
  out <- section_rows(module, LEXICON_PATH(), chunk)

  expect_equal(unique(out$note_id), c("n1", "n2"))
  expect_equal(nrow(out), 6L)
  expect_equal(out$section[1:3], c(module$OTHER, "allergies", "chief_complaint"))
})

test_that("section_rows zvládne dávku o jedné zprávě", {
  chunk <- list(note_id = "jediny", text = "Allergies:\nnone\n")
  out <- section_rows(sections_module(), LEXICON_PATH(), chunk)

  expect_equal(out$note_id, "jediny")
  expect_equal(out$section, "allergies")
  expect_equal(out$position, 0L)
})

test_that("section_rows vrátí pro prázdnou dávku typovaný data.frame s pěti sloupci", {
  chunk <- list(note_id = character(), text = character())
  out <- section_rows(sections_module(), LEXICON_PATH(), chunk)

  expect_equal(names(out), names(SECTION_COLUMNS))
  expect_equal(nrow(out), 0L)
  expect_type(out$note_id, "character")
  expect_type(out$position, "integer")
  expect_type(out$span_chars, "integer")
  expect_type(out$body_chars, "integer")
})

test_that("segment_corpus naplní tabulku jen zprávami vstupního korpusu", {
  texts <- c("Allergies:\nnone\n", "Chief Complaint:\npain\n")
  notes <- text_notes(paste0(strrep("x", SHORT_LIMIT), "\n", texts))
  notes$hadm_id <- c("h1", "h2")
  icd <- data.frame(hadm_id = "h1", icd_code = "Z99", icd_version = "10", stringsAsFactors = FALSE)
  short <- text_notes("Allergies:\nnone\n", prefix = "krátká")
  short$hadm_id <- "h1"
  con <- mimic_con(discharge = rbind(notes, short), diagnoses_icd = icd)
  name <- segment_corpus(con, sections_module(), LEXICON_PATH())
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s ORDER BY note_id, position", name))

  expect_equal(name, SECTIONS_TABLE)
  expect_equal(unique(out$note_id), "n1")
  expect_equal(out$section, c("other", "allergies"))
  expect_equal(sum(out$span_chars), nchar(notes$text[[1]]))
})

test_that("segment_corpus nad prázdným korpusem vytvoří prázdnou typovanou tabulku", {
  con <- mimic_con()
  name <- segment_corpus(con, sections_module(), LEXICON_PATH())
  out <- DBI::dbGetQuery(con, sprintf("SELECT * FROM %s", name))

  expect_equal(names(out), names(SECTION_COLUMNS))
  expect_equal(nrow(out), 0L)
})

mini_lexicon <- function(keys = c("a", "b", "c", "d")) {
  list(
    sections = data.frame(
      sekce = keys, label_cs = toupper(keys), synonymum = paste(keys, "x"),
      stringsAsFactors = FALSE
    ),
    preamble = "Name"
  )
}

# Zprávy jako pojmenovaný seznam sekcí v pořadí textu; délky těl a hlavičky se dají zadat zvlášť.
sections_frame <- function(notes, span = 10L, body = 5L) {
  rows <- lapply(names(notes), function(id) {
    sections <- notes[[id]]
    data.frame(
      note_id = id, position = seq_along(sections) - 1L, section = sections,
      span_chars = rep_len(as.integer(span), length(sections)),
      body_chars = rep_len(as.integer(body), length(sections)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

sections_con <- function(notes, env = parent.frame(), ...) {
  con <- mimic_con(env = env)
  frame <- if (length(notes) == 0L) {
    data.frame(
      note_id = character(), position = integer(), section = character(),
      span_chars = integer(), body_chars = integer(), stringsAsFactors = FALSE
    )
  } else {
    sections_frame(notes, ...)
  }
  DBI::dbWriteTable(con, SECTIONS_TABLE, frame, temporary = TRUE, overwrite = TRUE)
  con
}

ORDERED_NOTES <- list(
  n1 = c("other", "a", "b", "c"),
  n2 = c("a", "b", "c"),
  n3 = c("a", "c", "b"),
  n4 = c("b", "a"),
  n5 = c("a"),
  n6 = c("other", "a", "b", "d")
)

test_that("kostra: pokrytí, medián a IQR pozice mezi rozpoznanými sekcemi bez other", {
  con <- sections_con(ORDERED_NOTES)
  skeleton <- section_skeleton(con, mini_lexicon())
  row <- function(key) skeleton[skeleton$sekce == key, ]

  expect_equal(nrow(skeleton), 4L)
  expect_equal(row("a")$n, 6)
  expect_equal(row("a")$celkem, 6)
  expect_equal(c(row("a")$median, row("a")$iqr), c(0, 0))
  expect_equal(c(row("b")$median, row("b")$iqr), c(1, 0))
  expect_equal(c(row("c")$median, row("c")$iqr), c(2, 0.5))
})

test_that("kostra: pořadí podle mediánu, sekce pod pokrytím zůstane v tabulce mimo kostru", {
  skeleton <- section_skeleton(sections_con(ORDERED_NOTES), mini_lexicon())

  expect_equal(skeleton$sekce, c("a", "b", "c", "d"))
  expect_equal(skeleton$v_kostre, c(TRUE, TRUE, TRUE, FALSE))
  expect_equal(skeleton$kostra, c(1, 2, 3, NA))
})

test_that("kostra: shodný medián i pokrytí rozhodne pořadí v lexikonu", {
  tie <- list(n1 = c("a", "b"), n2 = c("b", "a"))

  expect_equal(section_skeleton(sections_con(tie), mini_lexicon(c("a", "b")))$sekce, c("a", "b"))
  expect_equal(section_skeleton(sections_con(tie), mini_lexicon(c("b", "a")))$sekce, c("b", "a"))
})

test_that("kostra: shodný medián se rozhodne sestupným pokrytím", {
  notes <- list(n1 = c("a", "b"), n2 = c("b", "a"), n3 = c("b"), n4 = c("a"))
    skeleton <- section_skeleton(sections_con(notes), mini_lexicon(c("b", "a")))
  expect_equal(skeleton$sekce, c("b", "a"))
  notes$n5 <- "a"
  skeleton <- section_skeleton(sections_con(notes), mini_lexicon(c("b", "a")))
  expect_equal(skeleton$sekce, c("a", "b"))
})

test_that("kostra: other pozice neovlivní", {
  with_other <- list(n1 = c("other", "a", "b"), n2 = c("other", "a", "b"))
  without <- list(n1 = c("a", "b"), n2 = c("a", "b"))
  lexicon <- mini_lexicon(c("a", "b"))

  expect_equal(
    section_skeleton(sections_con(with_other), lexicon),
    section_skeleton(sections_con(without), lexicon)
  )
})

test_that("konzistence: prohozené sekce kostry snižují podíl, jediná sekce kostry se nepočítá", {
  con <- sections_con(ORDERED_NOTES)
  skeleton <- section_skeleton(con, mini_lexicon())
  consistency <- order_consistency(con, skeleton)

  expect_equal(consistency$posuzovano, 5)
  expect_equal(consistency$v_poradi, 3)
})

test_that("precedence: podíly pro známé dvojice, součet dvojice je 1", {
  con <- sections_con(ORDERED_NOTES)
  skeleton <- section_skeleton(con, mini_lexicon())
  precedence <- section_precedence(con, skeleton)
  cell <- function(row, column) precedence[precedence$radek == row & precedence$sloupec == column, ]

  expect_equal(nrow(precedence), 6L)
  expect_equal(cell("a", "b")$podil, 0.8)
  expect_equal(cell("a", "b")$n_obou, 5)
  expect_equal(cell("a", "c")$podil, 1)
  expect_equal(cell("b", "c")$podil, 2 / 3)
  expect_equal(cell("b", "a")$podil + cell("a", "b")$podil, 1)
  expect_equal(cell("c", "b")$podil + cell("b", "c")$podil, 1)
  expect_false(any(precedence$radek == precedence$sloupec))
})

test_that("precedence: dvojice bez společné zprávy je NA", {
  con <- sections_con(list(n1 = "a", n2 = "a", n3 = "b", n4 = "b"))
  skeleton <- section_skeleton(con, mini_lexicon(c("a", "b")))
  precedence <- section_precedence(con, skeleton)

  expect_equal(nrow(precedence), 2L)
  expect_true(all(is.na(precedence$podil)))
  expect_equal(precedence$n_obou, c(0, 0))
})

test_that("kostra nad prázdnou tabulkou: žádná sekce v kostře a věta nelze posoudit", {
  con <- sections_con(list())
  skeleton <- section_skeleton(con, mini_lexicon())
  consistency <- order_consistency(con, skeleton)
  precedence <- section_precedence(con, skeleton)

  expect_equal(nrow(skeleton), 4L)
  expect_false(any(skeleton$v_kostre))
  expect_equal(consistency, list(posuzovano = 0, v_poradi = 0))
  expect_equal(nrow(precedence), 0L)
  expect_match(skeleton_text(skeleton, consistency), "nelze posoudit")
  expect_equal(nrow(skeleton_table(skeleton)), 4L)
})

test_that("skeleton_text skloňuje počet sekcí a jmenuje podíl pořadí", {
  one <- list(n1 = "a", n2 = "a")
  con <- sections_con(one)
  skeleton <- section_skeleton(con, mini_lexicon(c("a", "b")))
  text <- skeleton_text(skeleton, order_consistency(con, skeleton))
  expect_match(text, paste0("1", NBSP, "sekce"))
  expect_match(text, "nelze posoudit")

  con <- sections_con(ORDERED_NOTES)
  skeleton <- section_skeleton(con, mini_lexicon())
  text <- skeleton_text(skeleton, order_consistency(con, skeleton))
  expect_match(text, paste0("3", NBSP, "sekce"))
  expect_match(text, paste0("60,0", NBSP, "%"))
  expect_match(text, paste0("5", NBSP, "zpráv"))

  five <- paste0("s", 1:5)
  notes <- setNames(list(five, five), c("n1", "n2"))
  con <- sections_con(notes)
  skeleton <- section_skeleton(con, mini_lexicon(five))
  expect_match(skeleton_text(skeleton, order_consistency(con, skeleton)), paste0("5", NBSP, "sekcí"))
})

test_that("skeleton_text jmenuje prázdnou kostru, když žádná sekce nedosáhne pokrytí", {
  notes <- list(n1 = "a", n2 = "b", n3 = "c")
  con <- sections_con(notes)
  skeleton <- section_skeleton(con, mini_lexicon(c("a", "b", "c")))

  expect_match(skeleton_text(skeleton, order_consistency(con, skeleton)), "prázdná")
})

test_that("grafy kostry: osy v pořadí kostry, dlaždice NA je bílá, prázdná tabulka také projde", {
  disjoint <- list(
    n1 = c("a", "b"), n2 = c("a", "b"), n3 = c("a", "b"),
    n4 = c("d", "b"), n5 = c("d", "b"), n6 = c("d", "b")
  )
  con <- sections_con(disjoint)
  skeleton <- section_skeleton(con, mini_lexicon())
  order <- toupper(skeleton$sekce[skeleton$v_kostre])
  expect_equal(length(order), 3L)

  precedence <- section_precedence(con, skeleton)
  built <- ggplot2::ggplot_build(precedence_plot(precedence, skeleton))
  scale_x <- built$layout$panel_scales_x[[1]]
  scale_y <- built$layout$panel_scales_y[[1]]
  expect_equal(scale_x$get_limits(), order)
  expect_equal(scale_y$get_limits(), rev(order))
  no_common <- precedence$podil[is.na(precedence$podil)]
  expect_gt(length(no_common), 0L)
  expect_true("white" %in% built$data[[1]]$fill)

  points <- ggplot2::ggplot_build(skeleton_plot(skeleton))
  expect_equal(points$layout$panel_scales_y[[1]]$get_limits(), rev(order))

  empty <- sections_con(list())
  skeleton <- section_skeleton(empty, mini_lexicon())
  expect_no_error(ggplot2::ggplot_build(skeleton_plot(skeleton)))
  expect_no_error(ggplot2::ggplot_build(precedence_plot(section_precedence(empty, skeleton), skeleton)))
})

test_that("skeleton_table projde export_table s rozvržením sešitu, i nad prázdnou tabulkou", {
  for (notes in list(ORDERED_NOTES, list())) {
    con <- sections_con(notes)
    table <- skeleton_table(section_skeleton(con, mini_lexicon()))
    dir <- withr::local_tempdir()
    export_table(
      table,
      col.names = c("Sekce", "Popisek", "V kostře", "Pokrytí (%)", "Medián pozice", "IQR pozice"),
      align = "llcrrr",
      code = "sekce",
      label = "tbl-kostra",
      widths = list(24, 30, 10, 12, 12, 12),
      dir = dir
    )

    expect_true(file.exists(file.path(dir, "tbl-kostra.tex")))
  }
})
