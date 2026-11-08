local_clear_blocking_variables <- function(env = parent.frame()) {
  names <- vapply(data_access_rules(), `[[`, character(1), "name")
  withr::local_envvar(setNames(rep(NA_character_, length(names)), names), .local_envir = env)
}
