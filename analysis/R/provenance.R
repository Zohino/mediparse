write_sidecar <- function(output, input, manifest) {
  path <- paste0(output, ".meta.json")
  provenance <- list(
    input = list(path = input, sha256 = unname(tools::sha256sum(input))),
    run_manifest = list(
      path = manifest, sha256 = unname(tools::sha256sum(manifest)),
      row_id = jsonlite::fromJSON(manifest, simplifyVector = FALSE)$row_id
    )
  )
  jsonlite::write_json(provenance, path, auto_unbox = TRUE, pretty = TRUE)
  invisible(path)
}
