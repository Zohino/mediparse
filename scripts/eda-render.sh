# shellcheck shell=bash

eda_stage() {
    local root=$1 work=$2
    (cd "$root" && tar --create --exclude=eda/_freeze --exclude=eda/.quarto eda) | tar --extract --directory "$work"
}

eda_revision() {
    local root=$1
    MEDIPARSE_COMMIT=$(git -C "$root" rev-parse HEAD)
    if [[ -n "$(git -C "$root" status --porcelain)" ]]; then
        MEDIPARSE_DIRTY=true
    else
        MEDIPARSE_DIRTY=false
    fi
    export MEDIPARSE_COMMIT MEDIPARSE_DIRTY
}

eda_render() {
    local work=$1 number=$2 document stem
    if [[ -n "$number" ]]; then
        documents=("$work"/eda/"${number}"-*.qmd)
    else
        documents=("$work"/eda/*.qmd)
    fi
    [[ -f "${documents[0]}" ]] || {
        echo "Dokument eda/${number}-*.qmd neexistuje." >&2
        return 1
    }
    for document in "${documents[@]}"; do
        quarto render "$document"
    done
    for document in "${documents[@]}"; do
        stem=$(basename "$document" .qmd)
        [[ -f "$work/results/eda/$stem.pdf" && -d "$work/eda/_freeze/$stem" ]] || {
            echo "Render $stem nevytvořil PDF a _freeze. Quarto ve skryté cestě (například pod .worktrees/) ignoruje _quarto.yml; pracovní adresář ${work} nesmí mít složku s tečkou." >&2
            return 1
        }
    done
}
