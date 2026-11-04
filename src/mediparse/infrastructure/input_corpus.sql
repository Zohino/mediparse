CREATE OR REPLACE TEMP MACRO input_corpus_selection(min_chars) AS TABLE
WITH metrics AS (
    -- Otisk a délka vznikají před oknem: okno nad hash(text) by materializovalo
    -- celý text. PARTITION BY text spadl v sondě S23 na paměť.
    SELECT
        note_id,
        subject_id,
        hadm_id,
        hash(text) AS fingerprint,
        coalesce(length(text), 0) AS chars
    FROM discharge
),
base AS (
    SELECT
        note_id,
        subject_id,
        hadm_id,
        hadm_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM diagnoses_icd i WHERE i.hadm_id = metrics.hadm_id
        ) AS has_icd,
        chars >= min_chars AS long_enough,
        fingerprint
    FROM metrics
),
eligible AS (
    -- Filtry platí před deduplikací, aby kopie bez ICD nevyřadila platnou kopii;
    -- počty tak odpovídají sešitům EDA 5, 7 a 8.
    SELECT
        note_id,
        row_number() OVER (PARTITION BY fingerprint ORDER BY note_id) AS copy
    FROM base
    WHERE has_icd AND long_enough
)
SELECT b.note_id, b.subject_id, b.hadm_id, b.has_icd, e.copy
FROM base b
LEFT JOIN eligible e ON e.note_id = b.note_id;
