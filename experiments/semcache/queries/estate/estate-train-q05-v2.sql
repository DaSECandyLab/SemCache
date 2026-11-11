SELECT
  rowid AS id
FROM raw_nirvana_estate AS e
WHERE
  (
    SEMANTIC_FILTER('Does title {0} state three or four bedrooms?', CAST(e.Title AS TEXT))
    AND SEMANTIC_FILTER(
      'Is location {0} in Lekki Phase 1, Ikate, Osapa, Chevron or Agungi?',
      CAST(e.Location AS TEXT)
    )
  )
  AND SEMANTIC_FILTER('Does description {0} mention CCTV or gated security?', CAST(e.Details AS TEXT));
