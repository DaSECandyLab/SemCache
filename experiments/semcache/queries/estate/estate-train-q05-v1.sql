SELECT
  rowid AS id
FROM raw_nirvana_estate AS e
WHERE
  (
    SEMANTIC_FILTER('Does title {0} state exactly 4 bedrooms?', CAST(e.Title AS TEXT))
    AND SEMANTIC_FILTER(
      'Is location {0} in Ajah, Sangotedo, Osapa, Chevron or Agungi?',
      CAST(e.Location AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Does description {0} explicitly mention a fitted kitchen?',
    CAST(e.Details AS TEXT)
  );
