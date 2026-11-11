SELECT
  rowid AS id
FROM raw_nirvana_estate AS e
WHERE
  (
    SEMANTIC_FILTER('Does description {0} mention CCTV or gated security?', CAST(e.Details AS TEXT))
    AND SEMANTIC_FILTER(
      'Is the single asking price in {0} in NGN (N/₦/naira) and between 100 and 500 million? Other currencies, ranges or missing prices: false.',
      CAST(e.Details AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Does description {0} explicitly mention a fitted kitchen?',
    CAST(e.Details AS TEXT)
  );
