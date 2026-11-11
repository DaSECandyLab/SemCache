SELECT
  rowid AS id
FROM raw_nirvana_estate AS e
WHERE
  (
    SEMANTIC_FILTER('Does title {0} state two, three or four bedrooms?', CAST(e.Title AS TEXT))
    AND SEMANTIC_FILTER('Does description {0} explicitly mention a balcony?', CAST(e.Details AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Is the single asking price in {0} in NGN (N/₦/naira) and below 200 million? Other currencies, ranges or missing prices: false.',
    CAST(e.Details AS TEXT)
  );
