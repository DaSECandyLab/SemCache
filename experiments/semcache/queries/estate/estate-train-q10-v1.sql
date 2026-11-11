SELECT
  rowid AS id
FROM raw_nirvana_estate AS e
WHERE
  (
    SEMANTIC_FILTER('Does title {0} state exactly 4 bedrooms?', CAST(e.Title AS TEXT))
    AND SEMANTIC_FILTER('Does title {0} describe a duplex or terraced house?', CAST(e.Title AS TEXT))
    AND SEMANTIC_FILTER('Is location {0} in Lekki or Ajah?', CAST(e.Location AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Is the single asking price in {0} in NGN (N/₦/naira) and below 300 million? Other currencies, ranges or missing prices: false.',
    CAST(e.Details AS TEXT)
  );
