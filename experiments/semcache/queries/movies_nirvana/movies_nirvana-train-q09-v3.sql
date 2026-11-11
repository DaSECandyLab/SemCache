SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    SEMANTIC_FILTER('Is release year {0} after 1960?', CAST(m.Year AS TEXT))
    AND SEMANTIC_FILTER('Is Metascore {0} at least 65?', CAST(m.Metascore AS TEXT))
    AND SEMANTIC_FILTER('Is runtime {0} at least 90 minutes?', CAST(m.Runtime AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Does plot {0} describe an investigation, pursuit or escape?',
    CAST(m.Plot AS TEXT)
  );
