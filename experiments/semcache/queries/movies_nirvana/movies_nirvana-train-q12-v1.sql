SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    (
      SEMANTIC_FILTER('Is release year {0} after 1960?', CAST(m.Year AS TEXT))
      AND SEMANTIC_FILTER('Is IMDb rating {0} at least 8.1?', CAST(m.IMDB_rating AS TEXT))
    )
    AND SEMANTIC_FILTER(
      'Does plot {0} describe conflict with authority, crime or violence?',
      CAST(m.Plot AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Does plot {0} involve family, friends or personal relationships?',
    CAST(m.Plot AS TEXT)
  );
