SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    SEMANTIC_FILTER('Is IMDb rating {0} at least 8.1?', CAST(m.IMDB_rating AS TEXT))
    AND SEMANTIC_FILTER('Is runtime {0} at most 180 minutes?', CAST(m.Runtime AS TEXT))
    AND SEMANTIC_FILTER('Is genre label {0} Drama, Crime, Action or Adventure?', CAST(m.Genre1 AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Does plot {0} describe betrayal, revenge or divided loyalties?',
    CAST(m.Plot AS TEXT)
  );
