SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    SEMANTIC_FILTER('Is IMDb rating {0} at least 8.1?', CAST(m.IMDB_rating AS TEXT))
    AND SEMANTIC_FILTER('Is runtime {0} at most 150 minutes?', CAST(m.Runtime AS TEXT))
    AND SEMANTIC_FILTER('Is release year {0} before 2005?', CAST(m.Year AS TEXT))
  )
  AND SEMANTIC_FILTER('Is genre label {0} Action, Crime, Adventure or Comedy?', CAST(m.Genre1 AS TEXT));
