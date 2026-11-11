SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    SEMANTIC_FILTER('Is release year {0} before 2010?', CAST(m.Year AS TEXT))
    AND SEMANTIC_FILTER('Is runtime {0} between 90 and 180 minutes?', CAST(m.Runtime AS TEXT))
    AND SEMANTIC_FILTER('Is genre label {0} Drama, Crime, Action or Adventure?', CAST(m.Genre1 AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Does plot {0} describe a character challenging the law or political authority?',
    CAST(m.Plot AS TEXT)
  );
