SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    (
      SEMANTIC_FILTER('Is Metascore {0} at least 70?', CAST(m.Metascore AS TEXT))
      AND SEMANTIC_FILTER('Is genre label {0} Drama, Crime, Action or Adventure?', CAST(m.Genre1 AS TEXT))
    )
    AND SEMANTIC_FILTER('Does plot {0} describe mystery, crime or danger?', CAST(m.Plot AS TEXT))
  )
  AND SEMANTIC_FILTER('Does plot {0} follow a protagonist facing a moral dilemma?', CAST(m.Plot AS TEXT));
