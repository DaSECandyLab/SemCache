SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    (
      SEMANTIC_FILTER('Is Metascore {0} at least 75?', CAST(m.Metascore AS TEXT))
      AND SEMANTIC_FILTER('Is genre label {0} Drama, Crime or Action?', CAST(m.Genre1 AS TEXT))
    )
    AND SEMANTIC_FILTER('Is certificate {0} rated R?', CAST(m.Rated AS TEXT))
  )
  AND SEMANTIC_FILTER('Is runtime {0} at least 100 minutes?', CAST(m.Runtime AS TEXT));
