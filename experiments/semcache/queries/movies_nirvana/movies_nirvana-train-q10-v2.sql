SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    SEMANTIC_FILTER(
      'Is Rotten Tomatoes rating {0} at least 85 percent?',
      CAST(m."Rotten Tomatoes" AS TEXT)
    )
    AND SEMANTIC_FILTER('Is runtime {0} between 90 and 180 minutes?', CAST(m.Runtime AS TEXT))
    AND SEMANTIC_FILTER('Is release year {0} between 1950 and 2020?', CAST(m.Year AS TEXT))
  )
  AND SEMANTIC_FILTER('Does plot {0} describe a crime or investigation?', CAST(m.Plot AS TEXT));
