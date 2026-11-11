SELECT
  rowid AS id
FROM raw_nirvana_movies AS m
WHERE
  (
    (
      SEMANTIC_FILTER(
        'Is Rotten Tomatoes rating {0} at least 85 percent?',
        CAST(m."Rotten Tomatoes" AS TEXT)
      )
      AND SEMANTIC_FILTER('Is release year {0} before 2010?', CAST(m.Year AS TEXT))
    )
    AND SEMANTIC_FILTER('Does plot {0} describe betrayal, revenge or conflict?', CAST(m.Plot AS TEXT))
  )
  AND SEMANTIC_FILTER('Does plot {0} take place mainly in a city?', CAST(m.Plot AS TEXT));
