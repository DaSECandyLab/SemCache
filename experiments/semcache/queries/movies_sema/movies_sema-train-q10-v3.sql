SELECT
  m.movie_id
FROM raw_movie AS m
WHERE
  SEMANTIC_FILTER('Does overview {0} describe a criminal investigation?', CAST(m.overview AS TEXT))
  AND SEMANTIC_FILTER(
    'Does overview {0} describe a person missing, threatened or in danger?',
    CAST(m.overview AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does overview {0} involve police, a court or criminal activity?',
    CAST(m.overview AS TEXT)
  )
  AND SEMANTIC_FILTER('Is vote average {0} at least 6?', CAST(m.vote_average AS TEXT));
