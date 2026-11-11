SELECT
  m.movie_id
FROM raw_movie AS m
WHERE
  SEMANTIC_FILTER(
    'Does overview {0} involve work, money or social ambition? Title: {1}.',
    CAST(m.overview AS TEXT),
    CAST(m.title AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does overview {0} involve family, friends or partners? Title: {1}.',
    CAST(m.overview AS TEXT),
    CAST(m.title AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does overview {0} describe a romantic relationship? Title: {1}.',
    CAST(m.overview AS TEXT),
    CAST(m.title AS TEXT)
  );
