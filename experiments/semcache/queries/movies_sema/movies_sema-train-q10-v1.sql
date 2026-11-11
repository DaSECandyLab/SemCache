SELECT
  m.movie_id
FROM raw_movie AS m
WHERE
  SEMANTIC_FILTER(
    'Does overview {0} describe physical danger, combat or a pursuit?',
    CAST(m.overview AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does overview {0} involve a soldier, officer, investigator or criminal?',
    CAST(m.overview AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does overview {0} make a family relationship central to the plot?',
    CAST(m.overview AS TEXT)
  );
