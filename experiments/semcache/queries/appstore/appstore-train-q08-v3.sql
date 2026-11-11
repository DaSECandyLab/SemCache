SELECT
  ur.rowid AS review_id,
  p.rowid AS app_id
FROM raw_user_reviews AS ur
JOIN raw_playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'GAME'
  AND SEMANTIC_FILTER(
    'Does review {0} discuss graphics or visual style?',
    CAST(ur.Translated_Review AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Is sentiment polarity {0} between 0.1 and 0.5?',
    CAST(ur.Sentiment_Polarity AS TEXT)
  )
  AND SEMANTIC_FILTER('Does review {0} mention ads or paid content?', CAST(ur.Translated_Review AS TEXT))
  AND SEMANTIC_FILTER(
    'Does review {0} recommend the game and give a specific reason?',
    CAST(ur.Translated_Review AS TEXT)
  );
