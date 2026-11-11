SELECT
  ur.rowid AS review_id,
  p.rowid AS app_id
FROM raw_user_reviews AS ur
JOIN raw_playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'GAME'
  AND SEMANTIC_FILTER(
    'Does review {0} report crashes, freezing or lag?',
    CAST(ur.Translated_Review AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Is sentiment polarity {0} between -0.2 and 0.2?',
    CAST(CAST(ur.Sentiment_Polarity AS DECIMAL(5, 3)) AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does review {0} discuss a level''s difficulty or a control problem?',
    CAST(ur.Translated_Review AS TEXT)
  );
