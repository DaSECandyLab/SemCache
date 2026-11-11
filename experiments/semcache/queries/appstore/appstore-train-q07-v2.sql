SELECT
  ur.rowid AS review_id,
  p.rowid AS app_id
FROM raw_user_reviews AS ur
JOIN raw_playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'PRODUCTIVITY'
  AND SEMANTIC_FILTER(
    'Does review {0} discuss ads, subscriptions or payments?',
    CAST(ur.Translated_Review AS TEXT)
  )
  AND SEMANTIC_FILTER('Is sentiment polarity {0} above 0.1?', CAST(ur.Sentiment_Polarity AS TEXT))
  AND SEMANTIC_FILTER(
    'Does review {0} describe a drawback or request an improvement?',
    CAST(ur.Translated_Review AS TEXT)
  );
