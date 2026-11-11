SELECT
  ur.rowid AS review_id,
  p.rowid AS app_id
FROM raw_user_reviews AS ur
JOIN raw_playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'TOOLS'
  AND SEMANTIC_FILTER(
    'Does review {0} report a technical malfunction?',
    CAST(ur.Translated_Review AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Is sentiment polarity {0} between -0.25 and 0.25?',
    CAST(ur.Sentiment_Polarity AS TEXT)
  )
  AND SEMANTIC_FILTER(
    'Does review {0} mention a device, Android version or performance problem?',
    CAST(ur.Translated_Review AS TEXT)
  );
