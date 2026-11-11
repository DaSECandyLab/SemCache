SELECT
  ur.rowid AS id
FROM user_reviews AS ur
JOIN playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'GAME'
  AND SEMANTIC_FILTER(
    '{0} is a complete sentence and has informative content.',
    CAST(Translated_review AS TEXT)
  )
  AND SEMANTIC_FILTER('{0} is a positive user review.', CAST(Translated_review AS TEXT))
  AND SEMANTIC_FILTER(
    '{0} shows that users have a good experience and fun with this app',
    CAST(Translated_Review AS TEXT)
  );
