SELECT
  ur.rowid AS id
FROM user_reviews AS ur
JOIN playstore AS p
  ON ur.App = p.App
WHERE
  p.Category = 'TOOLS'
  AND SEMANTIC_FILTER(
    '{0} is a complete sentence and has informative content.',
    CAST(ur.Translated_Review AS TEXT)
  )
  AND SEMANTIC_FILTER('{0} is a positive user review.', CAST(ur.Translated_Review AS TEXT))
  AND SEMANTIC_FILTER('{0} shows that this app is a useful tool.', CAST(ur.Translated_Review AS TEXT));
