SELECT
  id
FROM (
  SELECT
    ur.rowid AS id,
    ur.Translated_Review
  FROM user_reviews AS ur
  JOIN playstore AS p
    ON ur.App = p.App
  WHERE
    p.Category = 'GAME'
) AS "__appstore_candidates"
WHERE
  SEMANTIC('{Translated_review} is a complete sentence and has informative content.')
  AND SEMANTIC('{Translated_review} is a positive user review.')
  AND SEMANTIC(
    '{Translated_Review} shows that users have a good experience and fun with this app'
  );
