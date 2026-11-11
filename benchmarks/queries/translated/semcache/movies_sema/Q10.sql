SELECT m.movie_id FROM movie m
WHERE SEMANTIC_FILTER('The {0} suggests that the movie is an action movie.', CAST(m.overview AS TEXT))
  AND SEMANTIC_FILTER('The {0} shows that the movie focuses on a central role.', CAST(m.overview AS TEXT))
  AND SEMANTIC_FILTER('The {0} shows that the movie is based on historical events ', CAST(m.overview AS TEXT));
