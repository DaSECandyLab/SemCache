SELECT m.movie_id FROM movie m
WHERE SEMANTIC('The {m.overview} suggests that the movie is an action movie.')
  AND SEMANTIC('The {m.overview} shows that the movie focuses on a central role.')
  AND SEMANTIC('The {m.overview} shows that the movie is based on historical events ');
