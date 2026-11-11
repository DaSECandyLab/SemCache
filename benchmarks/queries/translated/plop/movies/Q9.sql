SELECT MAX(CASE WHEN ISFINITE(TRY_CAST("IMDB_rating" AS DOUBLE)) THEN TRY_CAST("IMDB_rating" AS DOUBLE) ELSE NULL END) AS result FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * EXCLUDE (__mapped), __mapped."Genre" AS "Genre" FROM (SELECT *, STRUCT_PACK("Genre" := NULLIF(SEMANTIC_STRING('According to the movie plot, extract the genre(s) of each movie.
Inputs: Plot: {Plot}
Output protocol: If the requested value is missing, return exactly __SQL_NULL__. Otherwise return only the requested value. Preserve an empty string as empty.'), '__SQL_NULL__')) AS __mapped FROM (SELECT * FROM nirvana_movies) AS stage_0) AS project_1) AS stage_1 WHERE SEMANTIC('The rating is higher than 8.5.
Inputs: IMDB_rating: {IMDB_rating}')) AS stage_2 WHERE SEMANTIC('The rating is lower than 9.
Inputs: IMDB_rating: {IMDB_rating}')) AS stage_3 WHERE SEMANTIC('The movie belongs to crime movies.
Inputs: Genre: {Genre}')) AS stage_4;
