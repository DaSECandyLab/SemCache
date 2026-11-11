SELECT
  MAX(
    CASE
      WHEN ISFINITE(TRY_CAST("IMDB_rating" AS DOUBLE))
      THEN TRY_CAST("IMDB_rating" AS DOUBLE)
      ELSE NULL
    END
  ) AS result
FROM (
  SELECT
    *
  FROM (
    SELECT
      *
    FROM (
      SELECT
        *
      FROM (
        SELECT
          *
          EXCLUDE (__mapped),
          __mapped."Genre" AS "Genre"
        FROM (
          SELECT
            *,
            SEMANTIC_PROJECT(
              'According to the movie plot, extract the genre(s) of each movie.
Inputs: Plot: {0}',
              CAST("Plot" AS TEXT),
              CAST(NULL AS STRUCT("Genre" TEXT))
            ) AS __mapped
          FROM (
            SELECT
              *
            FROM nirvana_movies
          ) AS stage_0
        ) AS project_1
      ) AS stage_1
      WHERE
        SEMANTIC_FILTER(
          'The rating is higher than 8.5.
Inputs: IMDB_rating: {0}',
          CAST("IMDB_rating" AS TEXT)
        )
    ) AS stage_2
    WHERE
      SEMANTIC_FILTER(
        'The rating is lower than 9.
Inputs: IMDB_rating: {0}',
        CAST("IMDB_rating" AS TEXT)
      )
  ) AS stage_3
  WHERE
    SEMANTIC_FILTER(
      'The movie belongs to crime movies.
Inputs: Genre: {0}',
      CAST("Genre" AS TEXT)
    )
) AS stage_4;
