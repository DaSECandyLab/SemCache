SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      *
      EXCLUDE (__mapped),
      __mapped."Amenities" AS "Amenities"
    FROM (
      SELECT
        *,
        SEMANTIC_PROJECT(
          'Extract Amenities of the estate from the estate details.
Inputs: Details: {0}',
          CAST("Details" AS TEXT),
          CAST(NULL AS STRUCT("Amenities" TEXT))
        ) AS __mapped
      FROM (
        SELECT
          *
        FROM (
          SELECT
            *
          FROM nirvana_estate
        ) AS stage_0
        WHERE
          SEMANTIC_FILTER(
            'Whether the estate has more than 3 bedrooms
Inputs: Title: {0}',
            CAST("Title" AS TEXT)
          )
      ) AS stage_1
    ) AS project_2
  ) AS stage_2
  WHERE
    SEMANTIC_FILTER(
      'Whether the estate has less than 6 bedrooms.
Inputs: Title: {0}',
      CAST("Title" AS TEXT)
    )
) AS stage_3;
