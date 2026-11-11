SELECT
  MIN(
    CASE
      WHEN ISFINITE(TRY_CAST("Price" AS DOUBLE))
      THEN TRY_CAST("Price" AS DOUBLE)
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
          __mapped."Price" AS "Price"
        FROM (
          SELECT
            *,
            SEMANTIC_PROJECT(
              'Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.
Inputs: Details: {0}',
              CAST("Details" AS TEXT),
              CAST(NULL AS STRUCT("Price" DOUBLE))
            ) AS __mapped
          FROM (
            SELECT
              *
            FROM nirvana_estate
          ) AS stage_0
        ) AS project_1
      ) AS stage_1
      WHERE
        SEMANTIC_FILTER(
          'Whether the estate has more than 3 bedrooms.
Inputs: Title: {0}',
          CAST("Title" AS TEXT)
        )
    ) AS stage_2
    WHERE
      SEMANTIC_FILTER(
        'Whether the estate has less than 6 bedrooms.
Inputs: Title: {0}',
        CAST("Title" AS TEXT)
      )
  ) AS stage_3
  WHERE
    SEMANTIC_FILTER(
      'Whether the estate is a detached duplex.
Inputs: Title: {0}',
      CAST("Title" AS TEXT)
    )
) AS stage_4;
