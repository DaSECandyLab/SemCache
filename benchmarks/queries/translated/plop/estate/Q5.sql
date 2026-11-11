SELECT * FROM (SELECT * FROM (SELECT * EXCLUDE (__mapped), __mapped."Amenities" AS "Amenities" FROM (SELECT *, STRUCT_PACK("Amenities" := NULLIF(SEMANTIC_STRING('Extract Amenities of the estate from the estate details.
Inputs: Details: {Details}
Output protocol: If the requested value is missing, return exactly __SQL_NULL__. Otherwise return only the requested value. Preserve an empty string as empty.'), '__SQL_NULL__')) AS __mapped FROM (SELECT * FROM (SELECT * FROM nirvana_estate) AS stage_0 WHERE SEMANTIC('Whether the estate has more than 3 bedrooms
Inputs: Title: {Title}')) AS stage_1) AS project_2) AS stage_2 WHERE SEMANTIC('Whether the estate has less than 6 bedrooms.
Inputs: Title: {Title}')) AS stage_3;
