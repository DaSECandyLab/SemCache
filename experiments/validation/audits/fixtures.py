"""Controlled SQL fixtures and native semantic oracles; no real model requests."""

from __future__ import annotations

import json
import re
from collections import Counter

from experiments.core.paths import ROOT
from experiments.core.sql import identifier, literal

TABLES = {
    "appstore": ["user_reviews", "playstore"],
    "estate": ["nirvana_estate"],
    "movies": ["nirvana_movies"],
    "tpch": [
        "part",
        "supplier",
        "customer",
        "orders",
        "lineitem",
        "partsupp",
        "nation",
        "region",
    ],
    "movies_sema": [
        "movie",
        "movie_genres",
        "genre",
        "movie_languages",
        "language",
        "production_country",
        "country",
    ],
}


def fixture_rows():
    """Hand-chosen data, including strict boundaries, duplicate matches and NULLs."""
    rows = {
        "user_reviews": [
            {"App": "A", "Translated_Review": "Useful tool and fun for kids."},
            {"App": "A", "Translated_Review": "DROP review"},
            {"App": None, "Translated_Review": "Useful."},
            {"App": "B", "Translated_Review": "A good family game."},
            {"App": "A", "Translated_Review": None},
        ],
        "playstore": [
            {"App": "A", "Category": "TOOLS"},
            {"App": "A", "Category": "TOOLS"},
            {"App": "A", "Category": "GAME"},
            {"App": "A", "Category": "FAMILY"},
            {"App": "B", "Category": "GAME"},
            {"App": "B", "Category": "FAMILY"},
            {"App": None, "Category": "TOOLS"},
        ],
        "part": [
            {
                "p_partkey": 1,
                "p_name": "Retail part",
                "p_type": "BRASS",
                "p_brand": "Brand#12",
                "p_size": 10,
                "p_container": "SM BOX",
            },
            {
                "p_partkey": 2,
                "p_name": "DROP part",
                "p_type": "BRASS",
                "p_brand": "Brand#12",
                "p_size": 10,
                "p_container": "SM BOX",
            },
        ],
        "supplier": [
            {
                "s_suppkey": i,
                "s_name": name,
                "s_acctbal": 1000,
                "s_nationkey": 0,
                "s_comment": "dependable",
            }
            for i, name in [(1, "Supplier A"), (2, "DROP supplier")]
        ],
        "customer": [
            {
                "c_custkey": i,
                "c_name": name,
                "c_acctbal": 1000,
                "c_mktsegment": "BUILDING",
                "c_nationkey": 0,
            }
            for i, name in [(1, "Business A"), (2, "DROP customer"), (3, "Business A")]
        ],
        "orders": [
            {
                "o_orderkey": i,
                "o_custkey": i,
                "o_orderdate": "1996-02-01",
                "o_orderpriority": "1-URGENT",
                "o_orderstatus": "O",
                "o_totalprice": 150000,
            }
            for i in [1, 2, 3]
        ],
        "lineitem": [
            {
                "l_orderkey": i,
                "l_linenumber": 1,
                "l_partkey": p,
                "l_suppkey": s,
                "l_quantity": 20,
                "l_discount": 0.06,
                "l_shipmode": "AIR",
                "l_shipinstruct": "DELIVER IN PERSON",
                "l_shipdate": "1996-03-01",
            }
            for i, p, s in [(1, 1, 1), (2, 2, 1), (3, 1, 2)]
        ],
        "partsupp": [
            {"ps_partkey": p, "ps_suppkey": s, "ps_supplycost": 100, "ps_availqty": 100}
            for p, s in [(1, 1), (2, 1), (1, 2)]
        ],
        "nation": [{"n_nationkey": 0, "n_regionkey": 0, "n_name": "FRANCE"}],
        "region": [{"r_regionkey": 0, "r_name": "EUROPE"}],
    }
    rows["nirvana_estate"] = [
        {
            "Unnamed: 0": i,
            "Title": title,
            "Details": f"Price={price}; Amenities={amenities};",
            "Location": location,
        }
        for i, title, price, amenities, location in [
            (1, "4 bedroom detached duplex", "100M", "pool gym", "Lekki, Lagos"),
            (2, "5 bedroom detached duplex", "200M", "pool gym", "Lekki, Lagos"),
            (3, "3 bedroom detached duplex", "80M", "pool gym", "Lekki, Lagos"),
            (4, "6 bedroom detached duplex", "400M", "pool gym", "Ikoyi"),
            (
                5,
                "4 bedroom detached duplex",
                "₦100M for 4 bedrooms",
                "pool gym",
                "Lekki, Lagos",
            ),
            (6, "4 bedroom bungalow", "Not specified", "gym", "Lekki, Lagos"),
        ]
    ] + [{"Unnamed: 0": 7, "Title": None, "Details": None, "Location": None}]
    rows["nirvana_movies"] = [
        {
            "id": str(i),
            "Title": title,
            "Plot": f"Genre={genre}; Character={character};",
            "IMDB_rating": rating,
            "Runtime": runtime,
        }
        for i, title, genre, character, rating, runtime in [
            (1, "A", "crime", "Alice", 8.5, "120 min"),
            (2, "B", "crime", "Bob", 8.6, "140 min"),
            (3, "B", "crime", "Bob", 8.6, "140 min"),
            (4, "C", "crime", "Carol", 8.9, "160 min"),
            (5, "D", "crime", "Dan", 9.0, "180 min"),
            (6, "E", "crime", "Eve", 9.1, "2h 10min"),
            (7, "F", "comedy", "Frank", 9.2, "100 min"),
            (8, "G", "crime", "Grace", 9.3, None),
        ]
    ] + [{"id": "9", "Title": "H", "Plot": None, "IMDB_rating": None, "Runtime": None}]
    return rows


def create_fixture(con, dataset):
    if dataset == "movies_sema":
        source = ROOT / "benchmarks/datasets/movies/data/sema/movies_4.duckdb"
        con.execute(f"ATTACH {literal(str(source))} AS source_movies (READ_ONLY)")
        rows = {
            "movie": [
                {
                    "movie_id": i,
                    "overview": text,
                    "budget": 100,
                    "revenue": 200,
                    "movie_status": "Released",
                    "popularity": 20,
                    "release_date": "2010-01-01",
                    "vote_average": 8,
                }
                for i, text in [
                    (1, "background=modern;plot=crime;conflict=Society;"),
                    (1, "background=modern;plot=crime;conflict=Society;"),
                    (2, "DROP overview"),
                    (3, None),
                ]
            ],
            "genre": [
                {"genre_id": 1, "genre_name": "Comedy"},
                {"genre_id": 2, "genre_name": "Western"},
            ],
            "movie_genres": [
                {"movie_id": 1, "genre_id": 1},
                {"movie_id": 1, "genre_id": 2},
                {"movie_id": 2, "genre_id": 1},
                {"movie_id": None, "genre_id": 1},
            ],
            "language": [{"language_id": 1, "language_name": "English"}],
            "movie_languages": [
                {"movie_id": 1, "language_id": 1},
                {"movie_id": 1, "language_id": 1},
                {"movie_id": 2, "language_id": 1},
            ],
            "country": [{"country_id": 1, "country_iso_code": "US"}],
            "production_country": [
                {"movie_id": 1, "country_id": 1},
                {"movie_id": 1, "country_id": 1},
                {"movie_id": 2, "country_id": 1},
            ],
        }
        schemas = {}
        for table in TABLES[dataset]:
            schema = con.execute(
                f"DESCRIBE source_movies.{identifier(table)}"
            ).fetchall()
            schemas[table] = [(c[0], c[1]) for c in schema]
            con.execute(
                f"CREATE TABLE {identifier(table)} AS SELECT * FROM source_movies.{identifier(table)} WHERE FALSE"
            )
            con.executemany(
                f"INSERT INTO {identifier(table)} VALUES ("
                + ",".join("?" for _ in schema)
                + ")",
                [[r.get(c[0]) for c in schema] for r in rows[table]],
            )
        con.execute("DETACH source_movies")
        return schemas
    if dataset == "appstore":
        con.execute(
            f"ATTACH {literal(str(ROOT / 'benchmarks/datasets/appstore/data/sema/appstore.duckdb'))} AS source_appstore (READ_ONLY)"
        )
    data = fixture_rows()
    schemas = {}
    for table in TABLES[dataset]:
        if dataset == "appstore":
            source = "source_appstore." + identifier(table)
        elif dataset == "estate":
            source = f"(SELECT * EXCLUDE(image) FROM read_parquet({literal(str(ROOT / 'benchmarks/datasets/estate/data/nirvana/multimodal_real_estate.parquet'))}))"
        elif dataset == "movies":
            source = f"read_csv_auto({literal(str(ROOT / 'benchmarks/datasets/movies/data/nirvana/movie_data.csv'))})"
        else:
            source = f"read_parquet({literal(str(ROOT / 'benchmarks/datasets/tpch/data/plop' / (table + '.parquet')))})"
        schema = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        schemas[table] = [(c[0], c[1]) for c in schema]
        con.execute(
            f"CREATE TABLE {identifier(table)} AS SELECT * FROM {source} WHERE FALSE"
        )

        def default(kind):
            if any(k in kind for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT")):
                return 0
            if kind == "DATE":
                return "1996-01-01"
            return "fixture"

        records = [
            [row.get(name, default(kind)) for name, kind in schemas[table]]
            for row in data[table]
        ]
        con.executemany(
            f"INSERT INTO {identifier(table)} VALUES ("
            + ",".join("?" for _ in schema)
            + ")",
            records,
        )
    if dataset == "appstore":
        con.execute("DETACH source_appstore")
    return schemas


def project_value(task, value, field):
    if value is None:
        return None
    m = re.search(re.escape(field) + r"=([^;]*);", str(value), flags=re.I)
    return m[1] if m else None


def filter_value(task, values):
    task = task.split("\nInputs:", 1)[0].lower()
    if any(v is None or "DROP" in str(v) for v in values):
        return False
    value = str(values[0]) if values else ""
    if "bedrooms" in task:
        match = re.search(r"(\d+) bedroom", value)
        return bool(match) and (
            int(match[1]) > 3 if "more than 3" in task else int(match[1]) < 6
        )
    if "detached duplex" in task:
        return "detached duplex" in value
    if "swimming pool" in task:
        return "pool" in value
    if "gym" in task:
        return "gym" in value
    if "lekki" in task:
        return value == "Lekki, Lagos"
    if "rating" in task:
        rating = float(value)
        return (
            rating > 8.5
            if "higher than 8.5" in task
            else rating < 9
            if "lower than 9" in task
            else rating > 9
        )
    if "crime" in task:
        return "crime" in value
    return True


class Oracle:
    def __init__(self, mode="mixed"):
        self.mode = mode

    def __call__(self, kind, task, rows, fields):
        if kind == "projection":
            return [
                {
                    f["name"]: None
                    if self.mode == "null_projection"
                    else project_value(task, row[0], f["name"])
                    for f in fields
                }
                for row in rows
            ]
        return [
            False if self.mode == "empty" else filter_value(task, row) for row in rows
        ]


def fixed_arity(function, count):
    """DuckDB Python UDF registration requires explicit positional parameters."""
    args = ",".join(f"a{i}" for i in range(count))
    namespace = {"target": function}
    exec(f"def invoke({args}):\n    return target({args})", namespace)
    return namespace["invoke"]


def payload(cursor):
    return {
        "columns": [[c[0], str(c[1])] for c in cursor.description],
        "rows": cursor.fetchall(),
    }


def bag(value):
    return Counter(
        json.dumps(list(r), ensure_ascii=False, default=str) for r in value["rows"]
    )
