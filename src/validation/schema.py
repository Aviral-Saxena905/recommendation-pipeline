
import pandera.pandas as pa
from pandera.pandas import Column, Check

ratings_schema = pa.DataFrameSchema({
    "userId": Column(int, nullable=False),
    "movieId": Column(int, nullable=False),
    "rating": Column(
        float,
        Check.in_range(0.5, 5.0),
        nullable=False
    ),
    "timestamp": Column(int, nullable=False)
}, strict=True)

users_schema = pa.DataFrameSchema({
    "userId": Column(int, nullable=False, unique=True),
    "age": Column(int, Check.in_range(18, 64), nullable=False),
    "occupation": Column(str, nullable=False),
    "location": Column(str, nullable=False)
}, strict=True)


movies_schema = pa.DataFrameSchema({
    "movieId": Column(int, nullable=False, unique=True),
    "title": Column(str, nullable=False),
    "genres": Column(str, nullable=False)
}, strict=True)

tags_schema = pa.DataFrameSchema({
    "userId": Column(int, nullable=False),
    "movieId": Column(int, nullable=False),
    "tag": Column(str, nullable=False),
    "timestamp": Column(int, nullable=False)
}, strict=True)

links_schema = pa.DataFrameSchema({
    "movieId": Column(int, nullable=False, unique=True),
    "imdbId": Column(int, nullable=False),
    "tmdbId": Column(float, nullable=True)
}, strict=True)
