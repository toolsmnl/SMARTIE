"""Custom exception hierarchy for the brute_force package.

All errors the pipeline can raise descend from BruteForceError so callers
can catch the whole family with one except clause when needed.
"""


class BruteForceError(Exception):
    """Base class for all brute_force errors."""


class DataLoadError(BruteForceError):
    """A replicate or targets file could not be loaded or parsed."""


class FeatureExtractionError(BruteForceError):
    """Feature computation failed for a dataset."""


class ConfigError(BruteForceError):
    """Invalid CLI arguments or datasets-file content."""


class ResultStoreError(BruteForceError):
    """Writing or reading results from disk failed."""


class QueryError(BruteForceError):
    """The results index is missing or a query is malformed."""
