from typing import List, Tuple, Optional, Union

from .base import ConfigBase


class VectorizerConfig(ConfigBase):
    # Convert all characters to lowercase or not
    lowercase: bool = True
    # If ‘english’, a built-in stop word list for English is used.
    # If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
    # If None, no stop words will be used.
    stop_words: Union[List[str], str, None] = None
    # Regular expression denoting what constitutes a “token”
    token_pattern: str = r"(?u)\b\w+\b"
    # The lower and upper boundary of the range of n-values for different word n-grams to be extracted.
    ngram_range: Tuple = (1, 1)
    # When building the vocabulary ignore terms that have
    # a document frequency  strictly higher than the
    # given threshold (corpus-specific stop words). If float, the parameter represents
    # a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    max_df: Union[float, int] = 1.0
    # When building the vocabulary ignore terms that have a document frequency strictly
    # lower than the given threshold. This value is also called cut-off in the literature.
    # If float, the parameter represents a proportion of documents, integer absolute counts.
    # This parameter is ignored if vocabulary is not None.
    min_df: Union[float, int] = 1
    # If not None, build a vocabulary that only consider the top max_features
    # ordered by term frequency across the corpus.
    max_features: int = None


class CountVectorizerConfig(VectorizerConfig):
    """参考https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html"""
    # If True, all non zero counts are set to 1.
    # This is useful for discrete probabilistic models
    # that model binary events rather than integer counts.
    binary: bool = False


class TfidfVectorizerConfig(VectorizerConfig):
    """参考https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"""
    # Each output row will have unit norm, either: * ‘l2’: Sum of squares of vector elements is 1.
    # The cosine similarity between two vectors is their dot product when l2 norm has been applied.
    # * ‘l1’: Sum of absolute values of vector elements is 1. See preprocessing.normalize.
    norm: str = "l2"

    # Enable inverse-document-frequency reweighting.
    use_idf: bool = True

    # Smooth idf weights by adding one to document frequencies,
    # as if an extra document was seen containing every term in
    # the collection exactly once. Prevents zero divisions.
    smooth_idf: bool = True

    # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    sublinear_tf: bool = False
