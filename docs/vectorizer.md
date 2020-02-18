本文件根据[../textclf/config/vectorizer.py](../textclf/config/vectorizer.py)自动生成

### VectorizerConfig



VectorizerConfig有以下属性：

 | Attribute name   | Type                        | Default        | Description                                                                                                                                                                                                                                                                                                     |
|------------------|-----------------------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| lowercase        | bool                        | True           | Convert all characters to lowercase or not                                                                                                                                                                                                                                                                      |
| stop_words       | Union[List[str], str, None] | None           | If ‘english’, a built-in stop word list for English is used.If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.If None, no stop words will be used.                                                                                                  |
| token_pattern    | str                         | r"(?u)\b\w+\b" | Regular expression denoting what constitutes a “token”                                                                                                                                                                                                                                                          |
| ngram_range      | Tuple                       | (1, 1)         | The lower and upper boundary of the range of n-values for different word n-grams to be extracted.                                                                                                                                                                                                               |
| max_df           | Union[float, int]           | 1.0            | When building the vocabulary ignore terms that havea document frequency  strictly higher than thegiven threshold (corpus-specific stop words). If float, the parameter representsa proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.                       |
| min_df           | Union[float, int]           | 1              | When building the vocabulary ignore terms that have a document frequency strictlylower than the given threshold. This value is also called cut-off in the literature.If float, the parameter represents a proportion of documents, integer absolute counts.This parameter is ignored if vocabulary is not None. |
| max_features     | int                         | None           | If not None, build a vocabulary that only consider the top max_featuresordered by term frequency across the corpus.                                                                                                                                                                                             |



### CountVectorizerConfig

参考https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

CountVectorizerConfig继承VectorizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                                                                                    |
|------------------|--------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|
| binary           | bool   | False     | If True, all non zero counts are set to 1.This is useful for discrete probabilistic modelsthat model binary events rather than integer counts. |



### TfidfVectorizerConfig

参考https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

TfidfVectorizerConfig继承VectorizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                                                                                                                                                                                                                   |
|------------------|--------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| norm             | str    | "l2"      | Each output row will have unit norm, either: * ‘l2’: Sum of squares of vector elements is 1.The cosine similarity between two vectors is their dot product when l2 norm has been applied.* ‘l1’: Sum of absolute values of vector elements is 1. See preprocessing.normalize. |
| use_idf          | bool   | True      | Enable inverse-document-frequency reweighting.                                                                                                                                                                                                                                |
| smooth_idf       | bool   | True      | Smooth idf weights by adding one to document frequencies,as if an extra document was seen containing every term inthe collection exactly once. Prevents zero divisions.                                                                                                       |
| sublinear_tf     | bool   | False     | Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).                                                                                                                                                                                                                 |

