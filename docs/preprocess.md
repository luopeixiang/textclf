本文件根据[../textclf/config/preprocess.py](../textclf/config/preprocess.py)自动生成

### PreprocessConfig



PreprocessConfig有以下属性：

 | Attribute name   | Type   | Default     | Description                                        |
|------------------|--------|-------------|----------------------------------------------------|
| train_file       | str    | "train.csv" |                                                    |
| valid_file       | str    | "valid.csv" |                                                    |
| test_file        | str    | "test.csv"  |                                                    |
| datadir          | str    | "dataset"   |                                                    |
| tokenizer        | str    | "char"      | Possible choices: space, char, nltk, jiaba         |
| nwords           | int    | -1          | number of words to retain, -1 表示不限制字典的大小 |
| min_word_count   | int    | 1           | min word count in dict                             |

