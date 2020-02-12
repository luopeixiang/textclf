
import joblib

from .base_tester import Tester
from textclf.utils.raw_data import create_tokenizer


class MLTester(Tester):
    """负责machine learning model的测试"""

    def __init__(self, config):
        super().__init__(config)
        # load model and vectorizer
        self.model = joblib.load(self.config.model_path)
        self.vectorizer = joblib.load(self.config.vec_path)
        self.tokenizer = create_tokenizer(self.config.tokenizer)

    def _preprocess(self, text):
        """process raw text for model input"""
        text_processed = " ".join(self.tokenizer(text))
        text_vec = self.vectorizer.transform([text_processed])
        return text_vec

    def predict_label(self, text):
        text_vec = self._preprocess(text)
        label = str(self.model.predict(text_vec)[0])
        return label

    def predict_prob(self, text):
        text_vec = self._preprocess(text)
        prob = list(self.model.predict_proba(text_vec)[0])
        return prob

    def get_all_labels(self):
        return list(self.model.classes_)
