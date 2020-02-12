import os
from os.path import join

import numpy as np
import joblib

from .base_trainer import Trainer
from textclf.utils.raw_data import load_raw_data
from textclf.utils.create import create_instance
from textclf.utils.config import get_instance_name


class MLTrainer(Trainer):
    """Main class for machine learning training."""

    def __init__(self, config):
        super().__init__(config)
        self.model = create_instance(self.config.model, asdict=True)
        self.model_name = get_instance_name(self.config.model)

    def prepare_data(self):
        """prepare data for model input"""
        raw_data = load_raw_data(self.config.raw_data_path)
        train_text, train_label = zip(*raw_data.train_pairs)
        valid_text, valid_label = zip(*raw_data.valid_pairs)
        test_text, test_label = zip(*raw_data.test_pairs)

        print(f"Building vectorizer...")
        self.vectorizer = create_instance(self.config.vectorizer, asdict=True)
        train_input = self.vectorizer.fit_transform(train_text)
        valid_input = self.vectorizer.transform(valid_text)
        test_input = self.vectorizer.transform(test_text)

        return (
            train_input, train_label,
            valid_input, valid_label,
            test_input, test_label
        )

    def train(self):
        train_input, train_label, _, _, test_input, test_label = self.prepare_data()

        print("Train Config:")
        print(self.config)
        print("Training...")
        self.model.fit(train_input, train_label)
        predicted = self.model.predict(test_input)
        acc = np.mean(predicted == np.array(test_label))
        print(f"Acc in test dataset: {acc*100}%")
        self.save()

    def save(self):
        """save model and vectorizer"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        model_path = join(self.config.save_dir, "model.joblib")
        print(f"Saving model to {model_path}...")
        joblib.dump(self.model, model_path)

        vec_path = join(self.config.save_dir, "vectorizer.joblib")
        print(f"Saving vectorizer to {vec_path}...")
        joblib.dump(self.vectorizer, vec_path)
