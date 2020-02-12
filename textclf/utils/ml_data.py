from .raw_data import load_raw_data


def prepare_ml_input(joblib_path, VectorizerConfig):
    raw_data = load_raw_data(joblib_path)
