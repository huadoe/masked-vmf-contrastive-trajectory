import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SignedMinMaxScaler(BaseEstimator, TransformerMixin):
    """自定义符号保持归一化（Signed Min-Max Normalization）"""
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.abs_max_ = None  # 存储每列的最大绝对值

    def fit(self, X, y=None):
        # 计算每列的最大绝对值
        self.abs_max_ = np.max(np.abs(X), axis=0)
        print(self.abs_max_)
        return self

    def transform(self, X):
        a, b = self.feature_range
        scale = (b - a) / 2
        normalized = np.sign(X) * (np.abs(X) / self.abs_max_) * scale

        # 根据输入类型决定输出
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(normalized, columns=X.columns)
        return np.array(normalized)

    def inverse_transform(self, X):
        a, b = self.feature_range
        scale = (b - a) / 2
        original = X * (self.abs_max_ / scale)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(original, columns=X.columns)
        return np.array(original)
        
