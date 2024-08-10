from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys
import numpy as np

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.preprocessing import MinMaxScaler
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.linear_model import LogisticRegression

classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables= config.NUM_FEATURES)),
        ('ModeImputation', pp.ModeImputer(variables= config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify = config.FEATURE_TO_MODIFY, variable_to_add = config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop= config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables= config.CAT_FEATURES)),
        ('LogTransforms', pp.LogTransforms(variables= config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)