"""
Yield Prediction Module

Orchestrates fruit detection history, weather data, and ML models
to generate reliable yield predictions for orchards.

Components:
- feature_aggregator: Aggregates detections and weather into feature vectors
- sampling_patterns: Extrapolates from samples to full orchard yield
- historical_baseline: Provides regional/historical baseline yields
- yield_predictor: ML model for yield estimation
"""

from .feature_aggregator import FeatureAggregator, DetectionAggregation, WeatherAggregation
from .sampling_patterns import SamplingPatternGenerator, SamplingResult
from .historical_baseline import HistoricalBaselineRegistry, YieldBaseline
from .yield_predictor import YieldPredictorModel, YieldPredictionResult

__all__ = [
    'FeatureAggregator',
    'DetectionAggregation',
    'WeatherAggregation',
    'SamplingPatternGenerator',
    'SamplingResult',
    'HistoricalBaselineRegistry',
    'YieldBaseline',
    'YieldPredictorModel',
    'YieldPredictionResult',
]
