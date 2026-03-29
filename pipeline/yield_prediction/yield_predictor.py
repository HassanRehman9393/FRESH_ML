"""
Yield Prediction: Core ML Model
Implements XGBoost-based yield prediction with Linear Regression fallback.
Generates predictions with confidence intervals and contributing factor analysis.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import joblib

logger = logging.getLogger(__name__)


@dataclass
class YieldPredictionResult:
    """Result of ML-based yield prediction"""
    predicted_yield_kg: float
    confidence: float
    confidence_lower_bound: float
    confidence_upper_bound: float
    contributing_factors: Dict[str, float]
    trend_direction: str
    model_used: str


class YieldPredictorModel:
    """
    ML-based yield prediction model.
    
    Uses XGBoost as primary model with Linear Regression fallback.
    Generates predictions with uncertainty quantification.
    """
    
    # Feature names in order
    FEATURE_NAMES = [
        'ripeness_ratio',
        'disease_percentage',
        'health_score',
        'average_confidence',
        'coverage_score',
        'temperature_norm',
        'rainfall_norm',
        'humidity_norm',
        'area_norm',
        'growth_stage_norm'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.xgb_model = None
        self.lr_model = None
        self.is_trained = False
        self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize linear regression fallback model"""
        try:
            from sklearn.linear_model import LinearRegression
            self.lr_model = LinearRegression()
            self.logger.info("Linear Regression fallback model initialized")
        except ImportError:
            self.logger.warning("scikit-learn not available for fallback model")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train XGBoost model on historical data.
        
        Args:
            X_train: Training features (N x 10 array)
            y_train: Training targets (N array) - yield in kg
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Dictionary with training metrics (RMSE, MAE, R²)
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0
            }
            
            # Train model
            train_data = xgb.DMatrix(X_train, label=y_train)
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(xgb.DMatrix(X_val, label=y_val), 'validation')]
            
            self.xgb_model = xgb.train(
                params,
                train_data,
                num_boost_round=100,
                evals=eval_set if eval_set else None,
                early_stopping_rounds=10 if eval_set else None,
                verbose_eval=False
            )
            
            self.is_trained = True
            
            # Calculate metrics
            y_pred_train = self.xgb_model.predict(train_data)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae = mean_absolute_error(y_train, y_pred_train)
            r2 = r2_score(y_train, y_pred_train)
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.xgb_model.predict(xgb.DMatrix(X_val, label=y_val))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                metrics['val_rmse'] = val_rmse
            
            self.logger.info(f"Model trained successfully: RMSE={rmse:.2f}, R²={r2:.3f}")
            return metrics
            
        except ImportError:
            self.logger.error("XGBoost not available, cannot train model")
            return {'error': 'XGBoost not installed'}
    
    def predict(
        self,
        features: Dict[str, float],
        extrapolated_fruit_count: int,
        orchard_area_hectares: float,
        fruit_type: str = 'default'
    ) -> YieldPredictionResult:
        """
        Predict yield using trained model.
        
        Args:
            features: Dictionary of normalized features [0-1]
            extrapolated_fruit_count: Total fruit count from sampling
            orchard_area_hectares: Orchard size
            fruit_type: Type of fruit
        
        Returns:
            YieldPredictionResult with prediction and confidence
        """
        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features)
            
            # Get prediction
            if self.is_trained and self.xgb_model is not None:
                prediction, confidence = self._predict_xgb(
                    feature_vector,
                    extrapolated_fruit_count,
                    orchard_area_hectares
                )
                model_used = 'xgboost'
            else:
                prediction, confidence = self._predict_baseline(
                    features,
                    extrapolated_fruit_count,
                    orchard_area_hectares,
                    fruit_type
                )
                model_used = 'baseline'
            
            # Calculate confidence intervals
            lower_bound, upper_bound = self._calculate_intervals(
                prediction,
                confidence
            )
            
            # Analyze contributing factors
            factors = self._analyze_factors(features)
            trend = self._determine_trend(features)
            
            return YieldPredictionResult(
                predicted_yield_kg=prediction,
                confidence=confidence,
                confidence_lower_bound=lower_bound,
                confidence_upper_bound=upper_bound,
                contributing_factors=factors,
                trend_direction=trend,
                model_used=model_used
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return safe default
            return self._fallback_prediction(extrapolated_fruit_count, orchard_area_hectares)
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare feature vector from features dictionary.
        
        Args:
            features: Dictionary of normalized features
        
        Returns:
            Numpy array of shape (1, 10) in correct order
        """
        feature_vector = np.zeros((1, len(self.FEATURE_NAMES)))
        
        for i, feature_name in enumerate(self.FEATURE_NAMES):
            feature_vector[0, i] = features.get(feature_name, 0.0)
        
        return feature_vector
    
    def _predict_xgb(
        self,
        feature_vector: np.ndarray,
        extrapolated_fruit_count: int,
        orchard_area_hectares: float
    ) -> Tuple[float, float]:
        """
        Make prediction using XGBoost model.
        
        Args:
            feature_vector: Prepared feature array (1, 10)
            extrapolated_fruit_count: Fruit count from sampling
            orchard_area_hectares: Orchard area
        
        Returns:
            Tuple of (prediction_kg, confidence)
        """
        import xgboost as xgb
        
        dmatrix = xgb.DMatrix(feature_vector)
        base_prediction = float(self.xgb_model.predict(dmatrix)[0])
        
        # Blend with direct estimate from fruit count
        direct_estimate = extrapolated_fruit_count * 0.25  # Rough avg fruit weight
        direct_estimate *= (1 + feature_vector[0, 1])  # Adjust by ripeness
        direct_estimate *= (1 - feature_vector[0, 10] * 0.3)  # Reduce by disease
        
        # Weighted blend: 70% ML model, 30% direct estimate
        blended_prediction = (base_prediction * 0.7) + (direct_estimate * 0.3)
        
        # Confidence based on model agreement
        if base_prediction > 0:
            agreement = min(abs(direct_estimate / base_prediction), 1.5)
            confidence = 0.5 + (0.5 * min(agreement / 1.5, 1.0))
        else:
            confidence = 0.7
        
        return blended_prediction, confidence
    
    def _predict_baseline(
        self,
        features: Dict[str, float],
        extrapolated_fruit_count: int,
        orchard_area_hectares: float,
        fruit_type: str
    ) -> Tuple[float, float]:
        """
        Make prediction using baseline/direct estimation.
        
        Used when XGBoost not trained or unavailable.
        
        Args:
            features: Normalized features
            extrapolated_fruit_count: Fruit count
            orchard_area_hectares: Orchard area
            fruit_type: Type of fruit
        
        Returns:
            Tuple of (prediction_kg, confidence)
        """
        # Base yield from fruit count
        fruit_weights = {
            'mango': 0.25,
            'orange': 0.18,
            'guava': 0.18,
            'grapefruit': 0.35,
            'default': 0.25
        }
        
        fruit_weight = fruit_weights.get(fruit_type.lower(), fruit_weights['default'])
        base_yield = extrapolated_fruit_count * fruit_weight
        
        # Apply condition multipliers
        health_mult = 0.6 + (features.get('health_score', 0.5) * 0.8)  # 0.6-1.4x
        weather_mult = 0.7 + (features.get('rainfall_norm', 0.5) * 0.6)  # 0.7-1.3x
        ripeness_mult = 0.3 + (features.get('ripeness_ratio', 0.3) * 1.4)  # 0.3-1.7x
        
        adjusted_yield = base_yield * health_mult * weather_mult * ripeness_mult
        
        # Confidence lower for baseline model
        coverage = features.get('coverage_score', 0.3)
        confidence = 0.5 + (coverage * 0.3)  # 0.5-0.8x
        
        return adjusted_yield, confidence
    
    def _calculate_intervals(
        self,
        prediction: float,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence intervals around prediction.
        
        Uses quantile-based approach:
        - Higher confidence -> tighter interval
        - Lower confidence -> wider interval
        
        Args:
            prediction: Point prediction
            confidence: Confidence score [0-1]
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Interval width inversely related to confidence
        interval_width = (1.0 - confidence) * prediction * 0.5  # Max ±50%
        
        lower_bound = max(0, prediction - interval_width)
        upper_bound = prediction + interval_width
        
        return lower_bound, upper_bound
    
    def _analyze_factors(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze which factors contribute most to prediction.
        
        Args:
            features: Normalized features
        
        Returns:
            Dictionary of factor scores [0-1]
        """
        health_score = features.get('health_score', 0.5)
        ripeness_ratio = features.get('ripeness_ratio', 0.3)
        disease_pct = 1.0 - features.get('disease_percentage', 0.2)  # Inverse
        
        weather_score = (
            (features.get('temperature_norm', 0.5) +
             features.get('rainfall_norm', 0.5) +
             features.get('humidity_norm', 0.5)) / 3
        )
        
        coverage_score = features.get('coverage_score', 0.5)
        
        return {
            'health_score': health_score,
            'weather_favorability': weather_score,
            'ripeness_condition': ripeness_ratio,
            'disease_impact': disease_pct,
            'data_coverage': coverage_score
        }
    
    def _determine_trend(self, features: Dict[str, float]) -> str:
        """
        Determine prediction trend (improving|stable|declining).
        
        Args:
            features: Normalized features
        
        Returns:
            Trend direction string
        """
        # Based on ripeness and health relative to disease
        ripeness = features.get('ripeness_ratio', 0.3)
        health = features.get('health_score', 0.5)
        
        if ripeness > 0.7 and health > 0.7:
            return 'improving'
        elif ripeness < 0.3 and health < 0.4:
            return 'declining'
        else:
            return 'stable'
    
    def _fallback_prediction(
        self,
        extrapolated_fruit_count: int,
        orchard_area_hectares: float
    ) -> YieldPredictionResult:
        """
        Return safe fallback prediction.
        
        Args:
            extrapolated_fruit_count: Fruit count
            orchard_area_hectares: Orchard area
        
        Returns:
            YieldPredictionResult with defaults
        """
        prediction = extrapolated_fruit_count * 0.20
        
        return YieldPredictionResult(
            predicted_yield_kg=prediction,
            confidence=0.5,
            confidence_lower_bound=prediction * 0.6,
            confidence_upper_bound=prediction * 1.4,
            contributing_factors={
                'health_score': 0.5,
                'weather_favorability': 0.5,
                'ripeness_condition': 0.3,
                'disease_impact': 0.7,
                'data_coverage': 0.3
            },
            trend_direction='unknown',
            model_used='fallback'
        )
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            joblib.dump(self.xgb_model, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            self.xgb_model = joblib.load(filepath)
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
