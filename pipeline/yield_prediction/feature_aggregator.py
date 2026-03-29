"""
Yield Prediction: Feature Aggregation Service
Aggregates detection history and weather data into normalized feature vectors
for ML model input.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionAggregation:
    """Aggregated detection metrics for an orchard"""
    total_fruits: int
    ripe_percentage: float
    unripe_percentage: float
    overripe_percentage: float
    disease_percentage: float
    health_score: float
    average_confidence: float
    coverage_score: float
    aggregation_period_days: int
    detection_count: int


@dataclass
class WeatherAggregation:
    """Aggregated weather data for an orchard location"""
    temp_avg: float
    temp_min: float
    temp_max: float
    rainfall_sum: float
    humidity_avg: float
    humidity_min: float
    humidity_max: float
    aggregation_period_days: int
    data_points: int


class FeatureAggregator:
    """
    Aggregates detection and weather data into ML-ready feature vectors.
    
    Responsibilities:
    - Process detection history into metrics
    - Aggregate weather observations
    - Normalize all features to [0, 1] range
    - Calculate health scores and quality metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def aggregate_detections(
        self,
        detections: List[Dict],
        aggregation_days: int = 30
    ) -> DetectionAggregation:
        """
        Aggregate detection history into metrics.
        
        Args:
            detections: List of detection records with:
                - id, fruit_count, ripeness_level, disease_present, confidence, created_at
            aggregation_days: Number of days to look back
        
        Returns:
            DetectionAggregation with calculated metrics
        """
        if not detections:
            self.logger.warning("No detections provided for aggregation")
            return self._empty_detection_aggregation(aggregation_days)
        
        # Filter detections by date
        cutoff_date = datetime.utcnow() - timedelta(days=aggregation_days)
        recent_detections = [
            d for d in detections
            if datetime.fromisoformat(d.get('created_at', '')) >= cutoff_date
        ]
        
        if not recent_detections:
            self.logger.warning(
                f"No detections within {aggregation_days} days"
            )
            return self._empty_detection_aggregation(aggregation_days)
        
        # Aggregate metrics
        total_fruits = sum(d.get('fruit_count', 0) for d in recent_detections)
        total_confidence = 0
        ripeness_counts = {'ripe': 0, 'unripe': 0, 'overripe': 0}
        disease_counts = {'healthy': 0, 'diseased': 0}
        
        for detection in recent_detections:
            total_confidence += detection.get('confidence', 0)
            ripeness = detection.get('ripeness_level', 'unripe').lower()
            if ripeness in ripeness_counts:
                ripeness_counts[ripeness] += detection.get('fruit_count', 0)
            
            disease_status = 'diseased' if detection.get('disease_present', False) else 'healthy'
            disease_counts[disease_status] += 1
        
        # Calculate percentages
        total_fruit_count = sum(ripeness_counts.values())
        ripe_pct = ripeness_counts['ripe'] / total_fruit_count * 100 if total_fruit_count > 0 else 0
        unripe_pct = ripeness_counts['unripe'] / total_fruit_count * 100 if total_fruit_count > 0 else 0
        overripe_pct = ripeness_counts['overripe'] / total_fruit_count * 100 if total_fruit_count > 0 else 0
        
        total_detections = len(recent_detections)
        disease_pct = (disease_counts['diseased'] / total_detections * 100) if total_detections > 0 else 0
        
        # Calculate health score (0-1)
        health_score = self._calculate_health_score(
            disease_pct=disease_pct,
            ripe_pct=ripe_pct,
            average_confidence=total_confidence / total_detections if total_detections > 0 else 0
        )
        
        # Coverage score (based on detection frequency)
        coverage_score = min(total_detections / 10, 1.0)  # 10+ detections = full coverage
        
        return DetectionAggregation(
            total_fruits=total_fruit_count,
            ripe_percentage=ripe_pct,
            unripe_percentage=unripe_pct,
            overripe_percentage=overripe_pct,
            disease_percentage=disease_pct,
            health_score=health_score,
            average_confidence=(total_confidence / total_detections) if total_detections > 0 else 0,
            coverage_score=coverage_score,
            aggregation_period_days=aggregation_days,
            detection_count=total_detections
        )
    
    def aggregate_weather(
        self,
        weather_data: List[Dict],
        aggregation_days: int = 30
    ) -> WeatherAggregation:
        """
        Aggregate weather observations into metrics.
        
        Args:
            weather_data: List of weather records with:
                - temperature, humidity, rainfall, recorded_at
            aggregation_days: Number of days to look back
        
        Returns:
            WeatherAggregation with calculated metrics
        """
        if not weather_data:
            self.logger.warning("No weather data provided for aggregation")
            return self._empty_weather_aggregation(aggregation_days)
        
        # Filter weather by date
        cutoff_date = datetime.utcnow() - timedelta(days=aggregation_days)
        recent_weather = [
            w for w in weather_data
            if datetime.fromisoformat(w.get('recorded_at', '')) >= cutoff_date
        ]
        
        if not recent_weather:
            self.logger.warning(f"No weather data within {aggregation_days} days")
            return self._empty_weather_aggregation(aggregation_days)
        
        temps = [w.get('temperature', 20) for w in recent_weather]
        humidity_vals = [w.get('humidity', 60) for w in recent_weather]
        rainfall_sum = sum(w.get('rainfall', 0) for w in recent_weather)
        
        return WeatherAggregation(
            temp_avg=sum(temps) / len(temps) if temps else 20,
            temp_min=min(temps) if temps else 15,
            temp_max=max(temps) if temps else 35,
            rainfall_sum=rainfall_sum,
            humidity_avg=sum(humidity_vals) / len(humidity_vals) if humidity_vals else 60,
            humidity_min=min(humidity_vals) if humidity_vals else 40,
            humidity_max=max(humidity_vals) if humidity_vals else 90,
            aggregation_period_days=aggregation_days,
            data_points=len(recent_weather)
        )
    
    def normalize_features(
        self,
        detections: DetectionAggregation,
        weather: WeatherAggregation,
        orchard_area_hectares: float,
        days_since_planting: int
    ) -> Dict[str, float]:
        """
        Normalize all features to [0, 1] range for ML model.
        
        Args:
            detections: Aggregated detection metrics
            weather: Aggregated weather metrics
            orchard_area_hectares: Orchard size in hectares
            days_since_planting: Days since planting (growth stage)
        
        Returns:
            Dictionary of normalized features [0, 1]
        """
        # Most features are already percentages (0-100) or normalized
        features = {
            'ripeness_ratio': detections.ripe_percentage / 100.0,
            'disease_percentage': detections.disease_percentage / 100.0,
            'health_score': detections.health_score,
            'average_confidence': detections.average_confidence,
            'coverage_score': detections.coverage_score,
            
            # Weather features normalized
            'temperature_norm': self._normalize_temperature(weather.temp_avg),
            'rainfall_norm': self._normalize_rainfall(weather.rainfall_sum),
            'humidity_norm': weather.humidity_avg / 100.0,
            
            # Orchard features
            'area_norm': min(orchard_area_hectares / 100, 1.0),  # Normalize to 100 ha
            'growth_stage_norm': min(days_since_planting / 365, 1.0),  # Normalize to 1 year
            
            # Raw values (for model inspection)
            'total_fruits': detections.total_fruits,
            'detection_count': detections.detection_count,
        }
        
        return features
    
    def _calculate_health_score(
        self,
        disease_pct: float,
        ripe_pct: float,
        average_confidence: float
    ) -> float:
        """
        Calculate overall health score (0-1).
        
        Components:
        - Disease burden: Lower is better (0 = 100% diseased, 1 = 0% diseased)
        - Ripeness maturity: Ripe fruits are better (0.3 = not ready, 1.0 = all ripe)
        - Detection confidence: Higher confidence is better
        """
        disease_score = 1.0 - (disease_pct / 100.0)  # 1 = no disease, 0 = all diseased
        ripeness_score = min(ripe_pct / 100.0, 1.0)  # Riper = better
        confidence_score = average_confidence  # 0-1
        
        # Weighted combination
        health = (disease_score * 0.4) + (ripeness_score * 0.4) + (confidence_score * 0.2)
        return max(0, min(health, 1.0))  # Constrain to [0, 1]
    
    def _normalize_temperature(self, temp_celsius: float) -> float:
        """
        Normalize temperature to [0, 1].
        Assumes optimal range 20-30°C for tropical fruits.
        """
        if temp_celsius < 15:
            return 0.3
        elif temp_celsius > 35:
            return 0.3
        else:
            # Optimal at 25°C
            return 1.0 - (abs(temp_celsius - 25) / 20)
    
    def _normalize_rainfall(self, rainfall_mm: float) -> float:
        """
        Normalize rainfall to [0, 1].
        Assumes optimal ~100-200mm per month for tropical fruits.
        """
        if rainfall_mm < 50:
            return 0.5 + (rainfall_mm / 100)
        elif rainfall_mm > 300:
            return 0.5  # Too much rainfall
        else:
            return min(rainfall_mm / 200, 1.0)
    
    def _empty_detection_aggregation(self, days: int) -> DetectionAggregation:
        """Return empty aggregation with default values"""
        return DetectionAggregation(
            total_fruits=0,
            ripe_percentage=0,
            unripe_percentage=100,
            overripe_percentage=0,
            disease_percentage=0,
            health_score=0.5,
            average_confidence=0,
            coverage_score=0,
            aggregation_period_days=days,
            detection_count=0
        )
    
    def _empty_weather_aggregation(self, days: int) -> WeatherAggregation:
        """Return empty aggregation with default values"""
        return WeatherAggregation(
            temp_avg=25,
            temp_min=20,
            temp_max=30,
            rainfall_sum=150,
            humidity_avg=70,
            humidity_min=50,
            humidity_max=90,
            aggregation_period_days=days,
            data_points=0
        )
