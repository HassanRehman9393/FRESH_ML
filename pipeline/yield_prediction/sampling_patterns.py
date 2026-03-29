"""
Yield Prediction: Sampling Pattern Generator
Implements W-shaped and zigzag sampling patterns to extrapolate
fruit counts from sample detections to full orchard yield.
"""

import math
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Result of sampling pattern extrapolation"""
    extrapolated_fruit_count: int
    sampling_factor: float
    pattern_used: str
    sample_coverage_percent: float
    confidence: float


class SamplingPatternGenerator:
    """
    Generate yield estimates using agricultural sampling patterns.
    
    Supported patterns:
    - W-shaped: Traverses field in W pattern, statistically optimal
    - Zigzag: Traverses field in zigzag pattern, simple and effective
    
    References:
    - W-shaped sampling is recommended for uniform crop surveys
    - Accounts for edge effects and spatial variability
    """
    
    # Standard fruit weights (kg) - for Pakistan/FRESH system
    FRUIT_WEIGHTS = {
        'mango': 0.25,           # 250g average mango
        'orange': 0.18,          # 180g average orange
        'guava': 0.18,           # 180g average guava
        'grapefruit': 0.35,      # 350g average grapefruit
        'default': 0.25
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extrapolate_w_shaped(
        self,
        detected_fruit_count: int,
        sample_area_m2: float,
        total_area_m2: float,
        fruit_type: str = 'default',
        consistency_factor: float = 0.85
    ) -> SamplingResult:
        """
        Extrapolate yield using W-shaped sampling pattern.
        
        The W-shaped pattern systematically covers the field:
        - Enters at one corner, follows W trajectory across field
        - Accounts for edge effects and spatial variability
        - More statistically rigorous than random sampling
        
        Args:
            detected_fruit_count: Number of fruits detected in sample
            sample_area_m2: Area covered by detections (m²)
            total_area_m2: Total orchard area (m²)
            fruit_type: Type of fruit (for weight conversion)
            consistency_factor: Accounts for non-uniform distribution (0-1)
        
        Returns:
            SamplingResult with extrapolated count and confidence
        """
        if sample_area_m2 <= 0 or total_area_m2 <= 0:
            self.logger.error(
                f"Invalid area: sample={sample_area_m2}, total={total_area_m2}"
            )
            return self._empty_result('w-shaped')
        
        # Base extrapolation factor
        base_factor = total_area_m2 / sample_area_m2
        
        # W-pattern uses cross-diagonal coverage
        # Reduces edge effect bias vs simple area-based extrapolation
        edge_effect_correction = 0.95  # Slightly reduce for edge effects
        w_factor = base_factor * edge_effect_correction
        
        # Apply consistency/variability factor
        final_factor = w_factor * consistency_factor
        
        # Extrapolate fruit count
        extrapolated_count = int(detected_fruit_count * final_factor)
        
        # Calculate coverage percentage
        coverage_pct = (sample_area_m2 / total_area_m2) * 100
        
        # Confidence decreases with lower coverage
        confidence = self._calculate_confidence(
            coverage_pct,
            detected_fruit_count,
            pattern='w-shaped'
        )
        
        return SamplingResult(
            extrapolated_fruit_count=extrapolated_count,
            sampling_factor=final_factor,
            pattern_used='w-shaped',
            sample_coverage_percent=coverage_pct,
            confidence=confidence
        )
    
    def extrapolate_zigzag(
        self,
        detected_fruit_count: int,
        sample_area_m2: float,
        total_area_m2: float,
        fruit_type: str = 'default',
        consistency_factor: float = 0.80
    ) -> SamplingResult:
        """
        Extrapolate yield using zigzag sampling pattern.
        
        The zigzag pattern traverses field in parallel lines:
        - Simple to implement in practice
        - Good for rectangular fields
        - Good for drone flight paths
        
        Args:
            detected_fruit_count: Number of fruits detected in sample
            sample_area_m2: Area covered by detections (m²)
            total_area_m2: Total orchard area (m²)
            fruit_type: Type of fruit (for weight conversion)
            consistency_factor: Accounts for non-uniform distribution (0-1)
        
        Returns:
            SamplingResult with extrapolated count and confidence
        """
        if sample_area_m2 <= 0 or total_area_m2 <= 0:
            self.logger.error(
                f"Invalid area: sample={sample_area_m2}, total={total_area_m2}"
            )
            return self._empty_result('zigzag')
        
        # Basic extrapolation
        base_factor = total_area_m2 / sample_area_m2
        
        # Zigzag pattern has more edge effects than W
        # Apply slight correction
        edge_effect_correction = 0.92
        zigzag_factor = base_factor * edge_effect_correction
        
        # Apply consistency/variability factor
        final_factor = zigzag_factor * consistency_factor
        
        # Extrapolate
        extrapolated_count = int(detected_fruit_count * final_factor)
        
        # Coverage percentage
        coverage_pct = (sample_area_m2 / total_area_m2) * 100
        
        # Confidence
        confidence = self._calculate_confidence(
            coverage_pct,
            detected_fruit_count,
            pattern='zigzag'
        )
        
        return SamplingResult(
            extrapolated_fruit_count=extrapolated_count,
            sampling_factor=final_factor,
            pattern_used='zigzag',
            sample_coverage_percent=coverage_pct,
            confidence=confidence
        )
    
    def convert_to_yield_kg(
        self,
        fruit_count: int,
        fruit_type: str = 'default'
    ) -> Tuple[float, float]:
        """
        Convert fruit count to yield in kilograms.
        
        Args:
            fruit_count: Total number of fruits
            fruit_type: Type of fruit (mango, citrus, etc.)
        
        Returns:
            Tuple of (yield_kg, confidence_in_weight_estimate)
        """
        fruit_weight = self.FRUIT_WEIGHTS.get(fruit_type, self.FRUIT_WEIGHTS['default'])
        
        # Some fruits vary more in weight than others
        weight_variability = {
            'mango': 0.15,      # ±15% weight variation
            'orange': 0.12,
            'guava': 0.12,
            'grapefruit': 0.15,
            'default': 0.12
        }
        
        variability = weight_variability.get(fruit_type, weight_variability['default'])
        
        # Calculate yield
        yield_kg = fruit_count * fruit_weight
        
        # Confidence decreases with fruit count uncertainty
        # More fruits = more averaging = higher confidence
        weight_confidence = 1.0 - variability
        count_confidence = min(1.0, fruit_count / 1000)  # High confidence at 1000+ fruits
        
        confidence = weight_confidence * count_confidence
        
        return yield_kg, confidence
    
    def get_optimal_sampling_factor(
        self,
        orchard_area_hectares: float,
        num_images: int,
        avg_detection_area_m2: int = 5000
    ) -> Dict[str, float]:
        """
        Calculate optimal sampling parameters based on orchard size and coverage.
        
        Args:
            orchard_area_hectares: Orchard size in hectares
            num_images: Number of drone images taken
            avg_detection_area_m2: Average area coverage per image (m²)
        
        Returns:
            Dictionary with sampling recommendations
        """
        total_area_m2 = orchard_area_hectares * 10000
        total_coverage_m2 = num_images * avg_detection_area_m2
        coverage_pct = (total_coverage_m2 / total_area_m2) * 100
        
        # Recommendations based on coverage
        if coverage_pct < 2:
            recommendation = 'INSUFFICIENT'
            consistency_factor = 0.70
        elif coverage_pct < 5:
            recommendation = 'LOW'
            consistency_factor = 0.75
        elif coverage_pct < 15:
            recommendation = 'MODERATE'
            consistency_factor = 0.85
        elif coverage_pct < 30:
            recommendation = 'GOOD'
            consistency_factor = 0.92
        else:
            recommendation = 'EXCELLENT'
            consistency_factor = 0.98
        
        return {
            'total_area_m2': total_area_m2,
            'coverage_m2': total_coverage_m2,
            'coverage_percentage': coverage_pct,
            'num_images': num_images,
            'avg_area_per_image_m2': avg_detection_area_m2,
            'recommendation': recommendation,
            'suggested_consistency_factor': consistency_factor
        }
    
    def _calculate_confidence(
        self,
        coverage_pct: float,
        fruit_count: int,
        pattern: str = 'w-shaped'
    ) -> float:
        """
        Calculate confidence score for extrapolation.
        
        Confidence increases with:
        - Higher coverage percentage (more data points)
        - Larger fruit detection count (less noise)
        - Better sampling pattern (W > zigzag)
        
        Args:
            coverage_pct: Percentage of field sampled
            fruit_count: Number of fruits detected
            pattern: Sampling pattern used
        
        Returns:
            Confidence score [0, 1]
        """
        # Coverage confidence: max at 20%+ coverage
        if coverage_pct < 1:
            coverage_conf = 0.3
        elif coverage_pct < 5:
            coverage_conf = 0.5
        elif coverage_pct < 15:
            coverage_conf = 0.75
        else:
            coverage_conf = 0.95
        
        # Fruit count confidence: more fruits = more reliable
        fruit_conf = min(fruit_count / 500, 1.0)  # 500+ fruits = full confidence
        
        # Pattern confidence
        pattern_conf = 0.90 if pattern == 'w-shaped' else 0.85
        
        # Combined confidence
        confidence = (coverage_conf * 0.4) + (fruit_conf * 0.4) + (pattern_conf * 0.2)
        
        return max(0, min(confidence, 1.0))
    
    def _empty_result(self, pattern: str) -> SamplingResult:
        """Return empty result with defaults"""
        return SamplingResult(
            extrapolated_fruit_count=0,
            sampling_factor=0,
            pattern_used=pattern,
            sample_coverage_percent=0,
            confidence=0
        )
