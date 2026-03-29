"""
Yield Prediction: Historical Baseline Data
Provides fallback yield values and regional averages for model initialization
and when insufficient detection data is available.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class YieldBaseline:
    """Expected yield baseline for a fruit type"""
    fruit_type: str
    region: str
    yield_per_hectare_kg: float
    yield_std_dev_kg: float
    min_yield_kg: float
    max_yield_kg: float
    source: str  # "regional_average", "historical", "literature"


class HistoricalBaselineRegistry:
    """
    Registry of historical and regional yield data.
    
    Provides baseline yields for:
    - Regional averages (published agricultural data)
    - Fruit-type specific standards
    - Quality adjustments (good/average/poor conditions)
    """
    
    # Regional yields (kg/hectare) - Pakistan only, for 4 fruit types
    # Based on Pakistan agricultural data and local growing conditions
    REGIONAL_YIELDS = {
        'mango': {
            'pakistan': {'avg': 8500, 'std_dev': 1500, 'min': 5000, 'max': 16000},
            'global_avg': 8500,
        },
        'orange': {
            'pakistan': {'avg': 32000, 'std_dev': 5000, 'min': 20000, 'max': 50000},
            'global_avg': 32000,
        },
        'guava': {
            'pakistan': {'avg': 22000, 'std_dev': 3500, 'min': 12000, 'max': 38000},
            'global_avg': 22000,
        },
        'grapefruit': {
            'pakistan': {'avg': 28000, 'std_dev': 4500, 'min': 15000, 'max': 42000},
            'global_avg': 28000,
        },
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_historical_data: Dict[str, List[Dict]] = {}  # user_id -> list of historical yields
    
    def get_regional_yield(
        self,
        fruit_type: str,
        region: str = 'pakistan'
    ) -> Optional[YieldBaseline]:
        """
        Get regional yield baseline for a fruit type.
        
        Args:
            fruit_type: Type of fruit (mango, orange, guava, grapefruit)
            region: Geographic region (currently only 'pakistan' supported)
        
        Returns:
            YieldBaseline object or None if not found
        """
        fruit_type_lower = fruit_type.lower()
        
        if fruit_type_lower not in self.REGIONAL_YIELDS:
            self.logger.warning(f"Unknown fruit type: {fruit_type}. Supported: mango, orange, guava, grapefruit")
            return self._get_default_baseline()
        
        # Always use Pakistan since that's our only region
        region_lower = 'pakistan'
        regional_data = self.REGIONAL_YIELDS[fruit_type_lower].get(
            region_lower,
            self.REGIONAL_YIELDS[fruit_type_lower]
        )
        
        if isinstance(regional_data, dict) and 'avg' in regional_data:
            baseline = YieldBaseline(
                fruit_type=fruit_type_lower,
                region=region_lower,
                yield_per_hectare_kg=regional_data['avg'],
                yield_std_dev_kg=regional_data['std_dev'],
                min_yield_kg=regional_data['min'],
                max_yield_kg=regional_data['max'],
                source='regional_average'
            )
            return baseline
        
        return self._get_default_baseline()
    
    def get_global_average_yield(self, fruit_type: str) -> float:
        """
        Get global average yield for a fruit type.
        
        Args:
            fruit_type: Type of fruit
        
        Returns:
            Average yield in kg/hectare
        """
        fruit_type_lower = fruit_type.lower()
        
        if fruit_type_lower in self.REGIONAL_YIELDS:
            return self.REGIONAL_YIELDS[fruit_type_lower].get('global_avg', 10000)
        
        return 10000  # Safe default
    
    def adjust_for_conditions(
        self,
        baseline_yield: float,
        health_score: float,
        weather_favorability: float,
        ripeness_condition: float
    ) -> float:
        """
        Adjust baseline yield based on observed conditions.
        
        Args:
            baseline_yield: Starting yield estimate (kg/hectare)
            health_score: Fruit health [0-1] (0=diseased, 1=healthy)
            weather_favorability: Weather conditions [0-1] (0=poor, 1=optimal)
            ripeness_condition: Ripeness stage [0-1] (0=immature, 1=ripe)
        
        Returns:
            Adjusted yield estimate (kg/hectare)
        """
        # Each factor multiplies the baseline
        health_multiplier = 0.5 + (health_score * 1.0)  # Range: 0.5 - 1.5
        weather_multiplier = 0.6 + (weather_favorability * 0.8)  # Range: 0.6 - 1.4
        ripeness_multiplier = 0.3 + (ripeness_condition * 1.2)  # Range: 0.3 - 1.5
        
        adjusted_yield = (
            baseline_yield * 
            health_multiplier * 
            weather_multiplier * 
            ripeness_multiplier
        )
        
        return adjusted_yield
    
    def estimate_from_detections(
        self,
        fruit_count: int,
        orchard_area_hectares: float,
        health_score: float,
        weather_favorability: float
    ) -> float:
        """
        Estimate yield from actual detection data.
        
        Uses direct observation rather than regional baseline.
        
        Args:
            fruit_count: Total fruit count (from sampling/extrapolation)
            orchard_area_hectares: Orchard size
            health_score: Overall health [0-1]
            weather_favorability: Weather conditions [0-1]
        
        Returns:
            Estimated yield in kg
        """
        if orchard_area_hectares <= 0:
            return 0
        
        # Base yield from detected fruits
        base_yield_kg = fruit_count * 0.20  # Rough average fruit weight
        
        # Apply condition adjustments
        health_adj = 0.7 + (health_score * 0.6)  # 0.7-1.3x
        weather_adj = 0.7 + (weather_favorability * 0.6)  # 0.7-1.3x
        
        adjusted_yield = base_yield_kg * health_adj * weather_adj
        
        return adjusted_yield
    
    def register_historical_yield(
        self,
        user_id: str,
        fruit_type: str,
        orchard_area_hectares: float,
        actual_yield_kg: float,
        date: str,
        conditions: Optional[Dict] = None
    ) -> bool:
        """
        Register actual harvested yield for future model training.
        
        Args:
            user_id: User identifier
            fruit_type: Type of fruit
            orchard_area_hectares: Orchard size
            actual_yield_kg: Actual harvested yield
            date: Harvest date
            conditions: Optional conditions dict (temperature, rainfall, etc.)
        
        Returns:
            Success flag
        """
        try:
            if user_id not in self.user_historical_data:
                self.user_historical_data[user_id] = []
            
            historical_record = {
                'fruit_type': fruit_type.lower(),
                'area_hectares': orchard_area_hectares,
                'yield_kg': actual_yield_kg,
                'yield_per_hectare': actual_yield_kg / orchard_area_hectares if orchard_area_hectares > 0 else 0,
                'date': date,
                'conditions': conditions or {}
            }
            
            self.user_historical_data[user_id].append(historical_record)
            self.logger.info(
                f"Registered historical yield for user {user_id}: "
                f"{actual_yield_kg}kg ({fruit_type})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to register historical yield: {e}")
            return False
    
    def get_user_average_yield(
        self,
        user_id: str,
        fruit_type: Optional[str] = None
    ) -> Optional[float]:
        """
        Get user's historical average yield.
        
        Args:
            user_id: User identifier
            fruit_type: Optional filter by fruit type
        
        Returns:
            Average yield (kg/hectare) or None if no data
        """
        if user_id not in self.user_historical_data:
            return None
        
        records = self.user_historical_data[user_id]
        
        if fruit_type:
            records = [r for r in records if r['fruit_type'] == fruit_type.lower()]
        
        if not records:
            return None
        
        avg_yield_per_ha = sum(r['yield_per_hectare'] for r in records) / len(records)
        return avg_yield_per_ha
    
    def get_yield_trend(
        self,
        user_id: str,
        fruit_type: Optional[str] = None,
        num_seasons: int = 3
    ) -> str:
        """
        Determine yield trend for user (improving|stable|declining).
        
        Args:
            user_id: User identifier
            fruit_type: Optional filter by fruit type
            num_seasons: Number of seasons to analyze
        
        Returns:
            Trend direction string
        """
        if user_id not in self.user_historical_data:
            return 'unknown'
        
        records = self.user_historical_data[user_id]
        
        if fruit_type:
            records = [r for r in records if r['fruit_type'] == fruit_type.lower()]
        
        # Sort by date
        records = sorted(records, key=lambda r: r['date'], reverse=True)
        records = records[:num_seasons]
        
        if len(records) < 2:
            return 'insufficient_data'
        
        yields = [r['yield_per_hectare'] for r in records]
        first_avg = sum(yields[len(yields)//2:]) / (len(yields) - len(yields)//2)
        recent_avg = sum(yields[:len(yields)//2]) / (len(yields)//2)
        
        change_pct = ((recent_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        if change_pct > 5:
            return 'improving'
        elif change_pct < -5:
            return 'declining'
        else:
            return 'stable'
    
    def _get_default_baseline(self) -> YieldBaseline:
        """Return safe default baseline"""
        return YieldBaseline(
            fruit_type='unknown',
            region='global',
            yield_per_hectare_kg=10000,
            yield_std_dev_kg=2000,
            min_yield_kg=5000,
            max_yield_kg=20000,
            source='default'
        )
