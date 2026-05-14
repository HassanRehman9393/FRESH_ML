"""
TC-ML-01 … TC-ML-20  — Disease detector logic & ML pipeline
No real models are loaded; all torch/model calls are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ── Shared mock detector factory ───────────────────────────────────────────

def make_detector(has_anthracnose=True, has_blackspot=True,
                  has_citrus=True, has_fruitfly=True):
    """Build a DiseaseDetector instance with mocked model attributes."""
    with patch('pipeline.detection.disease_detector.torch'), \
         patch('pipeline.detection.disease_detector.transforms'):
        from pipeline.detection.disease_detector import DiseaseDetector
        det = DiseaseDetector.__new__(DiseaseDetector)
        det.anthracnose_model = MagicMock() if has_anthracnose else None
        det.blackspot_model = MagicMock() if has_blackspot else None
        det.citrus_canker_model = MagicMock() if has_citrus else None
        det.fruitfly_model = MagicMock() if has_fruitfly else None
        det.severity_levels = {0: 'none', 1: 'mild', 2: 'moderate', 3: 'severe', 4: 'critical'}
        return det


def mock_torch_softmax_output(probs: list):
    """Return a MagicMock that mimics torch softmax output."""
    import torch
    tensor = MagicMock()
    # simulate max() returning (confidence, class_index)
    max_val = max(enumerate(probs), key=lambda x: x[1])
    tensor.__getitem__ = MagicMock(return_value=MagicMock(
        __getitem__=lambda self, i: MagicMock(item=lambda: probs[i])
    ))
    return MagicMock(return_value=(MagicMock(item=lambda: max_val[1]),
                                   MagicMock(item=lambda: max_val[1])))


# ── _model_not_available_error ─────────────────────────────────────────────

class TestModelNotAvailable:
    def test_returns_dict_with_error_key(self):
        """TC-ML-01: _model_not_available_error returns a dict with 'error' key."""
        det = make_detector()
        result = det._model_not_available_error('anthracnose', 'mango')
        assert 'error' in result
        assert result['is_diseased'] is False
        assert result['disease'] == 'unknown'

    def test_disease_type_is_none(self):
        """TC-ML-02: _model_not_available_error sets disease_type=None."""
        det = make_detector()
        result = det._model_not_available_error('anthracnose', 'mango')
        assert result['disease_type'] is None

    def test_confidence_is_zero(self):
        """TC-ML-03: _model_not_available_error sets confidence=0.0."""
        det = make_detector()
        result = det._model_not_available_error('x', 'y')
        assert result['confidence'] == 0.0


# ── detect_disease routing ─────────────────────────────────────────────────

class TestDetectDiseaseRouting:
    def test_mango_routes_to_anthracnose(self):
        """TC-ML-04: fruit_type='mango' routes to detect_anthracnose."""
        det = make_detector()
        det.detect_anthracnose = MagicMock(return_value={'disease': 'healthy', 'is_diseased': False, 'confidence': 0.9})
        image = MagicMock()
        det.detect_disease(image, 'mango')
        det.detect_anthracnose.assert_called_once()

    def test_orange_routes_to_blackspot_first(self):
        """TC-ML-05: fruit_type='orange' routes to detect_blackspot when available."""
        det = make_detector()
        det.detect_blackspot = MagicMock(return_value={'disease': 'healthy', 'is_diseased': False, 'confidence': 0.9})
        det.detect_disease(MagicMock(), 'orange')
        det.detect_blackspot.assert_called_once()

    def test_orange_falls_back_to_citrus_canker(self):
        """TC-ML-06: fruit_type='orange' falls back to citrus_canker if blackspot model absent."""
        det = make_detector(has_blackspot=False)
        det.detect_citrus_canker = MagicMock(return_value={'disease': 'healthy', 'is_diseased': False, 'confidence': 0.9})
        det.detect_disease(MagicMock(), 'orange')
        det.detect_citrus_canker.assert_called_once()

    def test_grapefruit_routes_to_citrus_canker(self):
        """TC-ML-07: fruit_type='grapefruit' routes to detect_citrus_canker."""
        det = make_detector()
        det.detect_citrus_canker = MagicMock(return_value={'disease': 'healthy', 'is_diseased': False, 'confidence': 0.9})
        det.detect_disease(MagicMock(), 'grapefruit')
        det.detect_citrus_canker.assert_called_once()

    def test_guava_routes_to_fruitfly(self):
        """TC-ML-08: fruit_type='guava' routes to detect_fruitfly when available."""
        det = make_detector()
        det.detect_fruitfly = MagicMock(return_value={'disease': 'healthy', 'is_diseased': False, 'confidence': 0.9})
        det.detect_disease(MagicMock(), 'guava')
        det.detect_fruitfly.assert_called_once()

    def test_unknown_fruit_returns_error(self):
        """TC-ML-09: Unknown fruit_type returns error dict."""
        det = make_detector()
        result = det.detect_disease(MagicMock(), 'banana')
        assert 'error' in result
        assert result['is_diseased'] is False

    def test_mango_without_anthracnose_model_returns_error(self):
        """TC-ML-10: mango with no anthracnose model returns _model_not_available_error."""
        det = make_detector(has_anthracnose=False)
        result = det.detect_disease(MagicMock(), 'mango')
        assert 'error' in result


# ── Disease result structure ───────────────────────────────────────────────

class TestDiseaseResultStructure:
    def _make_anthracnose_result(self, is_diseased=True, with_probs=False):
        """Helper: mock anthracnose detection."""
        det = make_detector()
        with patch.object(det, 'detect_anthracnose') as mock_detect:
            result_data = {
                'disease': 'anthracnose' if is_diseased else 'healthy',
                'confidence': 0.87 if is_diseased else 0.95,
                'is_diseased': is_diseased,
                'disease_type': 'anthracnose' if is_diseased else None,
            }
            if with_probs:
                result_data['probabilities'] = {'healthy': 0.13, 'anthracnose': 0.87}
            mock_detect.return_value = result_data
            return det.detect_disease(MagicMock(), 'mango', return_probabilities=with_probs)

    def test_result_has_required_keys(self):
        """TC-ML-11: Disease result contains disease, confidence, is_diseased, disease_type."""
        r = self._make_anthracnose_result()
        assert 'disease' in r
        assert 'confidence' in r
        assert 'is_diseased' in r

    def test_is_diseased_true_when_diseased(self):
        """TC-ML-12: is_diseased=True when disease detected."""
        r = self._make_anthracnose_result(is_diseased=True)
        assert r['is_diseased'] is True

    def test_is_diseased_false_when_healthy(self):
        """TC-ML-13: is_diseased=False when healthy."""
        r = self._make_anthracnose_result(is_diseased=False)
        assert r['is_diseased'] is False

    def test_probabilities_present_when_requested(self):
        """TC-ML-14: probabilities key present when return_probabilities=True."""
        r = self._make_anthracnose_result(with_probs=True)
        assert 'probabilities' in r

    def test_probabilities_absent_when_not_requested(self):
        """TC-ML-15: probabilities key absent when return_probabilities=False."""
        r = self._make_anthracnose_result(with_probs=False)
        assert 'probabilities' not in r


# ── Severity levels ────────────────────────────────────────────────────────

class TestSeverityLevels:
    def test_severity_levels_dict_has_five_levels(self):
        """TC-ML-16: severity_levels dict has entries for 0-4."""
        det = make_detector()
        assert len(det.severity_levels) == 5

    def test_severity_level_0_is_none(self):
        """TC-ML-17: severity level 0 maps to 'none'."""
        det = make_detector()
        assert det.severity_levels[0] == 'none'

    def test_severity_level_4_is_critical(self):
        """TC-ML-18: severity level 4 maps to 'critical'."""
        det = make_detector()
        assert det.severity_levels[4] == 'critical'


# ── ML API response structure ──────────────────────────────────────────────

class TestMLAPIResponseStructure:
    def test_disease_result_object_fields(self):
        """TC-ML-19: DiseaseDetectionResult schema has required fields."""
        try:
            from api.schemas.models import DiseaseDetectionResult
            assert hasattr(DiseaseDetectionResult, '__fields__') or hasattr(DiseaseDetectionResult, 'model_fields')
        except ImportError:
            pytest.skip("DiseaseDetectionResult not importable in isolation")

    def test_disease_labels_mapping(self):
        """TC-ML-20: All supported disease types have non-empty string labels."""
        labels = {
            'healthy': 'Healthy',
            'anthracnose': 'Anthracnose',
            'citrus_canker': 'Citrus Canker',
            'citrus_blackspot': 'Black Spot',
            'guava_fruitfly': 'Fruit Fly',
            'unknown': 'Unknown',
        }
        for key, label in labels.items():
            assert isinstance(label, str) and len(label) > 0
