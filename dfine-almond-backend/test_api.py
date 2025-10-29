#!/usr/bin/env python3
"""
API Test Suite for D-FINE Label Studio ML Backend
=================================================

Tests the ML backend endpoints to ensure proper functionality.

Usage:
    python test_api.py
    python test_api.py --url http://localhost:9091
    pytest test_api.py -v
"""

import argparse
import json
import sys
import requests
from typing import Dict, Any


class BackendTester:
    """Test suite for D-FINE ML backend"""

    def __init__(self, base_url: str = "http://localhost:9091"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def test_health(self) -> bool:
        """Test /health endpoint"""
        print("\n[TEST] Health Check")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            print("  ✓ PASSED")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    def test_setup(self) -> bool:
        """Test /setup endpoint"""
        print("\n[TEST] Setup Endpoint")
        try:
            response = self.session.post(f"{self.base_url}/setup", timeout=10)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  Model Version: {data.get('model_version', 'N/A')}")
                print(f"  Endpoint: {data.get('model_name', 'N/A')}")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            print("  ✓ PASSED")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    def test_predict(self, s3_url: str = "s3://test-bucket/test-image.jpg") -> bool:
        """Test /predict endpoint"""
        print("\n[TEST] Predict Endpoint")
        print(f"  Using S3 URL: {s3_url}")

        payload = {
            "tasks": [
                {
                    "data": {
                        "image": s3_url
                    }
                }
            ]
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=60
            )
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    result = data[0]
                    num_predictions = len(result.get('result', []))
                    print(f"  Predictions: {num_predictions}")
                    print(f"  Model Version: {result.get('model_version', 'N/A')}")

                    # Show first few predictions
                    if num_predictions > 0:
                        print(f"  Sample predictions:")
                        for i, pred in enumerate(result['result'][:3], 1):
                            label = pred['value']['rectanglelabels'][0]
                            score = pred.get('score', 0)
                            print(f"    {i}. {label}: {score:.3f}")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            print("  ✓ PASSED")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    def test_predict_invalid_task(self) -> bool:
        """Test /predict with invalid task"""
        print("\n[TEST] Predict with Invalid Task")

        payload = {
            "tasks": [
                {
                    "data": {}  # Missing 'image' key
                }
            ]
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )
            print(f"  Status: {response.status_code}")

            # Should handle gracefully (either 400 or 200 with empty result)
            assert response.status_code in [200, 400, 422], \
                f"Expected 200/400/422, got {response.status_code}"
            print("  ✓ PASSED (Error handled gracefully)")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    def run_all_tests(self, s3_url: str = None) -> Dict[str, bool]:
        """Run all tests"""
        print("="*70)
        print("D-FINE ML Backend - API Test Suite")
        print("="*70)
        print(f"Backend URL: {self.base_url}")

        results = {}

        # Test health endpoint
        results['health'] = self.test_health()

        # Test setup endpoint
        results['setup'] = self.test_setup()

        # Test predict endpoint (only if S3 URL provided)
        if s3_url:
            results['predict'] = self.test_predict(s3_url)
            results['predict_invalid'] = self.test_predict_invalid_task()
        else:
            print("\n[SKIP] Predict tests (no S3 URL provided)")
            print("  Use --s3-url to test prediction endpoint")

        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"  {test_name:20s} {status}")

        print(f"\nTotal: {passed}/{total} tests passed")
        print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(description='Test D-FINE ML Backend API')
    parser.add_argument(
        '--url',
        default='http://localhost:9091',
        help='Backend URL (default: http://localhost:9091)'
    )
    parser.add_argument(
        '--s3-url',
        help='S3 image URL for testing predictions (e.g., s3://bucket/image.jpg)'
    )

    args = parser.parse_args()

    tester = BackendTester(args.url)
    results = tester.run_all_tests(s3_url=args.s3_url)

    # Exit with error if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
