"""
Comprehensive API testing script for F1 Stats API.

Runs all edge cases and outputs results as JSON.

Usage:
    python test_api.py
    python test_api.py --output results.json
"""

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests

# API base URL
BASE_URL = "http://localhost:8000"


class APITester:
    """API testing harness."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {},
            "tests": [],
        }

    def test_endpoint(
        self,
        category: str,
        name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        expected_status: int = 200,
    ) -> Dict[str, Any]:
        """Test a single endpoint."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            response = requests.get(url, params=params, timeout=30)
            elapsed = time.time() - start_time

            result = {
                "category": category,
                "name": name,
                "endpoint": endpoint,
                "params": params,
                "status_code": response.status_code,
                "response_time_ms": round(elapsed * 1000, 2),
                "success": response.status_code == expected_status,
            }

            if response.status_code == 200:
                result["data"] = response.json()
            else:
                result["error"] = response.text

            return result

        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            return {
                "category": category,
                "name": name,
                "endpoint": endpoint,
                "params": params,
                "status_code": None,
                "response_time_ms": round(elapsed * 1000, 2),
                "success": False,
                "error": str(e),
            }

    def run_test(self, *args, **kwargs):
        """Run a test and store result."""
        result = self.test_endpoint(*args, **kwargs)
        self.results["tests"].append(result)
        status = "✅" if result["success"] else "❌"
        print(
            f"{status} [{result['category']}] {result['name']} - {result['response_time_ms']}ms"
        )
        return result

    def run_all_tests(self):
        """Run all test suites."""
        print("=" * 80)
        print("F1 STATS API - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print()

        # Health checks
        self._test_health()

        # Teammate battles
        self._test_teammate_battles()

        # Edge cases
        self._test_edge_cases()

        # Race-specific
        self._test_race_specific()

        # Track-specific
        self._test_track_specific()

        # Compound-specific TME
        self._test_compound_tme()

        # Teammate comparison endpoint
        self._test_teammate_comparison()

        # TPI testing
        self._test_tpi()

        # Error handling
        self._test_error_handling()

        # Generate summary
        self._generate_summary()

        print()
        print("=" * 80)
        print("TEST SUITE COMPLETE")
        print("=" * 80)
        print(f"Total tests: {self.results['test_summary']['total']}")
        print(f"Passed: {self.results['test_summary']['passed']}")
        print(f"Failed: {self.results['test_summary']['failed']}")
        print(
            f"Success rate: {self.results['test_summary']['success_rate']:.1f}%"
        )

    def _test_health(self):
        """Health check tests."""
        print("\n--- Health Checks ---")
        self.run_test("health", "Basic health check", "/health")
        self.run_test("health", "Model info", "/health/model")

    def _test_teammate_battles(self):
        """Teammate battle tests."""
        print("\n--- Teammate Battles (DAB) ---")

        teammates = [
            ("Red Bull", "VER", "PER"),
            ("Mercedes", "HAM", "RUS"),
            ("Ferrari", "LEC", "SAI"),
            ("McLaren", "NOR", "PIA"),
            ("Aston Martin", "ALO", "STR"),
            ("Alpine", "GAS", "OCO"),
            ("Williams", "ALB", "SAR"),
            ("Haas", "MAG", "HUL"),
            ("RB", "TSU", "RIC"),
            ("Sauber", "BOT", "ZHO"),
        ]

        for team, driver1, driver2 in teammates:
            self.run_test(
                "teammate_battle",
                f"{team}: {driver1} vs {driver2}",
                "/api/v1/drivers/compare",
                {
                    "driver_codes": [driver1, driver2],
                    "metric_type": "dab",
                    "season": 2024,
                },
            )

    def _test_edge_cases(self):
        """Edge case tests."""
        print("\n--- Edge Cases ---")

        # Frontrunner vs backmarker
        self.run_test(
            "edge_case",
            "Verstappen vs Bottas (frontrunner vs backmarker)",
            "/api/v1/drivers/compare",
            {
                "driver_codes": ["VER", "BOT"],
                "metric_type": "dab",
                "season": 2024,
            },
        )

        # Rookie vs veteran
        self.run_test(
            "edge_case",
            "Piastri vs Alonso (rookie vs veteran)",
            "/api/v1/drivers/compare",
            {
                "driver_codes": ["PIA", "ALO"],
                "metric_type": "dab",
                "season": 2024,
            },
        )

        # All 10 team representatives
        self.run_test(
            "edge_case",
            "All 10 teams comparison",
            "/api/v1/drivers/compare",
            {
                "driver_codes": [
                    "VER",
                    "HAM",
                    "LEC",
                    "NOR",
                    "ALO",
                    "GAS",
                    "ALB",
                    "MAG",
                    "TSU",
                    "BOT",
                ],
                "metric_type": "dab",
                "season": 2024,
            },
        )

    def _test_race_specific(self):
        """Race-specific tests."""
        print("\n--- Race-Specific Queries ---")

        race_ids = [1, 55, 100]
        drivers = ["VER", "LEC", "HAM"]

        for race_id in race_ids:
            for driver in drivers[:1]:  # Just test one driver per race
                self.run_test(
                    "race_specific",
                    f"DAB for {driver} in race {race_id}",
                    "/api/v1/metrics/dab",
                    {"driver_code": driver, "race_id": race_id},
                )

        # Race comparison
        self.run_test(
            "race_specific",
            "TPI comparison in race 55",
            "/api/v1/drivers/compare",
            {
                "driver_codes": ["VER", "LEC", "HAM"],
                "metric_type": "tpi",
                "race_id": 55,
            },
        )

    def _test_track_specific(self):
        """Track-specific tests."""
        print("\n--- Track-Specific Queries ---")

        tracks = ["monaco", "monza", "spa", "silverstone"]
        for track in tracks:
            self.run_test(
                "track_specific",
                f"VER DAB at {track}",
                "/api/v1/metrics/dab",
                {"driver_code": "VER", "season": 2024, "track_id": track},
            )

    def _test_compound_tme(self):
        """Compound-specific TME tests."""
        print("\n--- Compound-Specific TME ---")

        compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
        drivers = ["VER", "HAM", "LEC"]

        for compound in compounds:
            for driver in drivers[:1]:  # One driver per compound
                self.run_test(
                    "tme_compound",
                    f"{driver} on {compound}",
                    "/api/v1/metrics/tme",
                    {
                        "driver_code": driver,
                        "season": 2024,
                        "compound": compound,
                    },
                )

        # Full TME (all compounds)
        for driver in ["VER", "PER", "SAI"]:
            self.run_test(
                "tme_full",
                f"{driver} TME (all compounds)",
                "/api/v1/metrics/tme",
                {"driver_code": driver, "season": 2024},
            )

    def _test_teammate_comparison(self):
        """Teammate comparison endpoint tests."""
        print("\n--- Teammate Comparison Endpoint ---")

        drivers = ["VER", "HAM", "LEC", "NOR", "BOT"]
        for driver in drivers:
            self.run_test(
                "teammate_comparison",
                f"{driver} vs teammates",
                f"/api/v1/drivers/teammate-comparison/{driver}",
                {"season": 2024},
            )

    def _test_tpi(self):
        """TPI testing."""
        print("\n--- TPI (Clean Air Pace) ---")

        # Season TPI
        drivers = ["VER", "PER", "LEC", "SAI", "NOR", "PIA", "BOT", "SAR"]
        for driver in drivers:
            self.run_test(
                "tpi_season",
                f"{driver} season TPI",
                "/api/v1/metrics/tpi",
                {"driver_code": driver, "season": 2024},
            )

        # TPI comparison
        self.run_test(
            "tpi_comparison",
            "Teammate TPI comparison",
            "/api/v1/drivers/compare",
            {
                "driver_codes": ["VER", "PER", "LEC", "SAI"],
                "metric_type": "tpi",
                "season": 2024,
            },
        )

    def _test_error_handling(self):
        """Error handling tests."""
        print("\n--- Error Handling ---")

        # Invalid driver
        self.run_test(
            "error_handling",
            "Invalid driver code",
            "/api/v1/metrics/dab",
            {"driver_code": "XXX", "season": 2024},
            expected_status=404,
        )

        # Invalid race ID
        self.run_test(
            "error_handling",
            "Invalid race ID",
            "/api/v1/metrics/dab",
            {"driver_code": "VER", "race_id": 99999},
            expected_status=404,
        )

        # Missing parameters
        self.run_test(
            "error_handling",
            "Missing season/race_id parameter",
            "/api/v1/metrics/dab",
            {"driver_code": "VER"},
            expected_status=400,
        )

        # Invalid metric type
        self.run_test(
            "error_handling",
            "Invalid metric type",
            "/api/v1/drivers/compare",
            {
                "driver_codes": ["VER", "HAM"],
                "metric_type": "invalid",
                "season": 2024,
            },
            expected_status=422,
        )

        # Only one driver (needs min 2)
        self.run_test(
            "error_handling",
            "Too few drivers (min 2)",
            "/api/v1/drivers/compare",
            {"driver_codes": ["VER"], "metric_type": "dab", "season": 2024},
            expected_status=422,
        )

    def _generate_summary(self):
        """Generate test summary statistics."""
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t["success"])
        failed = total - passed

        # Category breakdown
        categories = {}
        for test in self.results["tests"]:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "failed": 0}
            categories[cat]["total"] += 1
            if test["success"]:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        # Response time statistics
        response_times = [
            t["response_time_ms"]
            for t in self.results["tests"]
            if t["response_time_ms"] is not None
        ]

        self.results["test_summary"] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "categories": categories,
            "response_times": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": (
                    sum(response_times) / len(response_times)
                    if response_times
                    else 0
                ),
            },
        }

    def save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test F1 Stats API")
    parser.add_argument(
        "--output",
        "-o",
        default="api_test_results.json",
        help="Output JSON file (default: api_test_results.json)",
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"API base URL (default: {BASE_URL})",
    )
    args = parser.parse_args()

    tester = APITester(base_url=args.url)
    tester.run_all_tests()
    tester.save_results(args.output)

    # Print some interesting findings
    print("\n" + "=" * 80)
    print("INTERESTING FINDINGS")
    print("=" * 80)

    # Find teammate battles
    teammate_battles = [
        t for t in tester.results["tests"] if t["category"] == "teammate_battle"
    ]
    print(f"\n📊 Teammate Battles ({len(teammate_battles)} comparisons):")
    for battle in teammate_battles[:3]:  # Show top 3
        if battle["success"] and "data" in battle:
            drivers = battle["data"].get("drivers", [])
            if len(drivers) >= 2:
                d1 = drivers[0]
                d2 = drivers[1]
                delta = abs(
                    d1["value"]["mean_dab"] - d2["value"]["mean_dab"]
                )
                print(
                    f"  {battle['name']}: Δ = {delta:.3f}s/lap (closer = more competitive)"
                )

    # Response time stats
    print(
        f"\n⏱️  Response Times: {tester.results['test_summary']['response_times']['mean']:.0f}ms avg, "
        f"{tester.results['test_summary']['response_times']['max']:.0f}ms max"
    )


if __name__ == "__main__":
    main()
