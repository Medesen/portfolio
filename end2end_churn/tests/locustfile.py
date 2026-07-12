"""
Load testing configuration for Churn Prediction API using Locust.

This file defines realistic user behavior patterns and performance tests
to validate SLO compliance under load.

Usage:
    # Headless (CI)
    locust -f tests/locustfile.py --headless --users 50 --spawn-rate 10 \
           --run-time 60s --host http://localhost:8000 --html report.html

    # Web UI (local)
    locust -f tests/locustfile.py --host http://localhost:8000
    # Then open http://localhost:8089

Load Testing for SLO Validation
"""

import random

from locust import HttpUser, between, events, task

# Sample customer data for realistic load testing
SAMPLE_CUSTOMERS = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
    },
    {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 48,
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 85.15,
        "TotalCharges": 4085.50,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    },
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 24,
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 1325.00,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    },
]


class ChurnAPIUser(HttpUser):
    """
    Simulates a user making requests to the Churn Prediction API.

    Task weights simulate realistic traffic patterns:
    - 10x more predictions than health checks
    - 3x more predictions than drift analysis

    Wait time: 1-3 seconds between requests (realistic user pacing)
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Called when a simulated user starts. Check if service is ready."""
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"⚠️  Service not ready: {response.status_code}")

    @task(10)  # Weight: 10x more common than other tasks
    def predict_churn(self):
        """
        Make a churn prediction request.

        This is the primary endpoint being load tested.
        Validates:
        - Response time < 500ms (p95 SLO)
        - Success rate > 99.9%
        - Correct response schema
        """
        # Randomly select a customer profile
        customer = random.choice(SAMPLE_CUSTOMERS)

        # Vary some fields for diversity
        customer = customer.copy()  # Don't modify original
        customer["tenure"] = random.randint(1, 72)
        customer["MonthlyCharges"] = round(random.uniform(20, 120), 2)
        # Recalculate TotalCharges to pass validation (tenure * monthly * factor)
        customer["TotalCharges"] = round(
            customer["tenure"] * customer["MonthlyCharges"] * random.uniform(0.8, 1.1), 2
        )

        with self.client.post(
            "/predict", json=customer, catch_response=True, name="/predict"
        ) as response:
            if response.status_code != 200:
                response.failure(f"Got status {response.status_code}")
            else:
                # Verify response structure (don't fail on slow responses - let SLO check handle that)
                try:
                    data = response.json()
                    if "churn_probability" not in data or "churn_prediction" not in data:
                        response.failure("Invalid response schema")
                    elif not (0 <= data["churn_probability"] <= 1):
                        response.failure(f"Invalid probability: {data['churn_probability']}")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"JSON parsing failed: {e}")

    @task(2)  # Weight: Less common than predictions
    def health_check(self):
        """
        Check API health endpoint.

        Health checks should be very fast (<10ms).
        """
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 0.1:
                response.failure(f"Health check slow: {response.elapsed.total_seconds():.3f}s")
            else:
                response.success()

    @task(1)  # Weight: Least common
    def readiness_check(self):
        """Check readiness probe (K8s)."""
        with self.client.get("/readyz", catch_response=True, name="/readyz") as response:
            if response.status_code not in [200, 503]:
                response.failure(f"Unexpected status: {response.status_code}")
            else:
                response.success()


# =============================================================================
# Performance Validation Hooks
# =============================================================================


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Validate SLO compliance at the end of the load test.

    This runs automatically after the test completes and checks if
    performance metrics meet our defined SLOs.
    """
    stats = environment.stats

    print("\n" + "=" * 70)
    print("SLO VALIDATION RESULTS")
    print("=" * 70)

    # Get stats for /predict endpoint (POST method)
    predict_stats = None
    for stat in stats.entries.values():
        if stat.name == "/predict" and stat.method == "POST":
            predict_stats = stat
            break

    if predict_stats and predict_stats.num_requests > 0:
        # Calculate percentiles
        p50 = predict_stats.get_response_time_percentile(0.5)
        p95 = predict_stats.get_response_time_percentile(0.95)
        p99 = predict_stats.get_response_time_percentile(0.99)
        error_rate = (predict_stats.num_failures / predict_stats.num_requests) * 100

        print("\n📊 /predict endpoint statistics:")
        print(f"   Total requests: {predict_stats.num_requests}")
        print(f"   Failures: {predict_stats.num_failures}")
        print(f"   Error rate: {error_rate:.2f}%")
        print(f"   p50 latency: {p50:.0f}ms")
        print(f"   p95 latency: {p95:.0f}ms")
        print(f"   p99 latency: {p99:.0f}ms")
        print(f"   Avg latency: {predict_stats.avg_response_time:.0f}ms")
        print(f"   RPS: {predict_stats.total_rps:.1f}")

        print("\n📋 SLO Compliance Check:")

        # SLO 1: p95 latency < 500ms (lenient for CI, production is 200ms)
        p95_slo = 500  # ms
        p95_pass = p95 < p95_slo
        print(
            f"   {'✅' if p95_pass else '❌'} p95 < {p95_slo}ms: {p95:.0f}ms {'PASS' if p95_pass else 'FAIL'}"
        )

        # SLO 2: p99 latency < 1000ms
        p99_slo = 1000  # ms
        p99_pass = p99 < p99_slo
        print(
            f"   {'✅' if p99_pass else '❌'} p99 < {p99_slo}ms: {p99:.0f}ms {'PASS' if p99_pass else 'FAIL'}"
        )

        # SLO 3: Error rate < 1%
        error_rate_slo = 1.0  # %
        error_pass = error_rate < error_rate_slo
        print(
            f"   {'✅' if error_pass else '❌'} Error rate < {error_rate_slo}%: {error_rate:.2f}% {'PASS' if error_pass else 'FAIL'}"
        )

        # Overall verdict
        all_pass = p95_pass and p99_pass and error_pass
        print(f"\n{'='*70}")
        if all_pass:
            print("✅ ALL SLOs MET - Performance is acceptable")
        else:
            print("❌ SLO VIOLATIONS DETECTED - Performance needs improvement")
        print("=" * 70)

        # Exit with error code if SLOs not met (for CI)
        if not all_pass:
            environment.process_exit_code = 1
    else:
        print("\n⚠️  No /predict requests recorded - cannot validate SLOs")
        print("=" * 70)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Log very slow requests for debugging.

    Any request >1s is logged as a warning.
    """
    if exception is None and response_time > 1000:  # 1 second
        print(f"⚠️  Slow request: {name} took {response_time:.0f}ms")
