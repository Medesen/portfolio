# Service Level Objectives (SLOs)

## Overview

This document defines the Service Level Objectives (SLOs) for the Churn Prediction API. SLOs are internal targets that guide operational decisions and alert thresholds.

**Note:** These are **objectives** (SLOs), not guarantees (SLAs). They represent our operational targets.

---

## Availability SLO

**Target: 99.5% monthly uptime**

- **Measurement Window**: 30 days (rolling)
- **Downtime Budget**: 3.6 hours/month
- **Excludes**: Planned maintenance windows
- **Metric**: `up{job="churn-service"}`

### What Counts as "Up"
- Health endpoint returns 200 OK
- Model is loaded and ready
- Service can process predictions

### Monitoring
```promql
# Current availability (last 30 days)
avg_over_time(up{job="churn-service"}[30d]) * 100
```

---

## Latency SLOs

### Request Latency (End-to-End)

**Targets:**
- **p50 (median) < 50ms** - Typical user experience
- **p95 < 200ms** - Most users get fast responses
- **p99 < 500ms** - Even outliers are reasonably fast

**Measurement:**
- Includes all middleware (auth, validation, logging, metrics)
- Measured at `/predict` endpoint
- Single-record predictions only

**PromQL Queries:**
```promql
# p50 latency
histogram_quantile(0.50, rate(churn_prediction_request_duration_seconds_bucket{endpoint="/predict"}[5m]))

# p95 latency
histogram_quantile(0.95, rate(churn_prediction_request_duration_seconds_bucket{endpoint="/predict"}[5m]))

# p99 latency
histogram_quantile(0.99, rate(churn_prediction_request_duration_seconds_bucket{endpoint="/predict"}[5m]))
```

### Model Inference Latency

**Targets:**
- **p50 < 10ms** - Fast inference
- **p95 < 50ms** - Consistent performance
- **p99 < 100ms** - No severe outliers

**Measurement:**
- Pure model.predict_proba() time
- Excludes preprocessing and serialization
- Single-record inference

**PromQL Queries:**
```promql
# Model inference p95
histogram_quantile(0.95, rate(churn_model_prediction_duration_seconds_bucket[5m]))
```

### Preprocessing Latency

**Targets:**
- **p50 < 5ms** - Validation and schema alignment
- **p95 < 25ms** - Even with complex validation

---

## Accuracy SLO (Model Performance)

**Target: ROC AUC > 0.80 on validation set**

- **Measurement**: Computed during training
- **Threshold**: Models below 0.80 should not be deployed
- **Metric**: Stored in metadata JSON, tracked in MLflow

### Monitoring
- Review validation metrics before deployment
- Compare new model vs current production model
- Track drift detection alerts

---

## Error Rate SLO

**Target: < 0.1% error rate**

- **Measurement Window**: 5 minutes (rolling)
- **Includes**: 5xx errors (server errors)
- **Excludes**: 4xx errors (client errors like validation failures)

**PromQL Query:**
```promql
# Error rate (5xx errors)
rate(churn_prediction_requests_total{endpoint="/predict",status="error"}[5m]) /
rate(churn_prediction_requests_total{endpoint="/predict"}[5m]) * 100
```

---

## Drift Detection SLO

**Target: Drift detected < 2 times per week**

- **Measurement**: Automatic drift detection triggers
- **Threshold**: More than 2 alerts/week suggests data quality issues
- **Action**: Investigate data pipeline if threshold exceeded

**Monitoring:**
```promql
# Drift detections in last 7 days
sum(increase(churn_drift_detected_total[7d]))
```

---

## SLO Monitoring & Alerting

### Grafana Dashboards

**Latency & SLO Dashboard:**
- Location: `grafana/provisioning/dashboards/churn_latency_slo.json`
- Access: http://localhost:3000 (when monitoring stack is running)
- Auto-provisioned: Yes

**Key Panels:**
1. Request Latency Percentiles (p50, p95, p99)
2. Model Inference Latency
3. Preprocessing Latency
4. SLO Compliance Indicator
5. Request Rate & Error Rate

### Recommended Alerts

**Critical (PagerDuty):**
```yaml
# p99 latency exceeds 1 second
- alert: HighP99Latency
  expr: histogram_quantile(0.99, rate(churn_prediction_request_duration_seconds_bucket[5m])) > 1.0
  for: 5m
  severity: critical

# Error rate > 1%
- alert: HighErrorRate
  expr: rate(churn_prediction_errors_total[5m]) / rate(churn_prediction_requests_total[5m]) > 0.01
  for: 5m
  severity: critical
```

**Warning (Slack):**
```yaml
# p95 latency exceeds 300ms
- alert: ElevatedP95Latency
  expr: histogram_quantile(0.95, rate(churn_prediction_request_duration_seconds_bucket[5m])) > 0.3
  for: 10m
  severity: warning

# Drift detected
- alert: DriftDetected
  expr: increase(churn_drift_detected_total[1h]) > 0
  severity: warning
```

---

## SLO Review Cadence

- **Daily**: Review latency dashboards (5-minute check)
- **Weekly**: SLO compliance report (availability, latency, errors)
- **Monthly**: SLO target review and adjustment
- **Quarterly**: Model performance review (accuracy, drift)

---

## Troubleshooting SLO Violations

### High Latency (p95 > 200ms)

**Possible Causes:**
1. Model complexity increased after retraining
2. Large batch requests
3. Resource contention (CPU/memory)
4. Drift analysis running concurrently

**Investigation Steps:**
```bash
# Check recent p95 latency
make prometheus-ui
# Navigate to: Graph → histogram_quantile(0.95, rate(...))

# Check slow requests in logs
docker compose logs api | grep "Slow request"

# Check resource usage
docker stats churn-api
```

**Remediation:**
- Scale horizontally (add replicas)
- Increase resource limits
- Optimize model (reduce depth, ensemble size)
- Add request queuing/throttling

### Low Availability (< 99.5%)

**Possible Causes:**
1. Model loading failures
2. OOM kills
3. Health check failures
4. Deployment issues

**Investigation:**
```bash
# Check health status
make health

# Check container restarts
docker compose ps

# Review logs
make logs
```

---

## Latency Breakdown

**Target End-to-End Latency (p95): < 200ms**

Typical breakdown:
- **Preprocessing**: ~10-20ms (validation + schema alignment)
- **Model Inference**: ~20-50ms (Random Forest prediction)
- **Response Serialization**: ~5-10ms (Pydantic + JSON)
- **Middleware Overhead**: ~5-15ms (logging, metrics, auth)
- **Total**: ~40-95ms (well under 200ms target)

**Optimization Priorities:**
1. Model inference (largest component)
2. Preprocessing (validation can be expensive)
3. Middleware (minimal impact, but can accumulate)

---

## Performance Testing

### Load Testing

Run load tests to validate SLOs under realistic traffic:

```bash
# Install locust
pip install locust

# Run load test
cd tests
locust -f locustfile.py \
  --headless \
  --users 50 \
  --spawn-rate 10 \
  --run-time 300s \
  --host http://localhost:8000
```

**Expected Results:**
- **RPS**: 100+ requests/second
- **p50**: < 50ms
- **p95**: < 200ms
- **p99**: < 500ms
- **Error Rate**: < 0.1%

### CI/CD Performance Gates

Performance tests should run in CI to prevent regression:
- p95 latency < 500ms (lenient for CI environment)
- Error rate < 1%
- No memory leaks over 5-minute test

---

## Version History

- **2025-10-31**: Initial SLO definition
  - Latency targets: p50 < 50ms, p95 < 200ms, p99 < 500ms
  - Availability target: 99.5%
  - Error rate target: < 0.1%

