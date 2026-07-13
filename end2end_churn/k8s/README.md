# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Churn Prediction API to a Kubernetes cluster.

> **These are example manifests, not a turnkey deployment.** Before they will
> run against a real cluster you must, at minimum:
> - **Build and push the image** to a registry and set it in `deployment.yaml`
>   (it ships as `churn-service:latest`, a placeholder).
> - **Populate the model volume** — a trained model + `.sha256` sidecar must be
>   present where the container expects them (the service fails closed without a
>   valid checksum).
> - **Mount training data** if you intend to use the `/retrain` endpoint (it runs
>   `train.py`, which needs the dataset).
> - **Replace the placeholder secret** in `secret.yaml` (it contains no real token).
>
> They are provided to show the intended production shape (replicas, HPA,
> probes, config/secret separation), not as a one-command deploy.

## Overview

The Kubernetes deployment includes:
- **Deployment**: Main application with 3 replicas for high availability
- **Service**: ClusterIP service for internal cluster access
- **ConfigMap**: Configuration management
- **HorizontalPodAutoscaler**: Automatic scaling based on CPU/memory
- **Ingress**: External access (optional)

## Prerequisites

### Local Testing (Minikube)

```bash
# Install minikube (macOS with Homebrew)
brew install minikube

# Install kubectl
brew install kubectl

# Start minikube cluster
minikube start --cpus 2 --memory 4096

# Enable ingress addon (optional, for external access)
minikube addons enable ingress
```

### Production Cluster

- Kubernetes cluster (GKE, EKS, AKS, or self-hosted)
- `kubectl` configured to access the cluster
- Container registry (Docker Hub, GCR, ECR, etc.)
- Ingress controller installed (nginx, traefik, etc.)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build image
docker build -t your-registry/churn-service:latest .

# Push to registry
docker push your-registry/churn-service:latest

# Update image in deployment.yaml
# Change: image: churn-service:latest
# To:     image: your-registry/churn-service:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
# kubectl apply -f k8s/ingress.yaml  # Optional
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -l app=churn-api

# Check deployment
kubectl get deployment churn-prediction-api

# Check service
kubectl get service churn-api-service

# View logs
kubectl logs -f deployment/churn-prediction-api

# Check autoscaler
kubectl get hpa churn-api-hpa
```

### 4. Access the API

**Port Forwarding (for testing):**

```bash
# Forward local port 8080 to service port 80
kubectl port-forward service/churn-api-service 8080:80

# Test the API
curl http://localhost:8080/health
```

**Via Ingress (if configured):**

```bash
# Get ingress address
kubectl get ingress churn-api-ingress

# Access via hostname
curl http://churn-api.example.com/health
```

**Via LoadBalancer (alternative to Ingress):**

```bash
# Change service type to LoadBalancer in service.yaml
# Get external IP
kubectl get service churn-api-service

# Access via external IP
curl http://<EXTERNAL-IP>/health
```

## Configuration

### ConfigMap

Edit `k8s/configmap.yaml` to change configuration:

```yaml
data:
  LOG_LEVEL: "INFO"           # DEBUG, INFO, WARNING, ERROR
  MAX_BATCH_SIZE: "1000"
  DRIFT_THRESHOLD_NUMERIC: "0.2"
  RATE_LIMIT_ENABLED: "true"
  RATE_LIMIT_PREDICT: "100/minute"
```

Apply changes:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/churn-prediction-api
```

### Secrets (for sensitive data)

```bash
# Create secret for service token
kubectl create secret generic churn-secrets \
  --from-literal=SERVICE_TOKEN=your-secret-token

# Reference in deployment.yaml:
# env:
#   - name: SERVICE_TOKEN
#     valueFrom:
#       secretKeyRef:
#         name: churn-secrets
#         key: SERVICE_TOKEN
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment churn-prediction-api --replicas=5

# Check status
kubectl get deployment churn-prediction-api
```

### Automatic Scaling (HPA)

The HorizontalPodAutoscaler automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

```bash
# View HPA status
kubectl get hpa churn-api-hpa

# Watch HPA in real-time
kubectl get hpa churn-api-hpa --watch

# Describe HPA for details
kubectl describe hpa churn-api-hpa
```

**Scaling behavior:**
- Min replicas: 3 (always running for HA)
- Max replicas: 10 (prevents runaway scaling)
- Scales up if CPU > 70% or memory > 80%
- Scales down if resource usage is low

## Monitoring

### View Logs

```bash
# Tail logs from all pods
kubectl logs -f -l app=churn-api

# Logs from specific pod
kubectl logs -f <pod-name>

# Logs from previous container (if crashed)
kubectl logs <pod-name> --previous
```

### Check Health

```bash
# Liveness probe (process alive?)
kubectl exec <pod-name> -- curl -f http://localhost:8000/healthz

# Readiness probe (ready to serve traffic?)
kubectl exec <pod-name> -- curl -f http://localhost:8000/readyz

# Detailed health check
kubectl exec <pod-name> -- curl -f http://localhost:8000/health
```

### Prometheus Metrics

```bash
# Port forward to access metrics
kubectl port-forward <pod-name> 8000:8000

# View metrics
curl http://localhost:8000/metrics
```

## Rolling Updates

### Update Image

```bash
# Build new image with tag
docker build -t your-registry/churn-service:v2 .
docker push your-registry/churn-service:v2

# Update deployment
kubectl set image deployment/churn-prediction-api \
  api=your-registry/churn-service:v2

# Watch rollout
kubectl rollout status deployment/churn-prediction-api

# Check rollout history
kubectl rollout history deployment/churn-prediction-api
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/churn-prediction-api

# Rollback to specific revision
kubectl rollout undo deployment/churn-prediction-api --to-revision=2
```

## Troubleshooting

### Pod Won't Start

```bash
# Check pod status
kubectl describe pod <pod-name>

# Common issues:
# - ImagePullBackOff: Image not found in registry
# - CrashLoopBackOff: Application crashes on startup
# - Pending: Insufficient resources

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### Health Check Failures

```bash
# Check liveness probe
kubectl describe pod <pod-name> | grep -A 5 "Liveness:"

# Check readiness probe
kubectl describe pod <pod-name> | grep -A 5 "Readiness:"

# Common fixes:
# - Increase initialDelaySeconds (model loading takes time)
# - Check health endpoint in pod: kubectl exec <pod> -- curl localhost:8000/health
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints churn-api-service

# If no endpoints, pods aren't ready
kubectl get pods -l app=churn-api

# Check service
kubectl describe service churn-api-service

# Test from another pod
kubectl run test --rm -it --image=curlimages/curl -- curl http://churn-api-service/health
```

## Resource Management

### View Resource Usage

```bash
# Current resource usage
kubectl top pods -l app=churn-api
kubectl top nodes

# Resource requests and limits
kubectl describe deployment churn-prediction-api | grep -A 5 "Limits:"
```

### Adjust Resources

Edit `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"   # Minimum guaranteed
    cpu: "250m"       # 0.25 CPU cores
  limits:
    memory: "2Gi"     # Maximum allowed
    cpu: "1000m"      # 1 CPU core
```

Apply changes:

```bash
kubectl apply -f k8s/deployment.yaml
```

## Production Checklist

Before deploying to production:

- [ ] Update image to use specific version tag (not `latest`)
- [ ] Set appropriate resource requests and limits
- [ ] Configure secrets for sensitive data (SERVICE_TOKEN)
- [ ] Set up persistent storage for models (PersistentVolumeClaim)
- [ ] Configure ingress with TLS/SSL
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation (ELK, Fluentd)
- [ ] Test rolling updates and rollbacks
- [ ] Set up alerting for pod failures
- [ ] Document disaster recovery procedures
- [ ] Review security policies (NetworkPolicy, PodSecurityPolicy)
- [ ] Configure backup strategy for models and data

## Learning Resources

### Kubernetes Concepts Explained

**Deployment:**
- Manages a set of identical pods
- Ensures desired number of replicas are running
- Handles rolling updates and rollbacks
- Self-healing: recreates pods if they fail

**Service:**
- Provides stable network endpoint for pods
- Load balances traffic across pods
- Types: ClusterIP (internal), NodePort (external), LoadBalancer (cloud)

**ConfigMap:**
- Stores non-sensitive configuration as key-value pairs
- Decouples configuration from container images
- Can be updated without rebuilding images

**HorizontalPodAutoscaler:**
- Automatically scales pods based on metrics
- Prevents manual scaling during traffic spikes
- Saves costs by scaling down during low traffic

**Probes:**
- Liveness: Is the application running? (restart if fails)
- Readiness: Is the application ready to serve traffic? (remove from service if fails)
- Startup: For slow-starting applications

**Ingress:**
- HTTP/HTTPS routing to services
- TLS termination
- Virtual hosting (multiple services on one IP)

### Next Steps

1. **Learn kubectl basics:** [Official Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
2. **Understand pod lifecycle:** [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
3. **Explore Helm:** Package manager for Kubernetes
4. **Try Kustomize:** Template-free configuration management
5. **Set up CI/CD:** Automate deployments (GitLab CI, GitHub Actions, ArgoCD)

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete individually
kubectl delete deployment churn-prediction-api
kubectl delete service churn-api-service
kubectl delete configmap churn-config
kubectl delete hpa churn-api-hpa
kubectl delete ingress churn-api-ingress

# Delete secrets (if created)
kubectl delete secret churn-secrets

# Stop minikube (local testing)
minikube stop
minikube delete
```

## Support

For issues or questions:
- Check pod logs: `kubectl logs -f deployment/churn-prediction-api`
- Describe resources: `kubectl describe <resource-type> <resource-name>`
- Check events: `kubectl get events --sort-by='.lastTimestamp'`
- Review Kubernetes documentation: https://kubernetes.io/docs/



