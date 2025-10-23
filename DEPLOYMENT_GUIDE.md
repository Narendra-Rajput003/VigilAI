# VigilAI Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying VigilAI from development to production, including all phases of the system.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+
- **RAM**: 16GB+ (32GB+ for production)
- **CPU**: 8+ cores (16+ cores for production)
- **Storage**: 100GB+ SSD
- **Network**: 1Gbps+ bandwidth

### Software Requirements
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Kubernetes
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv
```

## Phase 1: Development Environment

### 1.1 Clone Repository
```bash
git clone https://github.com/vigilai/vigilai.git
cd vigilai
```

### 1.2 Setup Python Environment
```bash
# Create virtual environment
python3.9 -m venv vigilai-env
source vigilai-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Start Development Services
```bash
# Start all services with Docker Compose
docker-compose -f phase4_deployment/docker/docker-compose.yml up -d

# Verify services are running
docker-compose ps
```

### 1.4 Run Tests
```bash
# Run comprehensive test suite
python tests/test_runner.py

# Run individual test suites
pytest tests/test_phase1.py -v
pytest tests/test_phase2.py -v
pytest tests/test_phase3_cloud_backend.py -v
pytest tests/test_phase4_deployment.py -v
```

## Phase 2: Local Production Testing

### 2.1 Build Docker Images
```bash
# Build all service images
docker build -t vigilai/api-gateway:latest phase3_scalability/cloud_backend/api_gateway/
docker build -t vigilai/user-service:latest phase3_scalability/cloud_backend/microservices/user_service/
docker build -t vigilai/device-service:latest phase3_scalability/cloud_backend/microservices/device_service/
docker build -t vigilai/analytics-dashboard:latest phase4_deployment/analytics/dashboard/
```

### 2.2 Configure Environment
```bash
# Copy environment configuration
cp phase4_deployment/docker/.env.example phase4_deployment/docker/.env

# Edit configuration
nano phase4_deployment/docker/.env
```

### 2.3 Start Production Stack
```bash
# Start production services
docker-compose -f phase4_deployment/docker/docker-compose.yml up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8007/health
```

## Phase 3: Kubernetes Deployment

### 3.1 Setup Kubernetes Cluster
```bash
# For local development (minikube)
minikube start --cpus=4 --memory=8192
minikube addons enable ingress

# For production (EKS/GKE/AKS)
# Follow cloud provider documentation
```

### 3.2 Deploy Namespaces
```bash
# Create namespaces
kubectl apply -f phase4_deployment/kubernetes/namespaces/
```

### 3.3 Deploy Services
```bash
# Deploy all services
kubectl apply -f phase4_deployment/kubernetes/deployments/
kubectl apply -f phase4_deployment/kubernetes/services/
kubectl apply -f phase4_deployment/kubernetes/ingress/
```

### 3.4 Configure Monitoring
```bash
# Deploy monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values phase4_deployment/monitoring/prometheus/values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values phase4_deployment/monitoring/grafana/values.yaml
```

## Phase 4: Production Deployment

### 4.1 Infrastructure Setup
```bash
# Using Terraform (example for AWS)
cd phase4_deployment/terraform/
terraform init
terraform plan -var-file=production.tfvars
terraform apply -var-file=production.tfvars
```

### 4.2 Database Setup
```bash
# Setup PostgreSQL with high availability
kubectl apply -f phase4_deployment/kubernetes/storage/postgresql-ha.yaml

# Setup Redis cluster
kubectl apply -f phase4_deployment/kubernetes/storage/redis-cluster.yaml
```

### 4.3 Security Configuration
```bash
# Setup RBAC
kubectl apply -f phase4_deployment/security/rbac/

# Setup network policies
kubectl apply -f phase4_deployment/security/network-policies/

# Setup secrets
kubectl create secret generic vigilai-secrets \
  --from-literal=jwt-secret=your-jwt-secret \
  --from-literal=db-password=your-db-password
```

### 4.4 Monitoring Setup
```bash
# Deploy monitoring stack
helm install monitoring phase4_deployment/monitoring/ \
  --namespace monitoring \
  --values phase4_deployment/monitoring/values.yaml

# Setup alerting
kubectl apply -f phase4_deployment/monitoring/alertmanager/
```

## Phase 5: Analytics and Dashboard

### 5.1 Deploy Analytics Dashboard
```bash
# Deploy analytics services
kubectl apply -f phase4_deployment/analytics/

# Setup data pipelines
kubectl apply -f phase4_deployment/analytics/data-pipelines/
```

### 5.2 Configure Dashboards
```bash
# Import Grafana dashboards
kubectl apply -f phase4_deployment/monitoring/grafana/dashboards/

# Setup data sources
kubectl apply -f phase4_deployment/monitoring/grafana/datasources/
```

## Phase 6: CI/CD Pipeline

### 6.1 GitHub Actions Setup
```bash
# Copy workflow files
cp -r phase4_deployment/ci-cd/github-actions/.github/ .

# Configure secrets in GitHub
# - DOCKER_USERNAME
# - DOCKER_PASSWORD
# - KUBECONFIG
# - DATABASE_URL
```

### 6.2 Jenkins Pipeline (Alternative)
```bash
# Setup Jenkins
kubectl apply -f phase4_deployment/ci-cd/jenkins/

# Configure pipeline
kubectl apply -f phase4_deployment/ci-cd/jenkins/pipeline.yaml
```

## Phase 7: Performance Optimization

### 7.1 Auto-scaling Configuration
```bash
# Setup HPA
kubectl apply -f phase4_deployment/kubernetes/autoscaling/

# Setup VPA
kubectl apply -f phase4_deployment/kubernetes/autoscaling/vpa.yaml
```

### 7.2 Resource Optimization
```bash
# Setup resource quotas
kubectl apply -f phase4_deployment/kubernetes/resource-quotas/

# Setup limit ranges
kubectl apply -f phase4_deployment/kubernetes/limit-ranges/
```

## Phase 8: Security and Compliance

### 8.1 Security Scanning
```bash
# Run security scans
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image vigilai/api-gateway:latest

# Scan Kubernetes manifests
kubectl apply -f phase4_deployment/security/security-scanning/
```

### 8.2 Compliance Setup
```bash
# Setup compliance monitoring
kubectl apply -f phase4_deployment/security/compliance/

# Setup audit logging
kubectl apply -f phase4_deployment/security/audit/
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check system health
kubectl get pods -n vigilai
kubectl get services -n vigilai
kubectl get ingress -n vigilai

# Check monitoring
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

### Log Management
```bash
# View logs
kubectl logs -f deployment/api-gateway -n vigilai
kubectl logs -f deployment/user-service -n vigilai

# Setup log aggregation
kubectl apply -f phase4_deployment/monitoring/elasticsearch/
```

### Backup and Recovery
```bash
# Database backup
kubectl exec -it postgresql-0 -n vigilai -- pg_dump -U vigilai vigilai > backup.sql

# Configuration backup
kubectl get all -n vigilai -o yaml > vigilai-backup.yaml
```

## Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n vigilai

# Check logs
kubectl logs <pod-name> -n vigilai
```

#### 2. Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it postgresql-0 -n vigilai -- psql -U vigilai -d vigilai

# Check connection strings
kubectl get configmap -n vigilai
```

#### 3. Monitoring Issues
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana connectivity
curl http://localhost:3000/api/health
```

### Performance Tuning

#### 1. Database Optimization
```sql
-- Optimize PostgreSQL
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

#### 2. Redis Optimization
```bash
# Configure Redis memory
kubectl patch configmap redis-config -n vigilai --patch '{"data":{"maxmemory":"512mb"}}'
```

#### 3. Application Optimization
```bash
# Scale services
kubectl scale deployment api-gateway --replicas=5 -n vigilai
kubectl scale deployment user-service --replicas=3 -n vigilai
```

## Production Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Security scans completed
- [ ] Performance benchmarks met
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Disaster recovery tested

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Performance metrics baseline
- [ ] User acceptance testing
- [ ] Documentation updated

## Support and Maintenance

### Regular Maintenance
- **Daily**: Check system health and alerts
- **Weekly**: Review performance metrics and logs
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and optimize system architecture

### Emergency Procedures
- **System Down**: Follow disaster recovery procedures
- **Security Incident**: Activate incident response plan
- **Performance Issues**: Scale resources and optimize
- **Data Loss**: Restore from backups

## Conclusion

This deployment guide provides comprehensive instructions for deploying VigilAI from development to production. Follow the phases sequentially and ensure all prerequisites are met before proceeding to the next phase.

For additional support, refer to the documentation in each phase directory or contact the development team.
