- [Toy Newsfeed](#toy-newsfeed)
  - [1. Tiny](#1-tiny)
  - [2. Web](#2-web)
  - [3. Backend](#3-backend)
  - [4. Load test](#4-load-test)
  - [5. Scale up TBD](#5-scale-up-tbd)
    - [Set up Prometheus and Grafana](#set-up-prometheus-and-grafana)

# Toy Newsfeed

## 1. Tiny
- [x] Newsfeed class implementation
- [x] Unit test 
- [ ][Low] CLI entrypoint

## 2. Web
- [x] Web interface and URL to run Newsfeed

## 3. Backend
- [x] Separate frontend (web) and backend
- [x] Separate feed and post services
- [ ][Mid] Correct logging, e.g. level, format, etc
- [ ][Low] Fix FastApi type for uids list[str] 

## 4. Load test
- [ ] Build monitoring system
  - [ ] QPS for each API
  - [ ] Latency
  - [ ] Error rate
- [ ] Load test services. Identify bottleneck

## 5. Scale up TBD
- [ ] Kafka to scale up post service
- [ ] LB for feed service

### Set up Prometheus and Grafana
1. brew install prometheus
2. brew install grafana
3. Config prometheus (checkout prometheus.yml)
4. Run Prometheus
  prometheus --config.file=./prometheus.yml
5. Run Grafana
  grafana-server --homepath "/usr/local/share/grafana"
  Or (grafana server --homepath "/opt/homebrew/opt/grafana/share/grafana")
6. Set up a Data Source in Grafana
   1. Open your browser to http://localhost:3000.
   2. Configuration → Data Sources → Add Data Source → Prometheus.
   3. URL field, put http://localhost:9090
7. Add Prometheus Instrumentation to FastAPI
   1. pip install prometheus-fastapi-instrumentator
   2. Checkout web_service.py
8. Monitor metrics
   1. Check Prometheus at http://localhost:9090/targets to see if your FastAPI app is “UP.”
   2. View metrics at http://localhost:9090/graph.
   3. Add query `rate(http_requests_total[1m]) * 60`