# Cmd
- `py web_service.py`
- `py post_service.py`
- `py feed_service.py`
- `prometheus --config.file=./prometheus.yml`
- `grafana-server --homepath "/usr/local/share/grafana"`
- `locust -f loadtest.py`

# Dashboard
- promethues http://localhost:9090/targets
- grafana http://localhost:3000/dashboard/
- locust http://0.0.0.0:8089/

# Loadtest results
1. 100 users, 50 QPS, latency(median, avg, p99): 3,5,44, error rate 0%
2. 200 users, 100 QPS, latency(median, avg, p99): 3,6,60, error rate 0%
3. 250 users, 125 QPS, latency(median, avg, p99): 3,6,67, error rate 2%
3. 300 users, 150 QPS, latency(median, avg, p99): 3,6,64, error rate 15% <-------- bottleneck found!!!