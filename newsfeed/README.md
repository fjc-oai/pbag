- [Toy Newsfeed](#toy-newsfeed)
  - [1. Tiny](#1-tiny)
  - [2. Web](#2-web)
  - [3. Backend](#3-backend)
  - [4. Load test](#4-load-test)
  - [5. Scale up TBD](#5-scale-up-tbd)

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
- [ ] Load test services. Identify bottleneck

## 5. Scale up TBD
- [ ] Kafka to scale up post service
- [ ] LB for feed service
