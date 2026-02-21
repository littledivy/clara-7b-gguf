Provides Apple's CLaRa-7B-Instruct model as GGUF. See https://github.com/apple/ml-clara/

- `clara-7b-f16.gguf` (full precision)
- `clara-7b-Q4_K_M.gguf` (quantized)

Quick start with Ollama https://ollama.com/divy/clara-7b-Q4_K_M:
```
$ ollama run divy/clara-7b-Q4_K_M
> Given the query "investigate why latency spiked" and these documents:
> [1] {
>   "id": "latency-debug",
>   "keyword": ["latency", "slow",  "p99", "debug", "investigate"],
>   "ops": ["kubectl top pods -n prod --sort-by=cpu", "curl -s localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_duration_seconds_bucket[5m]))",
>           "kubectl logs -n prod -l app=api --since=15m | grep -i slow", "psql -c 'SELECT pid,now()-pg_stat_activity.query_start AS duration,query FROM pg_stat_activity WHERE state != $$idle$$ ORDER BY duration DESC LIMIT 10;'"]
> }
> [2] {
>   "id": "clear-cache",
>   "keyword": ["cache", "clear", "invalidate", "redis"],
>   "ops": ["redis-cli -h redis.prod FLUSHDB", "kubectl rollout restart ..."],
> }
> Which document is most relevant? Respond in JSON with id and confidence. The id refers to the index of the document in the provided list.

{
      "id": 1,
      "confidence": 0.90
}
```

Or convert youself with main.py. This downloads the base Mistral model and applies the decoder_adapter LoRa via PEFT, then converts to GGUF using llama.cpp:

```
python main.py --output-dir /tmp/clara-build
```
