# Public Web-Traffic Datasets & Local Loader

`experiments/datasets.py` supports loading real public datasets from a local
file path without making network requests during tests.

## Recommended Public Datasets

1. **NASA-HTTP Web Server Logs** (Internet Traffic Archive)
   - Format: Common Log Format (CLF)
   - Size: ~3.46M requests
   - License: Public domain / academic research archive
   - Loader: `load_access_log("data/NASA_access_log_Jul95")`

2. **UCI Online Retail / Clickstream Logs**
   - Format: CSV (`timestamp,visitor_id,page_id,is_bot`)
   - License: CC BY 4.0
   - Loader: `load_clickstream_csv("data/clickstream.csv")`

## Fallback

If no local dataset file is present, `load_or_fallback(path)` seamlessly
falls back to `TrafficGenerator` with realistic Zipf (`s=1.1`) and diurnal
parameters.
