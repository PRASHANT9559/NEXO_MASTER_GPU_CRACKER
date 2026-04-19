# Feature Status (Requested vs Implemented)

This file tracks the 20 features requested by the user in this repo conversation.

## Count Summary
- Total features requested: **20**
- Features added by my recent commits in this thread: **6**
  - #1 CUDA Error Checking Macro
  - #2 Fix Multi-GPU Work Split
  - #3 SIGINT Handler (Ctrl+C) with checkpoint save
  - #4 SHA-256 multi-block padding fix (FIPS-style final block handling)
  - #5 Chunked Dictionary Loading (streaming chunks with carry-over handling)
  - #13 Device Selection (select specific GPU indices at runtime)

## Detailed Status

| # | Feature | Status in current codebase | Notes |
|---|---|---|---|
| 1 | CUDA Error Checking Macro | ✅ Added | `CUDA_CHECK(...)` macro and guarded CUDA runtime calls are present. |
| 2 | Fix Multi-GPU Work Split | ✅ Added | Per-device block distribution with remainder and dispatched work tracking is present. |
| 3 | SIGINT Handler (Ctrl+C) | ✅ Added | `signal(SIGINT, handleSigint)` and graceful checkpoint save path are present. |
| 4 | Fix SHA-256 Multi-Block | ✅ Added | Reworked SHA-256 to process full blocks plus correct FIPS-style final padding blocks. |
| 5 | Chunked Dictionary Loading | ✅ Added | Dictionary now streams fixed-size chunks with carry-over for split lines. |
| 6 | Rule Engine (Hashcat-style) | ❌ Not added | No rule-engine parser/executor found. |
| 7 | Hybrid Attack (Dict + Mask) | ❌ Not added | Not present in menu/modes. |
| 8 | Combinator Attack (Word1 + Word2) | ❌ Not added | Not present in menu/modes. |
| 9 | Markov Chains | ❌ Not added | Not present in kernels/workflow. |
| 10 | Auto-Tune Blocks/Threads | ❌ Not added | Static launch config currently used. |
| 11 | CUDA Streams (Async) | ❌ Not added | No stream-based async pipeline present. |
| 12 | Temperature/Power Monitor (NVML) | ❌ Not added | NVML integration not present. |
| 13 | Device Selection (GPU include/exclude) | ✅ Added | Runtime device list input supports all or explicit indices like `0,1,3`. |
| 14 | Kernel Separation (per hash type) | ❌ Not added | Hash-type switch remains inside kernels. |
| 15 | PBKDF2-HMAC-SHA256/SHA1 | ❌ Not added | Algorithms not implemented. |
| 16 | bcrypt | ❌ Not added | Not implemented. |
| 17 | WPA/WPA2 (PMKID/Handshake) | ❌ Not added | Not implemented. |
| 18 | JSON/CSV Export | ❌ Not added | No JSON/CSV reporting output path. |
| 19 | Webhook Notification (Discord/Slack) | ❌ Not added | Not implemented. |
| 20 | Hashcat .potfile Format | ⚠️ Partial/uncertain | Simple `hash:password` potfile exists; compatibility behavior not verified. |
