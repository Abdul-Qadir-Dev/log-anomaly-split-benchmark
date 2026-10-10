# BGL parse verification (Step 08 audit)

- Total lines read from `data/raw/BGL/BGL.log`: **4,747,963** (expected 4,747,963 -- MATCH)
- Lines dropped during parse (content field absent / fewer than 9 whitespace fields): **0** (breakdown by field count: {})
- Alert-labelled lines (first field != `-`): **348,460** (7.34% of all lines)
- Non-alert lines: **4,399,503**
- Drain3 templates: HDFS_v1 = **55**, BGL = **1346**
- BGL template count is within the 5,000 over-fragmentation flag threshold (bar: >5,000).

## BGL CHRONO seed 0 unseen_template_rate, two ways

- As recorded by `src/split.py` (per occurrence, what every H1/H3/H5 verdict used): **0.7746**
- Recomputed independently, per occurrence: **0.7746** (735,532 / 949,563 occurrences)
- Per unique template (fraction of *distinct* templates in test never seen in train): **0.7500** (450 / 600 distinct templates)

## Top 20 most frequent BGL templates (shown for verification regardless of the 5,000-template flag, since it was not triggered)

| Rank | Count | Template |
|---|---|---|
| 1 | 1,706,751 | `generating core.<NUM>` |
| 2 | 441,942 | `iar <*> dear <*>` |
| 3 | 295,764 | `<NUM> double-hummer alignment exceptions` |
| 4 | 267,725 | `<NUM> floating point alignment exceptions` |
| 5 | 201,206 | `CE sym <NUM>, at <HEX>, mask <HEX>` |
| 6 | 152,734 | `data TLB error interrupt` |
| 7 | 135,092 | `<NUM> microseconds spent in the rbs signal handler during <NUM> calls. <NUM> microseconds was the ma...` |
| 8 | 135,092 | `<NUM> total interrupts. <NUM> critical input interrupts. <NUM> microseconds total spent on critical ...` |
| 9 | 105,924 | `instruction cache parity error corrected` |
| 10 | 84,168 | `idoproxydb hit ASSERT condition: ASSERT expression=<NUM> Source file=idotransportmgr.cpp Source line...` |
| 11 | 63,493 | `data storage interrupt` |
| 12 | 61,460 | `ciod: generated <NUM> core files for program <*>` |
| 13 | 51,730 | `program interrupt: <*> <*>` |
| 14 | 51,227 | `instruction address: <HEX>` |
| 15 | 46,376 | `ciod: Error loading <*> invalid or missing program image, No such file or directory` |
| 16 | 43,643 | `ciod: LOGIN <*> failed: No such file or directory` |
| 17 | 42,912 | `ciod: Error loading <*> invalid or missing program image, Permission denied` |
| 18 | 39,457 | `ciod: Received signal <NUM>, code=<NUM>, errno=<NUM>, address=<HEX>` |
| 19 | 34,470 | `` |
| 20 | 33,939 | `<NUM> ddr errors(s) detected and corrected on rank <NUM>, symbol <NUM>, bit <NUM>` |