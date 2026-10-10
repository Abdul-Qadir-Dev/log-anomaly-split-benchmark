# BGL unseen-template sanity check (Step 10)

BGL/CHRONO/seed0. `results/full/` frozen -- reads the existing split's train/test session ids only, no cell re-run.

- Total test-session template occurrences: 949,563
- Unseen (never in train) occurrences: 735,532 (0.7746 = 77.5%)
- Similarity metric: **Jaccard similarity on lowercased whitespace-split tokens** of the Drain3-masked template string (e.g. `<NUM>`, `<HEX>`, `<*>` count as tokens).
- Fraction of unseen occurrences whose nearest training template has similarity > 0.8: **0.0155**

## 20 most frequent unseen test templates

| Rank | Occurrences | Unseen test template | Nearest train template | Jaccard sim |
|---|---|---|---|---|
| 1 | 257,777 | `iar <*> dear <*>` | `<*> <*> <NUM>:<NUM> <NUM>:<NUM>` | 0.2500 |
| 2 | 135,092 | `<NUM> microseconds spent in the rbs signal handler during <NUM> calls. <NUM> mic...` | `EndServiceAction is restarting the <*> in midplane R<NUM>-M<NUM> as part of Serv...` | 0.1333 |
| 3 | 135,092 | `<NUM> total interrupts. <NUM> critical input interrupts. <NUM> microseconds tota...` | `critical input interrupt enable...<NUM>` | 0.1250 |
| 4 | 39,457 | `ciod: Received signal <NUM>, code=<NUM>, errno=<NUM>, address=<HEX>` | `<NUM>, max=<NUM>` | 0.1250 |
| 5 | 27,794 | `ciod: Error reading message prefix on CioStream socket to <IP>, <*> <*> <*> <*>` | `ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <I...` | 0.5000 |
| 6 | 22,175 | `Kernel detected <NUM> integer alignment exceptions (<NUM>) iar <HEX>, dear <HEX>...` | `<NUM> double-hummer alignment exceptions` | 0.2500 |
| 7 | 18,176 | `ciod: Error loading <*> invalid or missing program image, Permission denied` | `ciod: Error loading <*> invalid or missing program image, Exec format error` | 0.6923 |
| 8 | 16,396 | `ciod: generated <NUM> core files for program <*>` | `program interrupt: <*> <*>` | 0.2222 |
| 9 | 14,481 | `MACHINE CHECK <*> <*> <*> <*> iar <HEX> lr <HEX>)` | `machine check <*>` | 0.4286 |
| 10 | 10,436 | `<NUM> ddr error(s) detected and corrected on rank <NUM>, symbol <NUM> over <NUM>...` | `<NUM> ddr errors(s) detected and corrected on rank <NUM>, symbol <NUM>, bit <NUM...` | 0.6429 |
| 11 | 10,415 | `total of <NUM> ddr error(s) detected and corrected over <NUM> seconds` | `total of <NUM> ddr error(s) detected and corrected` | 0.8000 |
| 12 | 5,120 | `rts: kernel terminated for reason <NUM>` | `rts: kernel terminated for reason <NUM>rts: bad message header: <*> <*> <*> type...` | 0.3571 |
| 13 | 5,120 | `Error receiving packet on tree network, expecting type <NUM> instead of type <NU...` | `ciod: In packet from node <NUM>.<NUM> <*> message code <NUM> is not <NUM> or <NU...` | 0.1481 |
| 14 | 3,544 | `critical input interrupt (unit=<HEX> bit=<HEX>): warning for <*> <*> wire, suppr...` | `critical input interrupt (unit=<HEX> bit=<HEX>): warning for <*> <*> wire` | 0.5000 |
| 15 | 3,511 | `<NUM> torus receiver <*> input pipe error(s) (dcr <HEX>) detected and corrected ...` | `<NUM> torus receiver <*> input pipe error(s) (dcr <HEX>) detected and corrected` | 0.8571 |
| 16 | 3,149 | `L<NUM> DCACHE summary averages: #ofDirtyLines: <NUM> out of <NUM> #ofDirtyDblWor...` | `number of correctable errors detected in L<NUM> <*>` | 0.1333 |
| 17 | 3,148 | `<NUM> tree receiver <NUM> in re-synch state event(s) (dcr <HEX>) detected over <...` | `<NUM> tree receiver <NUM> in re-synch state event(s) (dcr <HEX>) detected` | 0.8333 |
| 18 | 2,440 | `Lustre mount FAILED : bglio<NUM> : block_id : location` | `DCR <HEX> : <HEX>` | 0.1111 |
| 19 | 1,991 | `ciod: Error reading message prefix on CioStream socket to <IP>, Connection timed...` | `ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <I...` | 0.4500 |
| 20 | 1,532 | `ciod: Error reading message prefix after <*> on CioStream socket to <IP>: <*> <*...` | `ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <I...` | 0.6471 |

## Judgement

The occurrence-weighted picture says these are predominantly **genuinely new message types**, not Drain3 over-fragmenting variants of templates it already knew: only 1.55% of unseen test occurrences have a nearest training template with Jaccard similarity above 0.8. The top-20-by-frequency table shows why the headline number isn't the whole story, though -- a handful of the highest-occurrence unseen templates (rank 11, 0.80 similarity; rank 15, 0.86; rank 17, 0.83) are clearly the *same* underlying message as a training template with an extra trailing clause appended (e.g. rank 11's `total of N ddr error(s) detected and corrected over M seconds` vs. the training set's `total of N ddr error(s) detected and corrected`), which is Drain3 clustering a longer variant of a known message as a new template rather than a real content novelty. These few cases are a real but minority contributor given the overall occurrence-weighted rate: they are high-frequency individually, but still a small fraction of the 735,532 total unseen occurrences. The large majority of the mass -- including the single most frequent unseen template (`iar <*> dear <*>`, 257,777 occurrences, 0.25 similarity to its nearest training match) -- looks like genuinely different log content, consistent with BGL's chronological test period containing real operational drift rather than an artifact of parser sensitivity.