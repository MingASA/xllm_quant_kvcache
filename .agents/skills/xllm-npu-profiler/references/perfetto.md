<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# Viewing and analyzing traces in Perfetto

## Headless server analysis

A browser is not required to generate or inspect a trace on the server. If a
compatible Trace Processor is installed, run it directly on the exported file:

```bash
# In the environment holding the trace; use the installed executable path.
trace_processor query /absolute/path/to/msprof_timestamp.json \
  'SELECT COUNT(*) AS slices FROM slice;'
```

Check the installed version's help for syntax and platform support. If the tool
is unavailable, validate the exported JSON and provide its absolute server path,
size, SHA-256, rank/device, and the transfer command from
[capture.md](capture.md#4-make-artifacts-accessible-to-the-viewing-machine).
A second machine or a desktop installation is not required to finish capture.

SQL results can support timeline analysis, but report browser loading and
screenshots as pending until someone actually opens the trace in Perfetto.

## Load the actual file

1. Open `https://ui.perfetto.dev` with browser controls and use **Open trace file**
   on the viewing machine to select a trace accessible there, or use supported
   file drag-and-drop. This may be the server itself or a separate workstation.
2. Wait for parsing and confirm a nonempty timeline, process/thread or device
   stream tracks, and selectable events. Record import warnings. The welcome
   page alone does not mean the trace was loaded.
3. Start with an overview, then locate steady requests/steps. Search for kernel,
   HCCL, runtime API, or MSTX names that actually appear. Pin relevant tracks,
   zoom into a window, and inspect slice start times, durations, and arguments.
   Save overview and interval screenshots with rank, phase, and window in filenames.
4. Retain the original local trace. Sharing or public uploading is not required
   to open it. The website URL does not contain the local trace; deliver its
   actual file path and the machine holding it as well.

Start with a representative rank. Inspect other ranks when investigating
communication tails or load imbalance. Do not manually concatenate JSON files
to construct a global timeline across ranks.

## Large files or unavailable file selection

Perfetto can connect to a local native Trace Processor. Check the installed tool's
help and the [official large-trace guide](https://perfetto.dev/docs/visualization/large-traces).
For this browser-connected mode, run it on the **viewing machine**, with the
trace accessible there, for example:

```bash
# Use a separate tools directory; check for an existing installation first.
curl -fL https://get.perfetto.dev/trace_processor -o trace_processor
chmod +x trace_processor
./trace_processor --httpd /absolute/path/to/msprof_timestamp.json
```

Some versions also offer `trace_processor server http <trace>`; follow the installed
version's help. Open Perfetto, select the detected local accelerator, and confirm
it loaded the intended trace. The default endpoint is `127.0.0.1:9001`; do not bind
the service publicly to work around file selection. If browser controls run on
another machine, their localhost is not the capture server. Transfer the trace
to the viewing machine or retain server-side SQL analysis and report UI validation
as pending; do not assume a server loopback listener is reachable by that browser. Stop the trace processor
started for this task when finished.

## Cross-check with SQL

Inspect `slice` and `track` in the Perfetto Query/SQL panel first. This query lists
hotspots grouped by track and name; it does not directly give kernel-only shares:

```sql
SELECT s.track_id, t.name AS track, s.name,
       COUNT(*) AS calls,
       SUM(s.dur) / 1e6 AS total_ms,
       AVG(s.dur) / 1e3 AS avg_us
FROM slice AS s
LEFT JOIN track AS t ON t.id = s.track_id
WHERE s.dur > 0
GROUP BY s.track_id, t.name, s.name
ORDER BY total_ms DESC
LIMIT 40;
```

Perfetto SQL uses nanoseconds for `ts/dur`; source Chrome JSON commonly uses
microseconds. Do not mix units. Select the rank/device/stream and phase before
filtering `track_id` and `ts`. Clip slices crossing window boundaries to the
selected interval. Nested scopes and concurrent streams cannot simply be summed
to obtain busy time. If no slices appear, inspect import warnings and raw events
instead of drawing conclusions from an empty table.

Ascend JSON may trigger `slice_spill_overlapping_complete_event`: complete events
on one thread overlap without proper nesting, so Perfetto places them on overflow
tracks. Record importer explanations and counts, and compare source events with
imported slices. Do not delete events to suppress warnings or interpret display
overflow as additional physical streams or parallelism.

## Diagnostic rules

- Check capture coverage first. Report missing CPU, HCCL, or rank tracks as "not
  captured," rather than concluding the activity did not occur. Do not invent
  prefill/decode labels when phases cannot be distinguished.
- Aggregate kernel calls, total duration, and mean duration by name, then interpret
  hotspots in their stream and phase context. Do not add CPU scopes, runtime APIs,
  and device kernels together as device time.
- Measure communication overlap using compute/communication interval intersections
  on the same clock and within the same window. Summed stream durations can exceed
  wall time; calculate busy/idle time using interval unions and state the denominator.
- Investigate host bubbles using the previous device task's end, the next task's
  start, intervening Host APIs, synchronization, copies, graph replay, and other
  streams. Blank space or a single threshold does not establish a host bottleneck.
- Compare rank skew only for matching requests/steps with verified clocks. A
  single-rank trace cannot characterize an entire TP/EP group. Do not concatenate
  independent JSON files and introduce PID/TID or clock collisions.
- Timelines alone do not establish KV fragmentation, HBM bandwidth utilization,
  or fusion opportunities. Obtain the relevant metrics, operator shapes, and
  current source before concluding. Separate observations, hypotheses, and tests.
- Profiling explains bottlenecks. User-visible speedups require matched
  before/after measurements with profiling disabled; cumulative operator time
  cannot directly establish a throughput improvement.

## Record reproducible evidence

For each bottleneck, record the following in `timeline_notes.md`:

```text
Trace / SHA-256 / rank / device:
Phase and identification evidence:
Selected tracks:
Window [start, end], units, and time origin:
Observation: event names, call counts, durations, gaps, or overlap
Screenshot paths / SQL and filters:
Interpretation: confirmed facts, candidate causes, missing evidence
Next steps: source locations, testable changes, comparison without profiling
```

For decode gaps, identify adjacent device tasks, activity on other streams, host
activity, and synchronization waits. For HCCL, distinguish total communication
time from communication time not overlapped by compute. For graph replay,
distinguish initial capture/compilation from steady replay. Mark unmeasurable
fields as not covered; fixed percentage thresholds do not establish causality.

UI reference: [Official Perfetto UI documentation](https://perfetto.dev/docs/visualization/perfetto-ui).
