<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# NPU capture, export, and artifact transfer

Resolve PIDs, script paths, and run directories from the actual deployment before
running commands. Execute each command in the indicated environment; host and
container PIDs are not interchangeable. All names and paths below are examples.

## 1. Confirm the execution environment

Use the shell where the service runs. If already on the NPU server, skip SSH.
If already inside the service container, skip both SSH and container entry.
For a host deployment, skip container commands entirely.

```bash
# Only when the agent is on another machine:
ssh developer@npu-host
```

```bash
# Only when xLLM runs in a container and the current shell is on its host:
XLLM_CONTAINER=xllm-npu
sudo docker inspect "$XLLM_CONTAINER" --format '{{json .Mounts}}'
sudo docker exec -it "$XLLM_CONTAINER" bash
```

```bash
# In the service environment (server host or service container):
command -v msprof
msprof --help
ps -eo pid,ppid,args
npu-smi info
```

Identify the xLLM parent process using the process tree, launch logs, command line,
and `/proc/<pid>/exe`. Do not copy an example PID or simply select a worker PID
from `npu-smi`. Record `git remote -v`, `git branch --show-current`,
`git rev-parse HEAD`, and `git status --short` for the active checkout. When
synchronizing another checkout, compare these values and relevant uncommitted
contents too; matching HEADs alone do not prove that running code matches.

Locate and read the deployment's launch and request scripts in the execution environment.
Treat old scripts as configuration clues: verify the executable,
`python_model_path`, and current branch, and check platform support for options
such as cache dtype. Some NPU versions reject `kv_cache_dtype=int8`; do not retain
it just because an old script uses it. Prefer supported defaults when validating
the profiling workflow. When reusing a prebuilt executable, record its origin and
hash and distinguish this from validating a rebuild of the skill branch.

The service launch environment must include `export PROFILING_MODE=dynamic`.
Verify that the launch script preserves it. If needed, inspect only that variable
in `/proc/<pid>/environ`, without printing the entire environment. Restart an
existing service without the variable only within the task's authorization.
Starting the profiler cannot repair the earlier launch configuration.

The Python model path may create additional HCCL groups on the first request.
A ready service port does not prove those groups work. A fixed `HCCL_IF_BASE_PORT`
can cause a bind conflict at this point. Reuse a validated configuration; if logs
report `Communication_Error_Bind_IP_Port`, inspect the specific address, port,
and process, and check whether automatic HCCL port selection is appropriate.
Do not terminate unrelated jobs to release ports.

## 2. Warm up, then capture a bounded window

Send successful requests with the deployment's request script or task workload
until warmup completes. Ensure the script detects HTTP and service errors; an
error JSON returned by curl is not a successful request. Record actual input and
output lengths, concurrency, prefix-cache conditions, and early EOS behavior.

In interactive terminal A in the service PID namespace:

```bash
# Replace these example values with the verified parent PID and a unique run path.
XLLM_PARENT_PID=12345
PROFILE_RUN=/path/to/profiling/run_YYYYMMDD_HHMMSS
mkdir -p "$PROFILE_RUN"
set -o pipefail
msprof --dynamic=on --pid="$XLLM_PARENT_PID" \
  --output="$PROFILE_RUN" --model-execution=on \
  --runtime-api=on --aicpu=on 2>&1 | tee "$PROFILE_RUN/capture.log"
```

Check support for these options with the installed `msprof --help`. If unsupported,
consult documentation for that version; do not silently remove essential options
and still claim complete capture.

Wait for attachment readiness, then enter `start` in terminal A. Confirm capture
has started, run the measured workload in terminal B where the service endpoint is reachable, and
save its output and exit status in `workload.log`. After all measured requests
finish, enter `stop` in A, confirm capture has stopped, then enter `quit`. Wait
for msprof to exit and flush its data. Record control commands and their times in
capture.log or the manifest; tee may not record terminal input. Fixed sleeps do
not replace readiness checks. If a request fails, still stop this capture and
retain the failure evidence.

For automation, keep an interactive terminal session and send control commands
at the appropriate stages rather than piping them all at once. If using a FIFO
script, place it in the deployment's script directory. On failure, stop only this
collector, close the FIFO, and wait for exit before exporting.

## 3. Export timelines explicitly

Enumerate every `PROF_*` directory in this run, export each one, and record its
exit status:

```bash
msprof --export=on --output="$PROFILE_RUN/PROF_actual_name" \
  > "$PROFILE_RUN/export-PROF_actual_name.log" 2>&1
```

Do not select only the newest directory and discard other ranks. Find outputs:

```bash
find "$PROFILE_RUN" -type f \( -name 'msprof_*.json' \
  -o -name 'trace_view.json' -o -name '*.pt.trace.json' \)
```

A common location is `PROF_*/mindstudio_profiler_output/msprof_*.json`; older
versions may use `timeline/`. Inspect actual files instead of treating directory
existence as success. If only CSV files appear, inspect export logs and the
version's timeline export options. A `step_trace_*.json` containing only step
information does not replace a complete kernel timeline.

Retain original JSON and do not arbitrarily rescale timestamps. Chrome Trace
commonly uses an event array or an object containing `traceEvents`. Look for timed
events with `ph`, `ts`, `pid`, and `tid`. Complete slices typically use `ph=X` and
`dur`; paired `B/E` events are also possible. JSON parsing alone is insufficient:
verify actual tracks and slices in Perfetto next.

## 4. Make artifacts accessible to the viewing machine

For a server-only run, retain the exported files and report their absolute server
paths. No download is required for command-line analysis. If the browser already
has access to those files, open them directly. Transfer is needed only when the
browser runs on a different machine without access to the trace.

Record the server path and rank/device identity in the manifest. For containers,
also record the corresponding host path. A bind-mounted file is accessible from
the host; if the export is not mounted, stage only this run's artifacts there:


```bash
# Container deployment only, from its host: use the resolved container name.
sudo docker cp "$XLLM_CONTAINER:/container/path" /host/staging/path
```

Make staged files readable by the current user without changing permissions across
unrelated raw data.

```bash
# Run on the viewing machine only when transfer is needed; replace example values.
VIEW_ARTIFACT_DIR=/path/to/profile-artifacts/run_YYYYMMDD_HHMMSS
mkdir -p "$VIEW_ARTIFACT_DIR/timelines"
scp -r developer@npu-host:/host/path/to/exported_device_directory \
  "$VIEW_ARTIFACT_DIR/timelines/"
```

Preserve rank/device directories to avoid overwriting files with identical names.
Use `sha256sum` on Linux or `shasum -a 256` on macOS; when transferring, compare
source and destination hashes. Record byte counts and hashes even when keeping
files on the server. Raw PROF data may stay in the execution environment; viewing
one rank does not require copying all large files.

Export reference: [Ascend msprof documentation](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0005.html).
