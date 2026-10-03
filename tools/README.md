# NPU Timeline Generation Guide
## Prerequisites
- Python environment
- Chrome browser (for visualization)
## Implementation Steps
### 1. Code Modification
#### Register the subscriber
Add the following at the beginning of your program:
```cpp
MsptiMetrics::register_subscriber();
```
#### Add tracing to ACLNN functions (work for msprof as well)
Insert the following macro in your ACLNN functions where you want to measure performance:
```cpp
LLM_MSTX_RANGE();
```
#### Release the subscriber
Add this at the end of your program:
```cpp
MsptiMetrics::release_subscriber();
```
### 2. Log Processing
After running your program, process the generated log file using the timeline script:
```bash
python npu_timeline.py -i custom_log.log -o custom_output.json
```
### 3. Visualization
Open Chrome browser
Navigate to: chrome://tracing
Load the generated JSON file: custom_output.json