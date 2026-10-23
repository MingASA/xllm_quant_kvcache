# 原生 FlashInfer FP8 decode Graph：C++ admission 修复

基线 `72cb5810adb124b87042c71511129ca65a77980a`，开发分支 `bugfix/native-fp8-graph-admission`。本次只修 C++ execution-mode gate：CUDA build、Python model、`FP8_E4M3`、`XLLM_FLASHINFER_NATIVE_KV=1`、`XLLM_QUANTIZED_BACKEND=flashinfer`、`python_graph_backend=cudagraphs`、`enable_graph=true`、`enable_prefill_piecewise_graph=false` 必须同时成立。dtype 使用仓库现有枚举；现有 parser 的 `fp8` 别名仍代表 E4M3，正式命令使用 `fp8_e4m3`。

普通 MHA/GQA、CP=1、layerwise split=1、generate、无 speculation/MLA/linear attention/indexer/grouped cache、无 host offload/PD/XTensor/sleep/KV store/VLM 等原检查全部保留。其他量化路径仍限 eager；`auto` 的 BF16 路径不变。新增 INFO 只报告 admission，不证明 capture/replay；后者仍须检查 Python 的 `NATIVE_KV_BACKEND`（真实 graph buckets）和 `NATIVE_KV_GRAPH_REPLAY`。

用户已在 H800 验证 BF16 Graph 的 128 请求和 FP8 eager 的常驻 KV，实测 K/V scales 为 `0.125/0.0625`。本次没有登录 H800、接触结果、修改 Python/AIPerf/v4.1/v4.2 runner 或冻结数据，也没有进行 5060 上的大模型或性能模拟。

## 本机验证与修改范围

* CPU C++ GoogleTest **7/7 通过**：原生 FP8 Graph、Triton/mixed/其他 dtype 拒绝、native opt-in、非法 graph flags、eager、BF16。scratch CMake 实际复用仓库 `cc_test` 和新 target；不依赖 GPU/Torch。
* 实际修改后的 `worker_impl.cpp` **`-fsyntax-only` 通过**：沿用本机 `compile_commands.json` 的 CUDA 编译定义，补入已经存在的 msgpack、tvm-ffi include 路径；没有安装或升级依赖。没有完成全量链接，也没有执行 H800 Graph。
* clang-format **20.1.6**、`git diff --check` 通过。本机日志：`/mnt/e/AI/xllm-storage/reports/native-fp8-admission-20261022/{ctest.log,worker_impl.syntax-with-tvm.log}`；后者第一行保存完整编译命令。

变更文件：`xllm/core/runtime/{worker_impl.cpp,cuda_kv_cache_admission.h,CMakeLists.txt}`、`tests/core/runtime/{cuda_kv_cache_admission_test.cpp,CMakeLists.txt}` 和本文。没有更改子模块 revision。

## 5060 导出增量 bundle

在 5060 仓库根目录执行；只包含基线之后的小提交，不重传已经导入 H800 的历史：

```bash
ADMISSION_BASE=72cb5810adb124b87042c71511129ca65a77980a
ADMISSION_BRANCH=bugfix/native-fp8-graph-admission
ADMISSION_BUNDLE=/mnt/e/AI/xllm-storage/reports/native-fp8-admission-20261022/native-fp8-graph-admission.bundle
git bundle create "$ADMISSION_BUNDLE" "$ADMISSION_BRANCH" "^$ADMISSION_BASE"
git bundle verify "$ADMISSION_BUNDLE"
sha256sum "$ADMISSION_BUNDLE"
```

由用户中转 bundle 文件；Codex 不执行 SSH/SCP。

## H800 导入并创建隔离 worktree

以下由用户在 H800 上执行，三个 `/path/to/...` 改成实际路径。使用已经完成 BF16 Graph/FP8 eager 验证的开发容器及 Python/CUDA/FlashInfer 环境，不升级依赖。若使用容器，worktree、新 build 目录需可写，Git 元数据须可访问；不要沿用旧文档中只读挂载源码后复用旧 binary 的方式。

```bash
set -euo pipefail
export ADMISSION_REPO=/path/to/existing/h800/repository
export ADMISSION_BUNDLE=/path/to/transferred/native-fp8-graph-admission.bundle
# 仅只读读取已工作 build 的 FA2 AOT ABI 开关；不复制它的 cache/object/binary。
export H800_KNOWN_BUILD=/path/to/previous/working/h800/cmake-build
ADMISSION_BASE=72cb5810adb124b87042c71511129ca65a77980a
ADMISSION_BRANCH=bugfix/native-fp8-graph-admission
git -C "$ADMISSION_REPO" bundle verify "$ADMISSION_BUNDLE"
git -C "$ADMISSION_REPO" fetch "$ADMISSION_BUNDLE" \
  "refs/heads/$ADMISSION_BRANCH:refs/heads/$ADMISSION_BRANCH"
export ADMISSION_COMMIT="$(git -C "$ADMISSION_REPO" rev-parse "$ADMISSION_BRANCH")"
test "$(git -C "$ADMISSION_REPO" rev-parse "$ADMISSION_COMMIT^")" = "$ADMISSION_BASE"
export CANDIDATE="$(dirname "$ADMISSION_REPO")/xllm-native-fp8-admission-${ADMISSION_COMMIT:0:12}"
test ! -e "$CANDIDATE"
git -C "$ADMISSION_REPO" worktree add --detach "$CANDIDATE" "$ADMISSION_COMMIT"
git -C "$CANDIDATE" submodule update --init --recursive
cd "$CANDIDATE"
git diff --exit-code "$ADMISSION_BASE" HEAD -- .gitmodules third_party
git submodule status --recursive
```

主仓库 bundle 不含子模块对象；已有本地对象可复用，缺少的需从原 URL 或用户的离线镜像取得，严格使用 gitlink revision，不执行 `--remote`。全部 pin 可用 `git ls-tree -r HEAD third_party/` 查询。关键 pin：CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`、Mooncake `3eadb9c7eaf6c1111d42c477778f0d7494af1da4`、xllm_ops `4b24e6d0891a9e8604a3cde91a89a5cb979a0b1c`。新 build 会重新编译所需 C++/CUDA 目标及链接的第三方库；CUTLASS 为源码头文件依赖。CUDA 路径不执行 NPU 的 xllm_ops 预编译。FlashInfer/tvm-ffi 本次没有改动，保留已在 H800、相同环境验证过的库，无须因 admission 修复重建它们；不能移入 5060 的编译产物。

## H800 单独构建新 C++ binary

构建依据：[仓库 CUDA 复现文档](../hopper-mixed-v/README.md)、[quick start](../src/content/docs/en/getting_started/quick_start.md)、`setup.py::ExtBuild.build_extension`、`scripts/build_support/env.py::set_cuda_envs` 和根 `CMakeLists.txt`。下面显式调用同一 CMake 配置入口，仅构建 server 和新 CPU 测试，避免 `setup.py` 的 pre-build 安装动作及全量测试。固定新 `-B` 和输出目录，不使用旧 `$BASE/.../build/.../xllm`。

前提是已有开发环境包含仓库 CMake/native 依赖、CUDA Toolkit 和 Python development headers；保留原来工作的 `VCPKG_ROOT`/`DEPENDENCES_ROOT`、`CMAKE_PREFIX_PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH` 等依赖设置。仓库需要 CMake >=3.26；不满足时先报告环境阻碍，不覆盖共享依赖。

```bash
cd "$CANDIDATE"
export PYTHONDONTWRITEBYTECODE=1
export XLLM_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export LIBTORCH_ROOT="$("$XLLM_PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
export PYTORCH_INSTALL_PATH="$LIBTORCH_ROOT"
export PYTHON_LIB_PATH="$LIBTORCH_ROOT"
export PYTHON_INCLUDE_PATH="$("$XLLM_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export CUDA_TOOLKIT_ROOT_DIR="$("$XLLM_PYTHON" -c 'from torch.utils.cpp_extension import CUDA_HOME; assert CUDA_HOME; print(CUDA_HOME)')"
XLLM_CXX11_ABI="$("$XLLM_PYTHON" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
XLLM_FA2_ABI="$("$XLLM_PYTHON" - <<'PY'
import os, pathlib, re
text = (pathlib.Path(os.environ['H800_KNOWN_BUILD']) / 'CMakeCache.txt').read_text()
match = re.search(r'^FLASHINFER_FA2_0_6_18_ABI:BOOL=(ON|OFF)$', text, re.M)
assert match, 'Need the working H800 FA2 AOT ABI setting; do not infer it from the Python package version'
print(match.group(1))
PY
)"
export TORCH_CUDA_ARCH_LIST=9.0a
export XLLM_BUILD="$CANDIDATE/build/h800-native-fp8-admission"
test ! -e "$XLLM_BUILD"
cmake -S "$CANDIDATE" -B "$XLLM_BUILD" -G Ninja \
  -DUSE_CUDA=ON -DDEVICE_TYPE=USE_CUDA -DDEVICE_ARCH=X86 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGENERATE_SO=OFF \
  -DENABLE_HA=OFF -DUSE_ETCD=OFF -DBUILD_TESTING=ON \
  -DUSE_CXX11_ABI="$XLLM_CXX11_ABI" \
  -DCMAKE_CXX_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=$XLLM_CXX11_ABI" \
  -DCMAKE_CUDA_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=$XLLM_CXX11_ABI" \
  -DPython_EXECUTABLE:FILEPATH="$XLLM_PYTHON" \
  -DPython3_EXECUTABLE:FILEPATH="$XLLM_PYTHON" \
  -DTorch_DIR:PATH="$LIBTORCH_ROOT/share/cmake/Torch" \
  -DCMAKE_CUDA_COMPILER:FILEPATH="$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc" \
  -DTORCH_CUDA_ARCH_LIST=9.0a -DCMAKE_CUDA_ARCHITECTURES=90a \
  -DFLASHINFER_FA2_0_6_18_ABI="$XLLM_FA2_ABI" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH="$XLLM_BUILD/bin"
cmake --build "$XLLM_BUILD" --target xllm cuda_kv_cache_admission_test -j 4
ctest --test-dir "$XLLM_BUILD" -R '^CudaKVCacheAdmissionTest\.' --output-on-failure
export XLLM_SERVER_BINARY="$XLLM_BUILD/bin/xllm"
test -x "$XLLM_SERVER_BINARY"
```

必须在启动服务前核对源码、Torch、Python ABI 与动态链接。以下检查不会加载模型或运行 GPU workload：

```bash
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$XLLM_PYTHON" - <<'PY'
import json, os, pathlib, re, shlex, subprocess, sys, sysconfig, torch
build = pathlib.Path(os.environ['XLLM_BUILD'])
source = pathlib.Path(os.environ['CANDIDATE']).resolve()
cache = dict(re.findall(r'^([^#/:][^:]*):[^=]*=(.*)$', (build/'CMakeCache.txt').read_text(), re.M))
assert pathlib.Path(cache['CMAKE_HOME_DIRECTORY']).resolve() == source
assert cache['Python_EXECUTABLE'] == sys.executable == os.environ['XLLM_PYTHON']
assert cache['Python3_EXECUTABLE'] == sys.executable
assert pathlib.Path(cache['Torch_DIR']).resolve() == pathlib.Path(torch.__file__).parent.resolve()/'share/cmake/Torch'
assert cache['USE_CXX11_ABI'] in (str(int(torch.compiled_with_cxx11_abi())), 'ON' if torch.compiled_with_cxx11_abi() else 'OFF')
assert cache['CMAKE_CUDA_ARCHITECTURES'] == '90a'
entry = next(e for e in json.loads((build/'compile_commands.json').read_text()) if e['file'].endswith('/runtime/worker_impl.cpp'))
assert pathlib.Path(entry['file']).resolve() == source/'xllm/core/runtime/worker_impl.cpp'
assert '-DUSE_CUDA' in shlex.split(entry['command'])
assert f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}' in shlex.split(entry['command'])
ldd = subprocess.check_output(['ldd', os.environ['XLLM_SERVER_BINARY']], text=True)
(build/'xllm.ldd.txt').write_text(ldd)
assert 'not found' not in ldd, ldd
torch_lib = pathlib.Path(torch.__file__).parent.resolve()/'lib'
for name, path in re.findall(r'^\s*(\S+) => (\S+)', ldd, re.M):
    if name.startswith(('libtorch', 'libc10')):
        assert pathlib.Path(path).resolve().is_relative_to(torch_lib), (name, path)
    if name.startswith('libpython'):
        expected = pathlib.Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('LDLIBRARY')
        assert pathlib.Path(path).resolve() == expected.resolve(), (name, path, expected)
print('Verified source, CUDA build, Torch/Python environment and resolved libraries')
PY
strings "$XLLM_SERVER_BINARY" | grep -F 'Admitting native FlashInfer FP8 E4M3 resident KV cache'
git rev-parse HEAD > "$XLLM_BUILD/source-commit.txt"
sha256sum "$XLLM_SERVER_BINARY" > "$XLLM_BUILD/xllm.sha256"
export XLLM_PYTHON_MODEL_PATH="$CANDIDATE"
export PYTHONPATH="$CANDIDATE${PYTHONPATH:+:$PYTHONPATH}"
```

若依赖/header/link 检查失败，保留构建日志并解决目标开发环境问题；不能退回旧二进制来声称修复生效。H800 的完整链接、Graph 捕获和 replay 仍需在目标机确认。

下一次按既定 **16K+512 / C4 / 128 请求**做 BF16/FP8 配对，两臂均使用上面的新 `XLLM_SERVER_BINARY` 和 candidate Python 源码。保留原 runner、输入、scheduler 与其他实验参数。共同为 Python model、native KV、`--enable_graph=true --enable_prefill_piecewise_graph=false --python_graph_backend=cudagraphs`；BF16 用 `--kv_cache_dtype=auto` 并 unset `XLLM_QUANTIZED_BACKEND`，FP8 用 `--kv_cache_dtype=fp8_e4m3`、`XLLM_QUANTIZED_BACKEND=flashinfer`、`XLLM_FP8_K_SCALE=0.125`、`XLLM_FP8_V_SCALE=0.0625`。C++ admission INFO 后还必须有真实 Python capture/replay 证据；本次不另外要求一轮 GPU smoke 矩阵。
