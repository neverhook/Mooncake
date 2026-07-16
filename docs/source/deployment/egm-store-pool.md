# EGM Store Pool over NVLink

Mooncake Store can expose CPU-attached DRAM as an Extended GPU Memory (EGM)
Store pool through Multi-Node NVLink Fabric. A Consumer in the same NVLink
scale-up domain can put KV cache from HBM into that pool and get it back into
HBM without an application-visible CPU staging copy.

V1 implements EGM with CUDA VMM `CU_MEM_LOCATION_TYPE_HOST_NUMA` allocations.
The NUMA ID selects the Provider-side CPU DRAM node; it is an allocation and
physical-placement property, not a transport type, and it does not imply CPU
staging. See NVIDIA's
[Extended GPU Memory documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/egm.html)
for the CUDA terminology and interface.

This feature is disabled by default and is intended for GB200/NVL72-class
deployments with NVIDIA MNNVL and IMEX configured.
The architecture and scope are tracked in
[RFC #2914](https://github.com/kvcache-ai/Mooncake/issues/2914).

## Requirements and scope

- Build with `USE_CUDA=ON` and `USE_MNNVL=ON`.
- Use the legacy `NvlinkTransport` with `protocol="nvlink"`. TENT is not
  supported by this feature.
- Provider and Consumer must be in the same NVLink Fabric scale-up domain and
  have access to a working IMEX channel.
- Leave `MC_USE_NVLINK_IPC` unset. The EGM Store Pool requires Fabric handles,
  not legacy CUDA IPC handles.
- The supported data paths are HBM to EGM DRAM for Put and EGM DRAM to HBM for
  Get. EGM-to-EGM and ordinary pageable host buffers are not part of the V1
  contract.

The feature does not currently select RDMA for a Consumer outside the scale-up
domain. That dual-protocol fallback is a separate follow-up. Provider restart
can also expose the existing NVLink mapping ABA problem described in
[RFC #2832](https://github.com/kvcache-ai/Mooncake/issues/2832); do not treat a
restarted Provider's reused virtual address as proof that an old mapping is
still valid.

## Python configuration

EGM Store Pool controls are available only through the Python
configuration-dictionary overload of `MooncakeDistributedStore.setup()`. The
fixed-argument Python overload, C ABI, Rust API, and Go API do not expose these
controls.

```python
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
rc = store.setup({
    "local_hostname": "10.0.0.10:12345",
    "metadata_server": "http://10.0.0.10:8079/metadata",
    "master_server_addr": "10.0.0.10:50051",
    "protocol": "nvlink",
    "rdma_devices": "",
    "global_segment_size": 600 * 1024 * 1024,
    "local_buffer_size": 0,
    "enable_egm_store_pool": True,
    "egm_numa_nodes": "auto",
})
if rc != 0:
    raise RuntimeError(f"EGM Store Pool setup failed: {rc}")
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enable_egm_store_pool` | `false` | Use Store-owned EGM DRAM for the global Provider pool; requires `protocol="nvlink"` and `local_buffer_size=0`. |
| `egm_numa_nodes` | `auto` | Provider CPU DRAM nodes, either `auto` or comma-separated NUMA IDs such as `0,1`. |

`global_segment_size` is the requested Provider capacity. Mooncake aligns it to
the common Store/VMM granularity, distributes it across the selected NUMA
nodes, and splits it at `max_mr_size`; effective capacity can therefore be
smaller than requested capacity. V1 requires `local_buffer_size=0`; converting
the process-local workspace to EGM is a separate follow-up.

## Teardown and recovery

Always check `close()` and retain the Store object until it succeeds:

```python
import time

for attempt in range(3):
    rc = store.close()
    if rc == 0:
        break
    time.sleep(1)
else:
    raise RuntimeError("EGM Store Pool cleanup is still pending")
```

A failed close reverses as much publication as it safely can, retains the
remaining VMM ownership, and allows the same object's `close()` to be retried.
Do not discard the object or call `setup()` on it while cleanup is pending.

If the object is destroyed after cleanup still fails, Mooncake intentionally
leaks the remaining VMM owners rather than recycling a virtual address that may
still be published. Operators should resolve the Master, Transfer Engine, or
CUDA failure and restart the process.
