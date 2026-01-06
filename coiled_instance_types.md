# Coiled Instance Types (Workers + Scheduler)

This doc explains how to select Coiled instance types for the Dask CLI in this repo,
plus example instance types/specs from the current account.

## Config options (YAML)

You must specify either VM types or vCPU+memory for workers and scheduler.

Example (explicit instance types):
```yaml
runtime:
  engine: coiled
  coiled:
    worker_vm_types:
      - x2gd.2xlarge
    scheduler_vm_types:
      - x2gd.2xlarge
```

Example (let Coiled recommend types from vCPU/memory):
```yaml
runtime:
  engine: coiled
  coiled:
    worker_cpu: 8
    worker_memory: 64GiB
    scheduler_cpu: 2
    scheduler_memory: 8GiB
```

Notes:
- If you set `worker_vm_types` and omit scheduler settings, the scheduler defaults
  to the same VM types.
- If you set vCPU/memory and omit scheduler settings, the scheduler inherits the
  worker vCPU/memory.

## How to list options for your account

Use Coiled to query what instance types match your target vCPU/memory:
```bash
python - <<'PY'
import coiled
print(coiled.list_instance_types(cores=8, memory="122GiB"))
PY
```

The results depend on your cloud/account permissions. Coiled does not expose
availability zone filters here, so you still need to pick a type that exists
in your VPC subnets/AZs.

To filter by CPU architecture:
```bash
python - <<'PY'
import coiled
print(coiled.list_instance_types(cores=16, memory="256GiB", arch="arm64"))
PY
```

## Example options + specs (from current account)

Example output for 8 cores / ~122-128 GiB:

| Instance type | vCPU | Memory | Notes |
| --- | --- | --- | --- |
| f1.2xlarge | 8 | ~122 GiB | x86_64 + FPGA; use only if you need FPGA or x86_64 only |
| x2gd.2xlarge | 8 | 128 GiB | ARM/Graviton; memory-heavy |
| x8g.2xlarge | 8 | 128 GiB | ARM/Graviton; memory-heavy |

Example output for 16 cores / ~244-256 GiB:

| Instance type | vCPU | Memory | Notes |
| --- | --- | --- | --- |
| f1.4xlarge | 16 | ~244 GiB | x86_64 + FPGA; avoid unless needed |
| x2gd.4xlarge | 16 | 256 GiB | ARM/Graviton; memory-heavy |
| x8g.4xlarge | 16 | 256 GiB | ARM/Graviton; memory-heavy |

Example output for 32 cores / ~244-256 GiB:

| Instance type | vCPU | Memory | Notes |
| --- | --- | --- | --- |
| d3.8xlarge | 32 | 256 GiB | HDD-heavy; slower than NVMe for shuffle |
| i4i.8xlarge | 32 | 256 GiB | x86_64 + NVMe; strong for shuffle/spill |
| i7i.8xlarge | 32 | 256 GiB | x86_64 + NVMe; newer than i4i |
| i4g.8xlarge | 32 | 256 GiB | ARM/Graviton + NVMe |

## US-West picks (curated)

These are practical defaults that tend to work in us-west VPCs. Avoid FPGA
types (f1) due to AZ availability issues.

16 vCPU / 256 GiB:
- ARM: `x2gd.4xlarge` or `x8g.4xlarge`
- x86_64: not commonly available at 16/256 in AWS; use 32/256 instead

32 vCPU / 256 GiB:
- x86_64: `i4i.8xlarge` (best for Dask shuffle/spill), `i7i.8xlarge` (newer)
- fallback: `d3.8xlarge` if you want large disk and can tolerate HDD speed

## Recommended sizes + credits/hr (Coiled)

Credits/hr are from `coiled.list_instance_types()` (not USD). Convert to
cost using your Coiled plan's $/credit rate.

- 4c / 8 GiB: `c5d.xlarge` (4 credits/hr) / `c6gd.xlarge` or `c7gd.xlarge` (4 credits/hr)
- 4c / 16 GiB: `m4.xlarge` (4 credits/hr) / `m7gd.xlarge` (4 credits/hr)
- 4c / 32 GiB: `i3.xlarge` (4 credits/hr) / `i4g.xlarge` (4 credits/hr)
- 8c / 32 GiB: `m5.2xlarge` (8 credits/hr) / `m7gd.2xlarge` (8 credits/hr)
- 8c / 64 GiB: `i3.2xlarge` (8 credits/hr) / `i4g.2xlarge` (8 credits/hr)
- 16c / 128 GiB: `i3.4xlarge` (16 credits/hr) / `i4g.4xlarge` or `r6gd.4xlarge` (16 credits/hr)
- 16c / 256 GiB: x86_64 none (don’t use f1.4xlarge) / `x2gd.4xlarge` or `x8g.4xlarge` (16 credits/hr)
- 32c / 256 GiB: `i4i.8xlarge` (32 credits/hr) / `i4g.8xlarge` or `r6gd.8xlarge` (32 credits/hr)

## Recommendations

- If you get an InstanceTypeError, set `worker_vm_types`/`scheduler_vm_types` explicitly
  to one of the types returned by `coiled.list_instance_types(...)`.
- Prefer memory-optimized families (x2gd/x8g) when you need large memory per worker.
- Prefer NVMe-backed families (i4i/i7i) when you expect heavy shuffle or spill.
- Avoid FPGA families (f1) unless you specifically need FPGA hardware.
- Keep the scheduler smaller than workers (2-4 vCPU, 8-16 GiB) unless you run heavy
  compute on the scheduler.
- If you need x86_64 only, avoid ARM types (x2gd/x8g). Use a compatible x86_64 type
  or lower the memory target so Coiled can recommend a modern x86_64 family.
