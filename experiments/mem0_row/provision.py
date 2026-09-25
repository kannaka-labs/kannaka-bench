"""Provision ONE gpu-rtx-4090 BMA for the scored Mem0 row. Cutoff set before waiting.
Budget: coordinator cap ~900 credits; 4090 bills 1.45 credits/min -> max_session 590 min = 856 credits."""
import json, sys, time
from qbraid_core.services.compute import ComputeClient
c = ComputeClient()
dump = lambda o: o.model_dump() if hasattr(o, "model_dump") else (vars(o) if hasattr(o, "__dict__") else o)
inst = None
for attempt in range(30):                     # 4090 only (the calibrated recipe); retry capacity for ~30 min
    try:
        inst = c.provision_bma_instance("gpu-rtx-4090"); break
    except Exception as e:  # noqa: BLE001
        print(f"attempt {attempt}: {str(e)[:120]}", flush=True); time.sleep(60)
if inst is None:
    print("NO 4090 CAPACITY"); sys.exit(4)
d = dump(inst); iid = d.get("instance_id") or d.get("instanceId")
print("provisioned:", iid, flush=True)
if not iid:
    print(json.dumps(d, default=str)[:800]); sys.exit(2)
open("/home/nick/mem0row/iid", "w").write(iid)
try:
    c.update_bma_cutoff(iid, auto_stop_idle_minutes=30, max_session_minutes=590)
except Exception as e:  # noqa: BLE001
    print("cutoff call raised (may still have applied):", type(e).__name__, flush=True)
try:
    got = dump(c.get_bma_instance(iid))
    print("cutoff now:", {k: v for k, v in got.items() if any(s in k.lower() for s in ("idle", "runtime", "session", "cutoff", "rate", "credit"))}, flush=True)
except Exception as e:  # noqa: BLE001
    print("re-read failed:", type(e).__name__, str(e)[:200], flush=True)
t0 = time.time()
try:
    c.wait_for_bma_instance(iid, timeout=1200)
except Exception as e:  # noqa: BLE001
    print("wait raised:", type(e).__name__, str(e)[:300], flush=True)
print(f"running after {time.time() - t0:.0f}s", flush=True)
c.configure_ssh_for_instance(iid)
alias = c.bma_ssh_alias(iid)
open("/home/nick/mem0row/alias", "w").write(alias)
print("ALIAS:", alias, "IID:", iid, "at", time.strftime("%H:%M:%SZ", time.gmtime()), flush=True)
