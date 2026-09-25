"""Terminate the Mem0-row BMA and confirm nothing is left."""
from qbraid_core.services.compute import ComputeClient
c = ComputeClient()
iid = open("/home/nick/mem0row/iid").read().strip()
try:
    c.terminate_bma_instance(iid); print("terminate requested:", iid)
except Exception as e:  # noqa: BLE001
    print("terminate raised:", type(e).__name__, str(e)[:200])
import time
for _ in range(20):
    left = c.list_bma_instances()
    if not left:
        break
    time.sleep(15)
print("list_bma_instances:", left)
print("balance:", c.get_credits_balance().get("qbraidCredits"))
