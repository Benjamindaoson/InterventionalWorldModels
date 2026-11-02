"""Make a deterministic contract-only receipt. Never report it as physics evidence."""
from pathlib import Path
import json
from iwm.paired_replay import ReplaySpec, ToyBackend, paired_replay

if __name__ == "__main__":
    out = Path("artifacts/g0_toy_contract_receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = paired_replay(
        ToyBackend(),
        ReplaySpec(task_id="toy-action-difference",scene_id="toy-scene-v1",seed=11,
                   horizon=10,split="dev",backend_id="toy-contract-v1",
                   source_revision="local-standalone-smoke",
                   evidence_tier="TOY_CONTRACT_TEST"),
        actions=[[0.0,0.0],[0.5,0.0],[1.0,0.0]],
    )
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(out),"status":receipt["research_status"],
                      "branches":len(receipt["branches"]),
                      "receipt_sha256":receipt["receipt_sha256"]},indent=2))
