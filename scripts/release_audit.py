"""Read-only scan of every reachable Git blob before public branch publication."""
from __future__ import annotations

import json
import re
import subprocess

PATTERNS = {
    "private_key": rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
    "github_token": rb"(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}",
    "openai_key": rb"\bsk-[A-Za-z0-9_-]{20,}\b",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "password_assignment": rb"(?i)(?:password|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]",
}
SENSITIVE_NAME = re.compile(r"(^|/)(?:\.env(?:\..*)?|id_(?:rsa|ed25519)|.*\.(?:pem|p12|pfx|key))$", re.IGNORECASE)
WEIGHTS_OR_DATA = re.compile(r"\.(?:pt|pth|ckpt|npz|mp4)$", re.IGNORECASE)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def main() -> None:
    objects = git("rev-list", "--objects", "--all").decode("utf-8", errors="replace").splitlines()
    seen = set()
    findings = []
    large = []
    scanned = 0
    for line in objects:
        sha, _, path = line.partition(" ")
        if not path or sha in seen:
            continue
        seen.add(sha)
        if git("cat-file", "-t", sha).strip() != b"blob":
            continue
        scanned += 1
        size = int(git("cat-file", "-s", sha))
        if size > 10_000_000 or WEIGHTS_OR_DATA.search(path):
            large.append({"path": path, "object": sha, "bytes": size})
        if SENSITIVE_NAME.search(path):
            findings.append({"rule": "sensitive_filename", "path": path, "object": sha})
        if size > 10_000_000:
            continue
        data = git("cat-file", "blob", sha)
        for name, pattern in PATTERNS.items():
            if re.search(pattern, data):
                findings.append({"rule": name, "path": path, "object": sha})
    print(json.dumps({"reachable_blobs_scanned": scanned, "sensitive_findings": findings,
                      "large_or_binary_artifacts": large}, indent=2))


if __name__ == "__main__":
    main()
