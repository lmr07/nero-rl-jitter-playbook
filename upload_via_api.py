#!/usr/bin/env python3
"""Resume this repository's publication through the GitHub Git Data API.

The token is read from the terminal and never stored. Blob progress is cached in
.git so interrupted uploads can be resumed without retransmitting large files.
"""

import argparse
import base64
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


REPO = "lmr07/nero-rl-jitter-playbook"
API = "https://api.github.com/repos/" + REPO
ROOT = Path(__file__).resolve().parent
INLINE_LIMIT = 128 * 1024


def git(*args, binary=False):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=True
    )
    return result.stdout if binary else result.stdout.decode().strip()


def local_tree():
    entries = {}
    for record in git("ls-tree", "-r", "-z", "main", binary=True).split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, sha = metadata.decode("ascii").split()
        if kind != "blob":
            raise RuntimeError("Unsupported Git entry: " + kind)
        entries[path.decode("utf-8")] = (mode, sha)
    return entries


def blob_sha(data):
    header = ("blob %d\0" % len(data)).encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


class Github:
    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def call(self, method, path, payload=None):
        url = API + path
        for attempt in range(1, 6):
            try:
                response = self.session.request(
                    method, url, json=payload, timeout=(15, 180)
                )
            except requests.RequestException as exc:
                if attempt == 5:
                    raise RuntimeError("Network error after 5 attempts: " + str(exc))
                print("Network retry %d/5: %s" % (attempt, path), flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            if 200 <= response.status_code < 300:
                return response.json()
            if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < 5:
                print("HTTP %d retry %d/5: %s" %
                      (response.status_code, attempt, path), flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            raise RuntimeError("GitHub API %s %s: HTTP %d: %s" %
                               (method, path, response.status_code,
                                response.text[:400]))
        raise RuntimeError("Unexpected API retry failure")


def progress_path():
    return Path(git("rev-parse", "--git-dir")) / "nero_api_upload_progress.json"


def load_progress(local_sha):
    path = progress_path()
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("repo") == REPO and data.get("local_head") == local_sha:
            return path, data
    return path, {"repo": REPO, "local_head": local_sha, "blobs": {}}


def save_progress(path, progress):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-local", action="store_true",
                        help="Check local files without GitHub access or a token")
    args = parser.parse_args()

    if git("status", "--porcelain"):
        raise RuntimeError("Working tree is not clean; commit local edits first")
    local_head = git("rev-parse", "main")
    wanted_tree = git("rev-parse", "main^{tree}")
    wanted = local_tree()
    if args.check_local:
        unique = {sha for _, sha in wanted.values()}
        print("Local main:", local_head)
        print("Local tree:", wanted_tree)
        print("Files:", len(wanted), "unique blobs:", len(unique))
        return

    token = getpass.getpass("GitHub fine-grained token (Contents: read/write): ").strip()
    if not token:
        raise RuntimeError("No token entered")
    api = Github(token)
    del token

    branch = api.call("GET", "/branches/main")
    remote_head = branch["commit"]["sha"]
    remote_tree_sha = branch["commit"]["commit"]["tree"]["sha"]
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                       remote_head, "main"]).returncode != 0:
        raise RuntimeError("Remote main is not an ancestor of local main; stop and inspect it")
    if remote_tree_sha == wanted_tree:
        print("GitHub already has the complete file tree:", wanted_tree)
        return

    remote_data = api.call("GET", "/git/trees/" + remote_tree_sha + "?recursive=1")
    if remote_data.get("truncated"):
        raise RuntimeError("GitHub returned a truncated remote tree")
    remote = {row["path"]: (row["mode"], row["sha"])
              for row in remote_data["tree"] if row["type"] == "blob"}
    known_shas = {sha for _, sha in remote.values()}
    changed = {path: item for path, item in wanted.items()
               if remote.get(path) != item}
    removed = sorted(set(remote) - set(wanted))
    print("Remote:", remote_head[:12], "local:", local_head[:12], flush=True)
    print("Changed files:", len(changed), "removed files:", len(removed), flush=True)

    cache_path, progress = load_progress(local_head)
    tree_elements = []
    uploaded = 0
    for index, (path, (mode, sha)) in enumerate(sorted(changed.items()), 1):
        entry = {"path": path, "mode": mode, "type": "blob"}
        if sha in known_shas or progress["blobs"].get(sha) == sha:
            entry["sha"] = sha
        else:
            data = git("cat-file", "blob", sha, binary=True)
            if blob_sha(data) != sha:
                raise RuntimeError("Local blob checksum mismatch: " + path)
            try:
                inline = data.decode("utf-8") if len(data) <= INLINE_LIMIT else None
            except UnicodeDecodeError:
                inline = None
            if inline is not None and "\0" not in inline:
                entry["content"] = inline
            else:
                payload = {"content": base64.b64encode(data).decode("ascii"),
                           "encoding": "base64"}
                result = api.call("POST", "/git/blobs", payload)
                if result["sha"] != sha:
                    raise RuntimeError("Uploaded blob SHA mismatch: " + path)
                progress["blobs"][sha] = sha
                save_progress(cache_path, progress)
                entry["sha"] = sha
                uploaded += 1
                print("Uploaded blob %d: %s (%d bytes)" %
                      (uploaded, path, len(data)), flush=True)
        tree_elements.append(entry)
        if index % 100 == 0:
            print("Prepared %d/%d tree entries" % (index, len(changed)), flush=True)

    for path in removed:
        tree_elements.append({"path": path, "mode": remote[path][0],
                              "type": "blob", "sha": None})
    result = api.call("POST", "/git/trees",
                      {"base_tree": remote_tree_sha, "tree": tree_elements})
    if result["sha"] != wanted_tree:
        raise RuntimeError("GitHub tree mismatch: got %s, expected %s. "
                           "Remote branch was not changed." %
                           (result["sha"], wanted_tree))
    print("Verified exact Git tree:", wanted_tree, flush=True)

    commit = api.call("POST", "/git/commits", {
        "message": "Complete Nero training and deployment bundle",
        "tree": wanted_tree,
        "parents": [remote_head],
    })
    final_sha = commit["sha"]
    api.call("PATCH", "/git/refs/heads/main",
             {"sha": final_sha, "force": False})
    check = api.call("GET", "/branches/main")
    if check["commit"]["sha"] != final_sha:
        raise RuntimeError("Branch verification failed after ref update")
    print("GitHub upload complete:", final_sha)
    print("GitHub tree matches local main:", wanted_tree)
    print("The local main commit history differs from the API commit; "
          "sync it after Git network access recovers.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print("Upload stopped:", exc, file=sys.stderr)
        sys.exit(1)
