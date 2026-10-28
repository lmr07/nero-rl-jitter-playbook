#!/usr/bin/env bash
# Push the prepared linear history one small commit at a time. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Working tree is not clean; commit or discard local edits before uploading.' >&2
    exit 1
fi

mapfile -t commits < <(git rev-list --reverse main)
for commit in "${commits[@]}"; do
    remote=$(git ls-remote origin refs/heads/main | awk '{print $1}')
    if [[ -n "$remote" ]]; then
        if ! git cat-file -e "${remote}^{commit}" 2>/dev/null; then
            echo "Remote main points to an unknown commit: $remote" >&2
            exit 1
        fi
        if git merge-base --is-ancestor "$commit" "$remote"; then
            echo "Already uploaded: ${commit:0:12}"
            continue
        fi
        if ! git merge-base --is-ancestor "$remote" "$commit"; then
            echo "Remote main has diverged at $remote; refusing to overwrite it." >&2
            exit 1
        fi
    fi
    echo "Uploading: ${commit:0:12}"
    git push origin "${commit}:refs/heads/main"
done

git push -u origin main
echo 'Upload complete. Check https://github.com/lmr07/nero-rl-jitter-playbook'
