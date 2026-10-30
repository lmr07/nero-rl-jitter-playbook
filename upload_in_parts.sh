#!/usr/bin/env bash
# Push the prepared linear history one small commit at a time. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Working tree is not clean; commit or discard local edits before uploading.' >&2
    exit 1
fi

mapfile -t commits < <(git rev-list --reverse main)
read_remote() {
    local attempt value
    for attempt in 1 2 3; do
        if value=$(timeout 30s git ls-remote origin refs/heads/main | awk '{print $1}'); then
            printf '%s' "$value"
            return 0
        fi
        echo "Remote check failed (attempt $attempt/3)." >&2
        sleep 2
    done
    return 1
}

remote=$(read_remote)
for commit in "${commits[@]}"; do
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
    uploaded=false
    for attempt in 1 2 3; do
        if timeout --foreground --signal=INT --kill-after=5s 45s \
            git push origin "${commit}:refs/heads/main"; then
            remote=$commit
            uploaded=true
            break
        fi
        echo "Push failed or timed out (attempt $attempt/3); checking remote." >&2
        remote=$(read_remote) || exit 1
        if [[ -n "$remote" ]] && git merge-base --is-ancestor "$commit" "$remote"; then
            uploaded=true
            break
        fi
        sleep 2
    done
    if [[ "$uploaded" != true ]]; then
        echo 'Upload paused after three failed attempts; rerun this script later.' >&2
        exit 1
    fi
done

git update-ref refs/remotes/origin/main "$(git rev-parse main)"
git config branch.main.remote origin
git config branch.main.merge refs/heads/main
echo 'Upload complete. Check https://github.com/lmr07/nero-rl-jitter-playbook'
