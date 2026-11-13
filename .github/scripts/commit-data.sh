#!/usr/bin/env bash
#
# Commit whatever a collector wrote, to the data branch, and push it.
#
# WHY THE DATA BRANCH. Streamlit Cloud rebuilds the app when the branch it
# serves receives a push. With one branch, the hourly tracker would restart the
# public site every hour on a storm night: cold start, and every cached frame
# discarded, so the first reader after each reboot pays for the map, the
# scorecard and a full region rebuild. The risk was never mainly a bad commit,
# it is ANY commit. Data goes to a branch nothing deploys, and new numbers
# reach the page with no restart at all.
#
# The data branch is checked out AT data/, so its root holds forecasts/,
# outcomes/, hourly/, calls/ and prediction_history/ directly. The Python
# writes to ./data/... exactly as it does locally and knows nothing about any
# of this.
#
# ONE-TIME BOOTSTRAP, if the branch does not exist yet:
#
#   git worktree add --detach ../snow-bootstrap
#   cd ../snow-bootstrap
#   git checkout --orphan data
#   git rm -rf --cached . >/dev/null
#   find . -mindepth 1 -not -path './.git*' -delete
#   cp -r ../scricsnow/data/. .
#   git add -A && git commit -m "Start the data branch"
#   git push -u origin data
#   cd ../scricsnow && git worktree remove ../snow-bootstrap
#
set -euo pipefail

message="${1:?usage: commit-data.sh <message>}"
cd "$(dirname "$0")/../../data"

if [ -z "$(git status --porcelain)" ]; then
  echo "nothing changed; no commit"
  exit 0
fi

git config user.name "scricsnow bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -m "$message"

# Three workflows push to one branch and their schedules overlap on a busy
# morning, so a rejected push is expected rather than exceptional. Rebase and
# retry: every writer here is append-or-replace on its own rows, so replaying
# a commit on top of somebody else's is exactly right.
for attempt in 1 2 3 4 5; do
  if git push origin HEAD:data; then
    echo "pushed on attempt $attempt"
    exit 0
  fi
  echo "push rejected (attempt $attempt); rebasing onto origin/data"
  git pull --rebase --autostash origin data || true
  sleep $((attempt * 3))
done

echo "::error::could not push to the data branch after 5 attempts"
exit 1
