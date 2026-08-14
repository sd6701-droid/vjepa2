#!/bin/bash

# --- macOS workaround -------------------------------------------------------
# The upstream repo tracks folders that differ only by case (vitG16 vs vitg16,
# vitG-384 vs vitg-384). On macOS's case-insensitive filesystem they collide:
# "git add" on them either dies with "will not add file alias" or silently
# stages the wrong twin. So on macOS we keep "git add" away from those folders
# entirely and stage the uppercase entries by exact path instead.
# Not needed on Linux (e.g. the cluster), where the folders don't collide.
if [ "$(uname)" = "Darwin" ]; then
  # 1) Pin the lowercase twins to their committed content and re-flag them as
  #    ignored (pulls and past "git add" runs can clobber both).
  git ls-tree -r HEAD \
    | awk '$4 ~ /^configs\/(eval_2_1\/vitg-384|train_2_1\/vitg16)\//{print $1","$3","$4}' \
    | while IFS= read -r spec; do
        git update-index --cacheinfo "$spec"
        git update-index --skip-worktree -- "${spec##*,}"
      done

  # 2) Stage everything EXCEPT the colliding folders.
  git add -- . \
    ':(exclude)configs/eval_2_1/vitG-384' ':(exclude)configs/eval_2_1/vitg-384' \
    ':(exclude)configs/train_2_1/vitG16' ':(exclude)configs/train_2_1/vitg16'

  # 3) Stage the uppercase entries explicitly by exact path.
  #    ("H " marks normal index entries, "S " the ignored lowercase ones.)
  git ls-files -v \
    | awk '/^H configs\/(eval_2_1\/vitG-384|train_2_1\/vitG16)\//{print substr($0, 3)}' \
    | while IFS= read -r f; do git update-index -- "$f"; done
else
  git add .
fi
# ----------------------------------------------------------------------------

# Commit with the message passed as the first argument (empty if none given)
git commit --allow-empty-message -m "${1:-}"

# Push to the current branch
git push
