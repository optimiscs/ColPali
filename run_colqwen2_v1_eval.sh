#!/usr/bin/env bash
set -u

cd /home/moxu/MMRAG/otherExp/colpali || exit 1

LOG_PATH=/tmp/colqwen2_v1_eval.log
PYTHON_BIN=/home/moxu/miniconda3/envs/colpali/bin/python

{
  echo "START $(date -Iseconds)"
  echo "PWD $(pwd)"
  echo "PYTHON ${PYTHON_BIN}"
} > "${LOG_PATH}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" eval_baseline.py \
  --baselines vidore/colqwen2-v1.0 \
  --show-progress-bar >> "${LOG_PATH}" 2>&1

STATUS=$?
echo "EXIT ${STATUS} $(date -Iseconds)" >> "${LOG_PATH}"
exit "${STATUS}"
