#!/bin/bash
# run_rc_hard_refusal_pipeline.sh
# Master chain for the new refusal_compliance_full_hard_refusal dataset: build (phase A)
# strictly followed by extract (phase B), never combined -- run this only once the 8 GPUs
# are actually free (e.g. after run_generation_rc_full.sh's ALL_RC_FULL_SWEEP_DONE).
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace

bash scripts_claude/run_build_rc_hard_refusal.sh
build_rc=$?
if [[ $build_rc -ne 0 ]]; then
    echo "!!! BUILD PHASE FAILED (exit ${build_rc}) -- not proceeding to extraction."
    exit 1
fi

bash scripts_claude/run_extract_rc_hard_refusal.sh
extract_rc=$?
if [[ $extract_rc -ne 0 ]]; then
    echo "!!! EXTRACT PHASE FAILED (exit ${extract_rc})."
    exit 1
fi

echo ""
echo "ALL_RC_HARD_REFUSAL_PIPELINE_DONE"
