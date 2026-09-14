#!/bin/bash
RANK=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}

ANALYSIS_TYPE="$1"
shift

VTUNE="/share/apps/intel/oneapi/vtune/2025.0/bin64/vtune"
DEST_DIR="vtune_results/${ANALYSIS_TYPE}_rank_${RANK}"

# Rimuove la vecchia cartella del singolo rank se già present
rm -rf "$DEST_DIR" 2>/dev/null || true
exec "$VTUNE" -collect "$ANALYSIS_TYPE" -result-dir "$DEST_DIR" -data-limit=0 -- "$@"
