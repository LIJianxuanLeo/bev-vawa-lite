#!/usr/bin/env bash
# Gibson Habitat reactive-safety parameter sweep.
#
# Evaluates the Stage-C bev_vawa checkpoint against the Gibson val split
# under 3 alternative safety configs + the PIB-Nav default safety knobs
# (inherited through gibson.yaml), at 4 seeds × 100 episodes each.
#
# Prerequisites inside the Pod:
#   /root/data/runs/gibson/stage_c.pt            # trained checkpoint
#   /root/data/scene_datasets/gibson/*.glb       # Gibson habitat 4+ pack
#   /root/data/datasets/pointnav/gibson/v2/...   # pointnav_gibson_v2
#   conda env `habitat` with hsim 0.3.3 + project
#
# Output: rows in /root/data/runs/gibson/safety_sweep.csv with
# method-name encoding the variant.
#
# Total runtime: ~30 min, ~¥1 billable at ¥1.89/h.

set -euxo pipefail
cd /root/data/bev-vawa-lite
export PYTHONPATH=$PWD
HP=/root/miniconda3/envs/habitat/bin

# Build val scene list
VAL_SCENES=()
for f in /root/data/datasets/pointnav/gibson/v2/val/content/*.json.gz; do
    name=$(basename "$f" .json.gz)
    glb="/root/data/scene_datasets/gibson/${name}.glb"
    [ -f "$glb" ] && VAL_SCENES+=("$glb")
done
echo "val scenes: ${#VAL_SCENES[@]}"

OUT_CSV=/root/data/runs/gibson/safety_sweep.csv
rm -f "$OUT_CSV"

# Variant matrix: config_path, label_tag
VARIANTS=(
    "configs/habitat/gibson.yaml|default"
    "configs/habitat/gibson_safety_tight.yaml|tight"
    "configs/habitat/gibson_safety_tighter.yaml|tighter"
    "configs/habitat/gibson_safety_sideheavy.yaml|sideheavy"
)

for entry in "${VARIANTS[@]}"; do
    CFG="${entry%%|*}"
    TAG="${entry##*|}"
    for SEED in 12345 42 7 31337; do
        echo "==== variant=${TAG} seed=${SEED} ====" $(date)
        $HP/python scripts/eval_habitat.py \
            --config "$CFG" \
            --scenes "${VAL_SCENES[@]}" \
            --policy bev_vawa --ckpt /root/data/runs/gibson/stage_c.pt \
            --n-episodes 100 --seed $SEED \
            --method-name "BEV-VAWA Gibson safety=${TAG} seed=${SEED}" \
            --safety \
            --results "$OUT_CSV"
    done
done

echo "==== SAFETY SWEEP DONE ====" $(date)
cat "$OUT_CSV"
