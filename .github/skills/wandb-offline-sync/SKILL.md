# W&B Offline Run Tracking (BNL cluster)

On BNL (`hostname` contains `"bnl"`), jobs run with `wandb offline`. Each Python training call creates a new `offline-run-<timestamp>-<id>/` directory under `${wandb_run_dir}` (typically `${REPO_ROOT}/coral/wandb/wandb/`). Sweep scripts must record these paths so they can be synced later.

## Pattern used in sweep scripts

1. **Before** calling `python inr_sample/single_image_inr.py`, snapshot existing offline run dirs:
   ```bash
   before_runs="$(find "$wandb_run_dir" -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' | sort)"
   ```

2. **After** the Python call, snapshot again and diff to find the new dir:
   ```bash
   after_runs="$(find "$wandb_run_dir" -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' | sort)"
   new_runs="$(comm -13 <(printf "%s\n" "$before_runs") <(printf "%s\n" "$after_runs") || true)"
   ```

3. **Record** each new dir (full path) into a TSV sync-path log:
   ```bash
   # Format: run_name<TAB>offline_run_dir
   echo -e "${run_name}\t${wandb_run_dir}/${run_dir_name}" >> "$SYNC_PATH_LOG"
   # Fallback when detection fails:
   echo -e "${run_name}\tNOT_FOUND" >> "$SYNC_PATH_LOG"
   ```

4. **After all runs**, automatically generate a sync shell script:
   ```bash
   sync_script="${REPO_ROOT}/outputs/wandb_sync_<experiment_name>.sh"
   {
     echo "#!/bin/bash"
     echo "conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling"
     echo ""
     while IFS=$'\t' read -r rname rpath; do
       [[ "$rname" == "#"* ]] && continue
       if [ "$rpath" = "NOT_FOUND" ]; then
         echo "# SKIPPED (not found): $rname"
       else
         echo "wandb sync \"$rpath\""
       fi
     done < "$SYNC_PATH_LOG"
   } > "$sync_script"
   chmod +x "$sync_script"
   ```

5. **After the job completes**, run the sync script on a node with internet access:
   ```bash
   bash outputs/wandb_sync_<experiment_name>.sh
   ```

## Reference implementations
- `script/inr_sample/compare-adaptive-topk-grid-IC2.sh` — original reference; uses `SYNC_PATH_LOG` written alongside a sweep of adaptive grid configs.
- `script/inr_sample/compare_coefficient.sh` — applies the same pattern inside `run_one()` when `is_bnl=true`; auto-generates `outputs/wandb_sync_compare_coefficient.sh` at the end.

## Key variables (BNL branch)
| Variable | Purpose |
|---|---|
| `wandb_base_dir` | Root W&B dir (`${REPO_ROOT}/coral/wandb` by default) |
| `wandb_run_dir` | Where offline run dirs are created (`$wandb_base_dir/wandb`) |
| `SYNC_PATH_LOG` | TSV log of `run_name<TAB>path` for all runs in the sweep |
| sync script | Auto-generated `.sh` that activates conda and calls `wandb sync` for each path |
