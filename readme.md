2d burgers code generation: /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/utils/data/generate_burgers2d.py

paper position: /pscratch/sd/g/gzhao27/Papers/INR-Refined-Grid-Sampling-Neurips-Workshop/ICLR_INR_RGS.tex


python utils/data/generate_burgers2d.py \
  --H 1024 --W 1024 --T_steps 100 \
  --dt 0.001 --substeps 100 \
  --out_dir /pscratch/sd/g/gzhao27/INR/data/2D_Burgers_1024

python utils/data/generate_burgers2d.py   --H 256 --W 256 --T_steps 100 --dt 0.001   --n_modes 20   --ic_decay_power 1.2   --ic_highfreq_boost 0.4   --ic_target_rms 1.0