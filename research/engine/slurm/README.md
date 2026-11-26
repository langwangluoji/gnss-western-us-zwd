# Slurm campaign scripts

`gnss_field.orchestration.write_campaign` creates phase array scripts, immutable task JSON and a campaign manifest in the chosen campaign directory. It creates the log directory before any submission. Task argv stays in JSON for the worker dispatcher. Shell interpolation is confined to the Slurm array index.

The default GPU profile requests one L40, seven CPUs, 64 GB and at most 24 hours. CPU work and `afterany` summaries use the configured `cpu_partition`, defaulting to `intel`. The bootstrap pack, audit and doctor jobs request at most two CPUs, 8 GB and 15 minutes. Summaries request one CPU, 4 GB and 10 minutes. Each array uses `%2`. The coordinator releases one active phase at a time and adds an `afterok` barrier. Arrays use `--signal=USR1@180` and launch the Python worker with `srun` so the signal reaches job steps.

The requested eight-CPU design was corrected to seven after checking the current Tongji guide. The guide sets a maximum of seven CPUs per GPU and includes an eight-CPU rejection example. Runtime configurations requesting eight CPUs are rejected by the coordinator. [Tongji job submission guide](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/quickStart/jobSubmit)

The bootstrap path accepts a verified Tongji account/partition/static resource preflight with an explicit benchmark cap up to four L40 card hours. CUDA and driver checks run inside that allocated benchmark. Formal execution requires the driver result and a budget lock whose plan, data, code and protocol SHA256 values match the campaign. Windows execution and a local RTX profile cannot unlock the real Slurm adapter.

Each generated GPU array time limit is bounded by its shortest member reservation and the runtime segment limit. Every attempt reserves its declared card hours before submission. Failed and paused attempts retain that reservation. The default formal DAG allows two attempts per task. The initial four-hour benchmark allows one. Budget locks reserve the full declared attempt count.

Scripts load `/share/apps/miniconda3/etc/profile.d/conda.sh` and activate the configurable `gnss-field` environment. Persistent artifacts and single-owner study journals should reside in persistent storage. [Tongji Conda guide](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/software/conda)

`estimate_gpu_cost` provides GPU-only estimates using L40 at 2 yuan per card hour and A800 at 4 yuan per card hour. CPU billing is left to the platform account statement. [Tongji charging guide](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/platformDes/charge)
