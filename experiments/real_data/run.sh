python3 tree_cover_exp.py --"noise_std"=0.1 --"num_neighbors"=1 --"lipschitz_bound"=0.2  --"num_threads"=20 --"num_seeds"=20 --"seeds_to_plot"=7
python3 tree_cover_exp_lipschitz.py --"noise_std"=0.1 --"num_neighbors"=1 --"bandwidth"=0.1  --"num_threads"=20 --"num_seeds"=250 --"seeds_to_plot"=7
python3 plot_tree_cover_covariates.py   