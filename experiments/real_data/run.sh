python save_data.py
python plot_tree_cover_covariates.py 

python tree_cover_exp.py --"num_neighbors"=1 --"lipschitz_bound"=0.2  --"num_threads"=36 --"num_seeds"=250 --"seeds_to_plot"=7 --"region"="south"
python tree_cover_exp.py --"num_neighbors"=1 --"lipschitz_bound"=0.2  --"num_threads"=36 --"num_seeds"=250 --"seeds_to_plot"=7 --"region"="west"

python tree_cover_exp_lipschitz.py --"num_neighbors"=1  --"num_threads"=36 --"num_seeds"=250 --"seeds_to_plot"=7 --"region"="south"
python tree_cover_exp_lipschitz.py --"num_neighbors"=1  --"num_threads"=36 --"num_seeds"=250 --"seeds_to_plot"=7 --"region"="south"
