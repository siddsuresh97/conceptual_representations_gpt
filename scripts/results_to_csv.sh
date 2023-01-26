# python src/results_to_csv.py --exp_name 'feature_listing' --dataset_name 'reptile_tool' --model 'ada'




# python src/results_to_csv.py --exp_name 'feature_listing' --dataset_name 'reptile_tool' --model 'davinci'
# python src/results_to_csv.py --exp_name 'triplet' --dataset_name 'reptile_tool' --model 'davinci'



# python src/results_to_csv.py --exp_name 'triplet' --dataset_name 'reptile_tool' --model 'davinci'


# python src/results_to_csv.py --exp_name 'triplet' --dataset_name 'social_categories' --model 'davinci'



#10/3/22 - running with temperature 0 All previous experiments were run with temperature 0.7
# python src/results_to_csv.py --exp_name 'feature_listing' --dataset_name 'reptile_tool' --model 'davinci' --temperature 0.0

#1/24/23 - running triplet with temperature 0 All previous experiments were run with temperature 0.7
python src/results_to_csv.py --exp_name 'triplet' --dataset_name 'reptile_tool' --model 'davinci' --temperature 0.0