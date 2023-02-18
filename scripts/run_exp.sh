# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'feature_listing' --feature_list_fname 'GPT_3_feature_df - Sheet1.csv' --model 'ada'

# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'feature_listing' --feature_list_fname 'GPT_3_feature_df - Sheet1.csv' --model 'davinci'

# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'triplet' --feature_list_fname 'triplets.pkl' --model 'davinci'


#  python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'triplet' --feature_list_fname 'triplets_reptile_tools.pkl' --model 'davinci'


# python src/prompt_gpt.py --dataset_name 'social_categories' --exp_name 'triplet' --feature_list_fname 'triplets_social_categories.pkl' --model 'davinci'


#10/3/22 - running with temperature 0 All previous experiments were run with temperature 0.7
# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'feature_listing' --feature_list_fname 'GPT_3_feature_df - Sheet1.csv' --model 'davinci' --temperature 0

#1/24/23 - running triplet with temperature 0 All previous experiments were run with temperature 0.7
# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'triplet' --feature_list_fname 'triplets_reptile_tools.pkl' --model 'davinci' --temperature 0

# #2/1/23 - running with flan temperature 0. Though temerature doesn't matter for flan, not sure why
# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'feature_listing' --feature_list_fname 'GPT_3_feature_df - Sheet1.csv' --model 'flan' --temperature 0


#2/8/23 - running with flan temperature 0. Though temerature doesn't matter for flan, not sure why
# python src/prompt_gpt.py --dataset_name 'reptile_tool' --exp_name 'triplet' --feature_list_fname 'triplets_reptile_tools.pkl' --model 'flan' --temperature 0


# #2/11/23 - run experiments to generate prompts for leuven norms to test on flan
# accelerate launch src/prompt_gpt.py --exp_name 'leuven_prompts_answers' --dataset_dir 'iclr/data/leuven' --model 'flan' --temperature 0 --results_dir 'iclr/data/leuven'


# #2/15/23 - run experiments with feature voerlap across animals and artifacts
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/prompt_gpt.py --exp_name 'leuven_prompts_answers' --dataset_dir 'iclr/data/leuven' --model 'flan' --temperature 0 --results_dir 'iclr/data/leuven'


#2/17/23 - run experiments with feature voerlap across animals and artifacts self consistency
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/prompt_gpt.py --exp_name 'leuven_prompts_answers' --dataset_dir 'iclr/data/leuven' --model 'flan' --temperature 0 --results_dir 'iclr/data/leuven' --exp_name 'self_consistency'



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=24 python -m torch.distributed.launch --nproc_per_node=8 src/prompt_gpt.py --exp_name 'leuven_prompts_answers' --dataset_dir 'iclr/data/leuven' --model 'flan' --temperature 0 --results_dir 'iclr/data/leuven'
