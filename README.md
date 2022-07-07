Set your api key as you environment variable. If using linux run, 

echo "export OPENAI_API_KEY_TIM='yourkey'" >> ~/.zshrc
source ~/.zshrc


run experiments using ./scripts/run_exp.sh

convert experiment results into csv using 
 ./scripts/results_to_csv.sh


When you run the triplet task, you are likely to get a few prompts for which gpt gave undesired responses. Copy and paste this from the terminal to a file in the results dir so that you can correct these mistakes manually. These have been marked by 'fill manually' in the gpt_choice column in davinci_triplet_feature_list.csv 


When you run the feature listing task, there are some prompts for which gpt responds with an answer that's not yes/no. There are about 7 of them out of ~16000. For these the yes/no column in davinci_feature_list.csv has been populated with 'something went wrong'. Manually correct this before moving on.
