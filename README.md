Set your api key as you environment variable. If using linux run, 

echo "export OPENAI_API_KEY_TIM='yourkey'" >> ~/.zshrc
source ~/.zshrc


run experiments using ./scripts/run_exp.sh

convert experiment results into csv using 
 ./scripts/generate_feature_list_from_gpt.sh


When you run the triplet task, you are likely to get a few prompts for which gpt gave undesired responses. Copy and paste this from the terminal to a file in the results dir so that you can correct these mistakes manually.
