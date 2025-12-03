# Instructions to preprocess Törnberg2024partyaffiliation

- Just run code.R script, it automatically downloads data from github and gives: 1) processed_data, which contain data for all countries; and 2) country_prompts, which is the country specific LLM prompts used by Törnberg. (The exact prompt was not given, only a general template. I wrote a script that almost re-creates prompts using the template - they are almost exactly similar as the main ones I think)

- If data needs to be country splitted, it should be easy to fix.

- The outcome variable is party affiliation. These are country specific. Perhaps we should group parties by left/right position. Then one could see if Left or Right parties are over-estimated in different countries.

Calculate expert annotators agreement (Krippendorff's Alpha) with `python -m data.ideology_tweets.calculate_expert_annotation_agreement`