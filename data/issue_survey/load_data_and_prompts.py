from data.data_utils import MyDataLoader
import copy


class IssueSurveyDataset(MyDataLoader):
    def __init__(self, data_directory='issue_survey', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "1": "Europe",
            "2": "Immigration",
            "3": "Economy-general",
            "4": "Economy-personal",
            "5": "Unemployment",
            "6": "Taxation",
            "7": "Debt/deficit",
            "8": "Inflation",
            "9": "Living costs",
            "10": "Health",
            "11": "Coronavirus",
            "12": "COVID-economy",
            "13": "Terrorism",
            "14": "Poverty",
            "15": "Inequality",
            "16": "Housing",
            "17": "Environment",
            "18": "Education",
            "19": "Welfare",
            "20": "Austerity",
            "21": "Social care",
            "22": "Transport/infrastructure",
            "23": "Pensions/ageing",
            "24": "Pol-neg",
            "25": "Partisan-neg",
            "26": "Societal divides",
            "27": "Morals",
            "28": "National identity, goals-loss",
            "29": "Racism/discrimination",
            "30": "Asylum",
            "31": "Crime",
            "32": "Foreign affairs",
            "33": "War",
            "34": "Defence",
            "35": "Pol values-authoritarian",
            "36": "Pol values-liberal",
            "37": "Gender/sexuality/family",
            "38": "Pol values-right",
            "39": "Pol values-left",
            "40": "Election outcome",
            "41": "Constitutional",
            "42": "International trade",
            "43": "Devolution",
            "44": "Scot-independence",
            "45": "Foreign emergency",
            "46": "Domestic emergency",
            "47": "Referendum unspecified",
            "48": "Other",

        }
        dict_with_mapping_options_no_numbers = {
            'Europe': 'Europe',
            'Immigration': 'Immigration',
            'Economy-general': 'Economy-general',
            'Economy-personal': 'Economy-personal',
            'Unemployment': 'Unemployment',
            'Taxation': 'Taxation',
            'Debt/deficit': 'Debt/deficit',
            'Inflation': 'Inflation',
            'Living costs': 'Living costs',
            'Health': 'Health',
            'Coronavirus': 'Coronavirus',
            'COVID-economy': 'COVID-economy',
            'Terrorism': 'Terrorism',
            'Poverty': 'Poverty',
            'Inequality': 'Inequality',
            'Housing': 'Housing',
            'Environment': 'Environment',
            'Education': 'Education',
            'Welfare': 'Welfare',
            'Austerity': 'Austerity',
            'Social care': 'Social care',
            'Transport/infrastructure': 'Transport/infrastructure',
            'Pensions/ageing': 'Pensions/ageing',
            'Pol-neg': 'Pol-neg',
            'Partisan-neg': 'Partisan-neg',
            'Societal divides': 'Societal divides',
            'Morals': 'Morals',
            'National identity, goals-loss': 'National identity, goals-loss',
            'Racism/discrimination': 'Racism/discrimination',
            'Asylum': 'Asylum',
            'Crime': 'Crime',
            'Foreign affairs': 'Foreign affairs',
            'War': 'War',
            'Defence': 'Defence',
            'Pol values-authoritarian': 'Pol values-authoritarian',
            'Pol values-liberal': 'Pol values-liberal',
            'Gender/sexuality/family': 'Gender/sexuality/family',
            'Pol values-right': 'Pol values-right',
            'Pol values-left': 'Pol values-left',
            'Election outcome': 'Election outcome',
            'Constitutional': 'Constitutional',
            'International trade': 'International trade',
            'Devolution': 'Devolution',
            'Scot-independence': 'Scot-independence',
            'Foreign emergency': 'Foreign emergency',
            'Domestic emergency': 'Domestic emergency',
            'Referendum unspecified': 'Referendum unspecified',
            'Other': 'Other',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping no numbers': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_no_numbers}
            },
        }

    def get_reported_results(self):
        return []


    def get_dataset_specific_groups(self, df):
        return \
            self.split_by_column_with_n_most_frequent_values(df, "gender") + \
            self.split_by_column_with_n_most_frequent_values(df, "age group") + \
            self.split_by_column_with_n_most_frequent_values(df, "highest qualification") + \
            self.split_by_column_with_n_most_frequent_values(df, "education level") + \
            self.split_by_column_with_n_most_frequent_values(df, "gross personal income") + \
            self.split_by_column_with_n_most_frequent_values(df, "gross household income") + \
            self.split_by_column_with_n_most_frequent_values(df, "social grade") + \
            self.split_by_column_with_n_most_frequent_values(df, "which of these applies to you") + \
            self.split_by_column_with_n_most_frequent_values(df, "what kind of organisation do you work for") + \
            self.split_by_column_with_n_most_frequent_values(df, "do you own or rent the home in which you live") + \
            self.split_by_column_with_n_most_frequent_values(df, "what is your current marital or relationship status") + \
            self.split_by_column_with_n_most_frequent_values(df, "country") + \
            self.split_by_column_with_n_most_frequent_values(df, "government office region") + \
            self.split_by_column_with_n_most_frequent_values(df, "new parliamentary constituency 2024") + \
            self.split_by_column_with_n_most_frequent_values(df, "do you regard yourself as belonging to any particular religion and if so to w") + \
            self.split_by_column_with_n_most_frequent_values(df, "country of birth") + \
            self.split_by_column_with_n_most_frequent_values(df, "to which of these groups do you consider you belong") + \
            self.split_by_column_with_n_most_frequent_values(df, "are your day to day activities limited because of a health problem or disabilit") + \
            self.split_by_column_with_n_most_frequent_values(df, "which of the following best describes your sexuality") + \
            self.split_by_column_with_n_most_frequent_values(df, "party identification") + \
            self.split_by_column_with_n_most_frequent_values(df, "strength of party identification") + \
            self.split_by_column_with_n_most_frequent_values(df, "general election vote intention recalled vote in post election waves") + \
            self.split_by_column_with_n_most_frequent_values(df, "2019 ge vote choice") + \
            self.split_by_column_with_n_most_frequent_values(df, "2017 ge vote choice") + \
            self.split_by_column_with_n_most_frequent_values(df, "eu referendum vote earliest recorded") + \
            self.split_by_column_with_n_most_frequent_values(df, "eu referendum turnout earliest recorded") + \
            self.split_by_column_with_n_most_frequent_values(df, "attention to politics") + \
            self.split_by_column_with_n_most_frequent_values(df, "leftright position self") + \
            self.split_by_column_with_n_most_frequent_values(df, "redistribution scale self") + \
            self.split_by_column_with_n_most_frequent_values(df, "personal economic retrospective evaluation household") + \
            self.split_by_column_with_n_most_frequent_values(df, "general economic retrospective evaluation country") + \
            self.split_by_column_with_n_most_frequent_values(df, "british identity scale") + \
            self.split_by_column_with_n_most_frequent_values(df, "scotland q only scottish identity scale") + \
            self.split_by_column_with_n_most_frequent_values(df, "wales q only welsh identity scale") + \
            self.split_by_column_with_n_most_frequent_values(df, "english identity scale") + \
            self.split_by_column_with_n_most_frequent_values(df, "european integration scale 1990s version self") + \
            self.split_by_column_with_n_most_frequent_values(df, "vote intention in referendum on eu membership") + \
            self.split_by_column_with_n_most_frequent_values(
                df, "which daily newspaper do you read most often")

    def get_cols_to_stratify(self):
        return [
            "ground_truth",
            # columns used for grouping
            ("gender", 10),
            ("country", 10),
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the most important issue mentioned in the following response falls under the category {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short] [original]',
                'compatible_output_mapping': [
                    'default output mapping no numbers'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix C in https://journals.sagepub.com/doi/suppl/10.1177/20531680241231468/suppl_file/sj-pdf-1-rap-10.1177_20531680241231468.pdf
                        "content": """Here is an open-ended responses from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to the open ended text response, returning only the most relevant label.
health: include NHS
education
pol-neg: complaints about politics, the system, the media, corruption where no politician or party is mentioned
partisan-neg: use this code for negative statements about a party or politician
societal divides
morals: including abortion, complaints about religiosity and society's character
national identity, goals-loss: ethnonationalism
racism/discrimination
welfare
terrorism
immigration: include overpopulation. do not include refugees
asylum
crime
europe: including Brexit
election outcome
constitutional: UK constitutional issues except brexit, devolution and scottish independence. Include electoral system here.
international trade
devolution
scot-independence
foreign affairs: not including war
war: including Ukraine, Russia and Syria include desire for peace
defence
foreign emergency: short-term not including war
domestic emergency: short-term e.g. Flooding, storms and Grenfell Tower but not COVID
economy-general
economy-personal
unemployment
taxation
debt/deficit: national/government debt not personal debt
inflation: price rises
living costs: including energy crisis
poverty
austerity: cuts and lack of spending on services
inequality
housing: including homelessness
social care: including care for elderly, adults, disabled and children
transport/infrastructure
pensions/ageing
environment
pol values-authoritarian: pro-authoritarian responses or anti-socially liberal responses that don't fit better anywhere else: e.g. complaints about political correctness, LGBT people, wokeness etc
pol values-liberal: pro- socially liberal responses or anti-authoritarian responses that don't fit better anywhere else e.g. responses that are pro women's rights, LGBT people, free speech and other minorities or responses that are concerned about authoritarian groups
pol values-right: pro-economic right wing or anti-left wing responses that don't fit better anywhere else
pol values-left: pro-economic left wing or anti-right wing responses that don't fit better anywhere else
Referendum unspecified: where you can't tell whether response is about Brexit or indyref
coronavirus: covid but not its economic impacts
covid-economy
gender/sexuality/family
other: for responses that do not fit into any other category

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the label for the first issue mentioned.

Code this case: a bad economy
economy-general

Code this case: immergration
immigration

Code this case: climate change and unemployment
environment,unemployment
Please only return one code per response (if multiple match use the first issue listed). The correct response should be:
environment

Code this case: {text}"""
                    }
                ],
            },
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """I will provide you with an OPEN-ENDED TEXT RESPONSES from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to the OPEN-ENDED TEXT RESPONSE, returning only the most relevant category.

Categories:
- "1": Europe
- "2": Immigration
- "3": Economy-general
- "4": Economy-personal
- "5": Unemployment
- "6": Taxation
- "7": Debt/deficit
- "8": Inflation
- "9": Living costs
- "10": Health
- "11": Coronavirus
- "12": COVID-economy
- "13": Terrorism
- "14": Poverty
- "15": Inequality
- "16": Housing
- "17": Environment
- "18": Education
- "19": Welfare
- "20": Austerity
- "21": Social care
- "22": Transport/infrastructure
- "23": Pensions/ageing
- "24": Pol-neg
- "25": Partisan-neg
- "26": Societal divides
- "27": Morals
- "28": National identity, goals-loss
- "29": Racism/discrimination
- "30": Asylum
- "31": Crime
- "32": Foreign affairs
- "33": War
- "34": Defence
- "35": Pol values-authoritarian
- "36": Pol values-liberal
- "37": Gender/sexuality/family
- "38": Pol values-right
- "39": Pol values-left
- "40": Election outcome
- "41": Constitutional
- "42": International trade
- "43": Devolution
- "44": Scot-independence
- "45": Foreign emergency
- "46": Domestic emergency
- "47": Referendum unspecified
- "48": Other

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the category for the first issue mentioned.

OPEN-ENDED TEXT RESPONSE: a bad economy
Answer: 4 economy-general

OPEN-ENDED TEXT RESPONSE: immergration
Answer: 2 immigration

OPEN-ENDED TEXT RESPONSE: climate change and unemployment
Answer: 18 environment

Now output the category of the following OPEN-ENDED TEXT RESPONSE. Output only a number.

OPEN-ENDED TEXT RESPONSE: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix C in https://journals.sagepub.com/doi/suppl/10.1177/20531680241231468/suppl_file/sj-pdf-1-rap-10.1177_20531680241231468.pdf
                        "content": """I will provide you with an OPEN-ENDED TEXT RESPONSES from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to the OPEN-ENDED TEXT RESPONSE, returning only the most relevant category.

Categories:
- "1": Europe (Responses referencing Brexit, the EU and the EU referendum, including consequences of Brexit like "Industries leaving UK due to brexit")
- "2": Immigration (Responses mentioning immigration or related issues, except asylum/refugees which are coded separately. Includes "immigration", "too many migrants", "Illegal immigration", "Uncontrolled immigration", "control of our borders")
- "3": Economy-general (Responses referring to the economy in general, whether by saying "economy" or related terms like "recession" or "Financial stability". Includes generic "supply chain issues" and strikes/labour relations)
- "4": Economy-personal (Responses focusing specifically on respondent's own economic circumstances or family, like "not being able to pay my bills", "family finances", "christmas debt")
- "5": Unemployment (Issues relating to the labour market, including "unemployment", "jobs", "work", work conditions like "Zero hours contracts", and labour shortages)
- "6": Taxation (Responses mentioning "taxation" generally, "high taxes" or specific taxes like "Inheritance Tax", plus "Tax avoidance" and related terms)
- "7": Debt/deficit (References to public debt or "the deficit", like "National debt", "Government debt", "Debt Crisis", public waste or "overspending by Government")
- "8": Inflation (Responses focusing directly on inflation, like "inflation and interest rate rise")
- "9": Living costs (Responses focused on cost of living, including "cost of living", prices of essentials like "price of food", "cost of gas and electric", "Utilities", wage impacts like "low pay and rising prices")
- "10": Health (Responses referring to the health service itself, whether directly or referencing threats to it: "NHS", "Health Service", "NHS crisis", "NHS underfunding", "Privatisation of NHS". Excludes Covid-19 specific responses)
- "11": Coronavirus (All references to Covid-19 from wave 20 onwards, except economic effects. Includes "Coronavirus pandemic", "Covid-19", indirect references to effects/response/aftermath, and conspiracy theories about Covid)
- "12": COVID-economy (Concern about impact of Covid-19 pandemic and policy response on the economy, like "Economy during coronavirus", "Business failure due covid19", "Furlough scheme/covid")
- "13": Terrorism (Responses mentioning terrorism directly, "extremism", terrorist group names like "ISIS", or specific terrorist acts like "Manchester bombing")
- "14": Poverty (Responses directly referencing poverty or related topics like "food banks" or "People suffering below the breadline")
- "15": Inequality (Mentions of "inequality" and related concepts like "the wealth divide", "massively widening gap between rich and poor", "Creating a fair society", regional inequality)
- "16": Housing (All references to housing, including "Lack of affordable housing", "house prices", "affordable social housing", "Housing bubble", and homelessness unless explicitly poverty-related)
- "17": Environment (References to environment directly, climate change and consequences, like "Global warming", "Confronting climate change", "Environmental degradation", "Sustainability". Includes climate change challenges and single word "energy"/"food"/"fuel" responses)
- "18": Education (Apart from "education", includes related responses like "poor standards of education", "underfunding in education", "skill shortages", "university tuition fee", "state of schools")
- "19": Welfare (Responses mentioning welfare system or benefits generally, including "benefits", "Welfare state", "ATOS", whether positive or negative about welfare state)
- "20": Austerity (References to "austerity" directly, or cuts to public services more generally, like "Cuts", "Public Services being destroyed", "cut backs at local government level")
- "21": Social care (Responses specifically mentioning social care or related topics like "Care for the elderly", other state-funded care, carer's allowance)
- "22": Transport/infrastructure (Responses mentioning transport or infrastructure generally, including specific types like "Lack of buses", "Railways", "pot holes on the road", "energy generation", "energy provisions", or particular infrastructure projects like "wasting money on HS2")
- "23": Pensions/ageing (Two main themes: mentions of pensions specifically, and references to effects of ageing population like "Cost of aging population", "OLD PEOPLE", specific pensions issues like "WASPI Women")
- "24": Pol-neg (Responses negative about current states of politics or politicians in general, like "Lying politicians", "corruption", "untrustworthy MPs", "political chaos", "Dysfunctional political system", media bias complaints)
- "25": Partisan-neg (Responses negative about specific political party or politician, like "getting the tories OUT", "Keeping Jeremy Corbyn out", "useless Labour party". Only when party/politician explicitly named)
- "26": Societal divides (Responses referring to division or lack of cohesion in British society without specific economic/cultural reference, including regional division like "Widening gap between London and rest of country")
- "27": Morals (Responses referring to morals directly or related issues like "Declining moral standards", "ethics", religious/spiritual guidance, non-criminal moral behaviour seen as corrupting like "greed", "selfishness")
- "28": National identity, goals-loss (Concern about cultural change, sometimes extreme/prejudiced, without direct immigration reference. Includes "Islam", "Britain losing its identity", opposition to Black Lives Matter, conspiracy theories)
- "29": Racism/discrimination (Responses identifying racism or discrimination directly as most important issue, including against other marginalised groups like disabled/immigrants. Examples: "Intolerance", "Xenophobia", "Rising racism")
- "30": Asylum (Responses specifically referencing asylum or refugees, whether using solely these particular words or in longer responses, like "too many asylum seekers/immigrants coming into our country". Includes those in support of refugees, such as "Fair deal for asylum seekers", "uk should be taking in more refugees")
- "31": Crime (Responses mentioning "crime", "Anti-social behaviour", specific crimes like "Knife crime", references to "law", "law and order", "policing", "drugs" generally or specific drugs)
- "32": Foreign affairs (Residual code for foreign affairs responses not specifically about war/defence or Brexit/EU, like "global instability", "Our relationship with other nations", "Trump", foreign aid)
- "33": War (Specific references to war or military intervention, like "prospect of war", "world war 3", "Nuclear war", "War in Ukraine", "Russia's invasion", plus peace references like "world peace")
- "34": Defence (Responses focusing on "Defence" or "Security", military funding like "Level of Armed Forces", specific military spending like "Trident renewal", but not mentioning war/conflict)
- "35": Pol values-authoritarian (Responses aligned with authoritarian position: issue-based like "Political correctness gone mad", tribal like "Left wing anarchists", outlook-based like "respect for authority", "social order")
- "36": Pol values-liberal (Liberal-aligned responses: policy issues like "animal welfare", "state intrusion into privacy", tribal opposition like "Fascism", "alt right", outlook like "freedom", "Civil Liberties", "Human rights")
- "37": Gender/sexuality/family (References to gender, sexuality or family issues regardless of political perspective, including abortion, LGBT+ issues, feminist/anti-feminist perspectives, family life quality)
- "38": Pol values-right (Right-wing economic responses: policy like "unions getting more power", "Nanny state", tribal opposition like "Far Left", "Marxism", outlook like "People's lack of responsibility", "Too much government")
- "39": Pol values-left (Left-wing economic responses: policy like "privatisation of public services", "Workers rights", tribal opposition like "Capitalism", "Neo Liberalism", outlook like "Redistribution of wealth", "Social Justice")
- "40": Election outcome (Used when respondent refers to the result of a recent or upcoming election in a relatively neutral way, or positive about a particular outcome. Includes references to internal party elections. Examples: "election", "The up coming General Election", "hung parliament", "23-Jun-2016", "Ensuring the Conservative party are successful", "the next prime minister", "The possible election of Jeremy Corbyn", "Choosing the right Government!")
- "41": Constitutional (Constitutional topics excluding devolution/Scottish independence: "constitution", electoral system changes, monarchy abolition, Welsh independence, West Lothian question, Northern Ireland governance)
- "42": International trade (Responses focusing on trade or related terms like "Balance of Trade", "TRADE DEFICIT", "TTIP", post-Brexit trade deals, UK self-sufficiency concerns)
- "43": Devolution (Responses specifically referencing devolution, either generally or for specific UK nations, like "Devolution of power across the UK", "Devo Max")
- "44": Scot-independence (Specific reference to Scottish independence or related subjects, like "Scottish independence vote", "Preservation of the Union", "The fight for independence")
- "45": Foreign emergency (References to foreign emergency not specifically about war, like "Hong Kong", "Natural disasters", "Genocide")
- "46": Domestic emergency (References to particular domestic emergency affecting Britain, like "Flooding", "storms", "Grenfell Tower", future potential emergencies like power blackouts)
- "47": Referendum unspecified (Respondent refers to referendum but unclear which one - Scottish Independence, EU, or another)
- "48": Other (Any response clearly mentioning political issue/policy area but cannot be placed in previous categories)

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the category for the first issue mentioned.

OPEN-ENDED TEXT RESPONSE: a bad economy
Answer: 4 economy-general

OPEN-ENDED TEXT RESPONSE: immergration
Answer: 2 immigration

OPEN-ENDED TEXT RESPONSE: climate change and unemployment
Answer: 18 environment

Now output the category of the following OPEN-ENDED TEXT RESPONSE. Output only a number.

OPEN-ENDED TEXT RESPONSE: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """I will provide you with an OPEN-ENDED TEXT RESPONSES from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to the OPEN-ENDED TEXT RESPONSE, returning only the most relevant category.

Categories:
- "1": Europe
- "2": Immigration
- "3": Economy-general
- "4": Economy-personal
- "5": Unemployment
- "6": Taxation
- "7": Debt/deficit
- "8": Inflation
- "9": Living costs
- "10": Health
- "11": Coronavirus
- "12": COVID-economy
- "13": Terrorism
- "14": Poverty
- "15": Inequality
- "16": Housing
- "17": Environment
- "18": Education
- "19": Welfare
- "20": Austerity
- "21": Social care
- "22": Transport/infrastructure
- "23": Pensions/ageing
- "24": Pol-neg
- "25": Partisan-neg
- "26": Societal divides
- "27": Morals
- "28": National identity, goals-loss
- "29": Racism/discrimination
- "30": Asylum
- "31": Crime
- "32": Foreign affairs
- "33": War
- "34": Defence
- "35": Pol values-authoritarian
- "36": Pol values-liberal
- "37": Gender/sexuality/family
- "38": Pol values-right
- "39": Pol values-left
- "40": Election outcome
- "41": Constitutional
- "42": International trade
- "43": Devolution
- "44": Scot-independence
- "45": Foreign emergency
- "46": Domestic emergency
- "47": Referendum unspecified
- "48": Other

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the category for the first issue mentioned.

Now output the category of the following OPEN-ENDED TEXT RESPONSE. Output only a number.

OPEN-ENDED TEXT RESPONSE: a bad economy"""
                    },
                    {
                        "role": "assistant",
                        "content": "4"
                    },
                    {
                        "role": "user",
                        "content": """OPEN-ENDED TEXT RESPONSE: immergration"""
                    },
                    {
                        "role": "assistant",
                        "content": "2"
                    },
                    {
                        "role": "user",
                        "content": """OPEN-ENDED TEXT RESPONSE: climate change and unemployment"""
                    },
                    {
                        "role": "assistant",
                        "content": "18"
                    },
                    {
                        "role": "user",
                        "content": """OPEN-ENDED TEXT RESPONSE: {text}"""
                    },
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short] [orig*]',
                'original_description': '[zero-shot-short] [original]',
                'compatible_output_mapping': ['default output mapping no numbers'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """You will read an unedited answer from the British Election Study to the question “What is the most important issue facing the country?”  
Your task: pick the single category that best fits the response and output only that category name. The category must be one of the dictionary keys shown below—do not output anything else.

If several issues are mentioned, choose the label that matches the first issue named.

Note: Russia’s invasion of Ukraine occurred before fieldwork; any mention of Ukraine or Russia refers to War.

Allowed output labels (dictionary keys) and brief guidance
Health – NHS or healthcare  
Education – schools, universities, etc.  
Pol-neg – criticism of politics/system/media/corruption without naming a party or politician  
Partisan-neg – attacks on a specific party or politician  
Societal divides – social fragmentation, polarisation  
Morals – abortion, religiosity, society’s morals  
National identity, goals-loss – ethnonationalism, loss of national purpose  
Racism/discrimination – racism, sexism, discrimination  
Welfare – benefits, social security  
Terrorism – terrorism or extremism  
Immigration – immigration, overpopulation (exclude refugees)  
Asylum – refugees, asylum seekers  
Crime – crime, policing  
Europe – EU matters, Brexit  
Election outcome – who won, hung parliament, etc.  
Constitutional – UK constitutional issues except Brexit, devolution, Scottish independence; include electoral reform  
International trade – trade deals, tariffs  
Devolution – devolved powers within the UK  
Scot-independence – Scottish independence  
Foreign affairs – foreign policy not involving war  
War – armed conflict, Ukraine, Russia, Syria, calls for peace  
Defence – defence spending, armed forces (non-war)  
Foreign emergency – short-term overseas crises not war  
Domestic emergency – short-term domestic crises (floods, storms, Grenfell) but not COVID  
Economy-general – overall economy  
Economy-personal – respondent’s personal finances  
Unemployment – jobs, unemployment  
Taxation – taxes  
Debt/deficit – government/national debt or deficit  
Inflation – general price rises  
Living costs – cost of living, energy bills  
Poverty – poverty, hardship  
Austerity – spending cuts, lack of public funding  
Inequality – inequality  
Housing – housing supply, prices, homelessness  
Social care – care for elderly, disabled, adults, children  
Transport/infrastructure – transport, infrastructure  
Pensions/ageing – pensions, ageing society  
Environment – climate change, environment  
Pol values-authoritarian – pro-authoritarian or anti-social-liberal views (e.g., “wokeness” complaints)  
Pol values-liberal – pro-social-liberal or anti-authoritarian views (e.g., women’s/LGBT/minority rights)  
Gender/sexuality/family – gender, sexuality, family policy  
Pol values-right – pro-economic-right or anti-left views  
Pol values-left – pro-economic-left or anti-right views  
Referendum unspecified – reference to a referendum where it is unclear if Brexit or independence  
Coronavirus – COVID-19 (health, not economic impacts)  
COVID-economy – economic effects of COVID-19  
Other – does not fit any other label  

Examples  
Input: a bad economy → Economy-general  
Input: immergration → Immigration  
Input: climate change and unemployment → Environment (first issue takes priority)

Classify this response: {text}"""
                    }
                ],
            },
        ]
