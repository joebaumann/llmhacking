from data.data_utils import MyDataLoader
import copy


class ManifestosIssuesDetailedDataset(MyDataLoader):
    def __init__(self, data_directory='manifestos', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_dataset_specific_groups(self, df):
        party_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'partyname', 50)
        parfam_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'parfam', 12)
        country_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'countryname', 49)
        year_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'year', 100)

        temporal_split_groups = self.add_temporal_split_groupings(
            timestamp_col='date')

        return party_groupings + parfam_groupings + country_groupings + year_groupings + temporal_split_groups

    def get_cols_to_stratify(self):
        return [
            "ground_truth",
            # columns used for grouping
            # ("partyname", 50),
            ("parfam", 12),
            # ("countryname", 49),
            # ("year", 100),
        ]

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'Technology and Infrastructure': 'Technology and Infrastructure',
            'Welfare State Expansion': 'Welfare State Expansion',
            'Controlled Economy': 'Controlled Economy',
            'Education Expansion': 'Education Expansion',
            'Equality': 'Equality',
            'Federalism': 'Federalism',
            'Economic Orthodoxy': 'Economic Orthodoxy',
            'Economic Growth': 'Economic Growth',
            'Labour Groups': 'Labour Groups',
            'Military/Defence': 'Military/Defence',
            'Underprivileged Minority Groups': 'Underprivileged Minority Groups',
            'Middle Class and Professional Groups': 'Middle Class and Professional Groups',
            'Culture': 'Culture',
            'Economic Goals': 'Economic Goals',
            'Protectionism': 'Protectionism',
            'Foreign Special Relationships': 'Foreign Special Relationships',
            'Keynesian Demand Management': 'Keynesian Demand Management',
            'Incentives': 'Incentives',
            'Traditional Morality': 'Traditional Morality',
            'Corporatism/Mixed Economy': 'Corporatism/Mixed Economy',
            'Internationalism': 'Internationalism',
            'Environmental Protection': 'Environmental Protection',
            'Constitutionalism': 'Constitutionalism',
            'Governmental and Administrative Efficiency': 'Governmental and Administrative Efficiency',
            'Political Corruption': 'Political Corruption',
            'Peace': 'Peace',
            'Free Market Economy': 'Free Market Economy',
            'Market Regulation': 'Market Regulation',
            'European Community/Union': 'European Community/Union',
            'Economic Planning': 'Economic Planning',
            'Non-economic Demographic Groups': 'Non-economic Demographic Groups',
            'Welfare State Limitation': 'Welfare State Limitation',
            'Education Limitation': 'Education Limitation',
            'Nationalisation': 'Nationalisation',
            'Anti-Growth Economy': 'Anti-Growth Economy',
            'Freedom and Human Rights': 'Freedom and Human Rights',
            'Democracy': 'Democracy',
            'Marxist Analysis': 'Marxist Analysis',
            'Multiculturalism': 'Multiculturalism',
            'Political Authority': 'Political Authority',
            'Civic Mindedness': 'Civic Mindedness',
            'Agriculture and Farmers': 'Agriculture and Farmers',
            'Anti-Imperialism': 'Anti-Imperialism',
            'Law and Order': 'Law and Order',
            'Centralisation': 'Centralisation',
            'National Way of Life': 'National Way of Life',
        }
        dict_with_mapping_options_with_numbers = {
            '45': 'National Way of Life',
            '44': 'Centralisation',
            '43': 'Law and Order',
            '42': 'Anti-Imperialism',
            '41': 'Agriculture and Farmers',
            '40': 'Civic Mindedness',
            '39': 'Political Authority',
            '38': 'Multiculturalism',
            '37': 'Marxist Analysis',
            '36': 'Democracy',
            '35': 'Freedom and Human Rights',
            '34': 'Anti-Growth Economy',
            '33': 'Nationalisation',
            '32': 'Education Limitation',
            '31': 'Welfare State Limitation',
            '30': 'Non-economic Demographic Groups',
            '29': 'Economic Planning',
            '28': 'European Community/Union',
            '27': 'Market Regulation',
            '26': 'Free Market Economy',
            '25': 'Peace',
            '24': 'Political Corruption',
            '23': 'Governmental and Administrative Efficiency',
            '22': 'Constitutionalism',
            '21': 'Environmental Protection',
            '20': 'Internationalism',
            '19': 'Corporatism/Mixed Economy',
            '18': 'Traditional Morality',
            '17': 'Incentives',
            '16': 'Keynesian Demand Management',
            '15': 'Foreign Special Relationships',
            '14': 'Protectionism',
            '13': 'Economic Goals',
            '12': 'Culture',
            '11': 'Middle Class and Professional Groups',
            '10': 'Underprivileged Minority Groups',
            '9': 'Military/Defence',
            '8': 'Labour Groups',
            '7': 'Economic Growth',
            '6': 'Economic Orthodoxy',
            '5': 'Federalism',
            '4': 'Equality',
            '3': 'Education Expansion',
            '2': 'Controlled Economy',
            '1': 'Welfare State Expansion',
            '0': 'Technology and Infrastructure',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping with numbers': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_with_numbers}
            },
        }

    def get_reported_results(self):
        return []


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following sentence from a political manifesto is primarily about {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Political manifesto sentence: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Decide whether this sentence from a political manifesto is about Foreign Special Relationships, Anti-Imperialism, Military/Defence, Peace, Internationalism, European Community/Union, Freedom and Human Rights, Democracy, Constitutionalism, Federalism, Centralisation, Governmental and Administrative Efficiency, Political Corruption, Political Authority, Free Market Economy, Incentives, Market Regulation, Economic Planning, Corporatism/Mixed Economy, Protectionism, Economic Goals, Keynesian Demand Management, Economic Growth, Technology and Infrastructure, Controlled Economy, Nationalisation, Economic Orthodoxy, Marxist Analysis, Anti-Growth Economy, Environmental Protection, Culture, Equality, Welfare State Expansion, Welfare State Limitation, Education Expansion, Education Limitation, National Way of Life, Traditional Morality, Law and Order, Civic Mindedness, Multiculturalism, Labour Groups, Agriculture and Farmers, Middle Class and Professional Groups, Underprivileged Minority Groups or Non-economic Demographic Groups.

Respond only with the one category that is most relevant.

---

Sentence: {text}

Classification: """
                    }
                ],
            },
            {
                'description': '[zero-shot-short] with numbers',
                'compatible_output_mapping': [
                    'default output mapping with numbers'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Sentence: {text}

Decide which of the following categories the above sentence from a political manifesto is about:
0: Technology and Infrastructure
1: Welfare State Expansion
2: Controlled Economy
3: Education Expansion
4: Equality
5: Federalism
6: Economic Orthodoxy
7: Economic Growth
8: Labour Groups
9: Military/Defence
10: Underprivileged Minority Groups
11: Middle Class and Professional Groups
12: Culture
13: Economic Goals
14: Protectionism
15: Foreign Special Relationships
16: Keynesian Demand Management
17: Incentives
18: Traditional Morality
19: Corporatism/Mixed Economy
20: Internationalism
21: Environmental Protection
22: Constitutionalism
23: Governmental and Administrative Efficiency
24: Political Corruption
25: Peace
26: Free Market Economy
27: Market Regulation
28: European Community/Union
29: Economic Planning
30: Non-economic Demographic Groups
31: Welfare State Limitation
32: Education Limitation
33: Nationalisation
34: Anti-Growth Economy
35: Freedom and Human Rights
36: Democracy
37: Marxist Analysis
38: Multiculturalism
39: Political Authority
40: Civic Mindedness
41: Agriculture and Farmers
42: Anti-Imperialism
43: Law and Order
44: Centralisation
45: National Way of Life

Respond with only one number, corresponding to the one category that is most relevant.

Classification: """
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping with numbers'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Sentence: {text}

Classify the above sentence from a political manifesto into ONE of the following 46 categories. Each category represents a major policy domain and includes both positive and negative stances on the issue:

**EXTERNAL RELATIONS:**
0: Technology and Infrastructure - Modernisation of industry, transport, communication; science and technological developments; infrastructure spending
9: Military/Defence - Military expenditure, armed forces, defence, security, disarmament, military treaties
15: Foreign Special Relationships - Relations with particular countries with special relationships; cooperation with specific nations
20: Internationalism - International cooperation, aid to developing countries, world planning, UN support vs. national independence, sovereignty
25: Peace - Peace as goal, peaceful crisis resolution, ending wars vs. military solutions to conflicts
28: European Community/Union - Support for or opposition to European integration, EU membership, regional integration
42: Anti-Imperialism - Opposition to imperial behavior, foreign financial influence, World Bank/IMF criticism

**FREEDOM AND DEMOCRACY:**
22: Constitutionalism - Support for or opposition to constitutional aspects, constitutional amendments
35: Freedom and Human Rights - Personal freedom, human rights, civil rights, refugee policies
36: Democracy - Democratic processes, representative vs. direct democracy, democratic participation

**POLITICAL SYSTEM:**
5: Federalism - Decentralisation vs. centralisation of political power, subsidiarity, local autonomy
23: Governmental and Administrative Efficiency - Government efficiency, administrative reform, civil service restructuring
24: Political Corruption - Eliminating corruption, clientelist structures, abuse of power
39: Political Authority - Party/leader competence, strong government, political authority
44: Centralisation - Support for unitary government, centralized political procedures

**ECONOMY:**
2: Controlled Economy - Direct government economic control, price controls, minimum wages
6: Economic Orthodoxy - Budget deficit reduction, economic retrenchment, strong currency, traditional economic institutions
7: Economic Growth - Economic growth paradigm, measures to aid economic expansion
13: Economic Goals - General economic statements without specific policy positions
14: Protectionism - Trade protection, tariffs, quotas vs. free trade
16: Keynesian Demand Management - Demand-side economic policies, consumer assistance, stimulus plans
17: Incentives - Supply-side policies, business assistance, tax breaks, subsidies
19: Corporatism/Mixed Economy - Cooperation between government, employers, and unions
26: Free Market Economy - Free market capitalism, laissez-faire, private enterprise, individual initiative
27: Market Regulation - Consumer protection, economic competition, preventing monopolies, social market economy
29: Economic Planning - Long-term government economic planning, consultative/indicative planning
33: Nationalisation - Government ownership of industries, keeping/expanding state enterprises
34: Anti-Growth Economy - Anti-growth politics, sustainability, opposition to growth-focused policies
37: Marxist Analysis - Marxist-Leninist ideology and terminology

**WELFARE AND QUALITY OF LIFE:**
1: Welfare State Expansion - Social services, healthcare, pensions, social housing expansion
3: Education Expansion - Educational provision improvement and expansion at all levels
4: Equality - Social justice, fair treatment, special protection for underprivileged, anti-discrimination
12: Culture - State funding for cultural facilities, arts, sports, museums, libraries
21: Environmental Protection - Environmental protection, climate change, green policies, animal rights
31: Welfare State Limitation - Limiting social expenditures, private care before state care
32: Education Limitation - Limiting educational expenditure, study fees, private schools

**FABRIC OF SOCIETY:**
18: Traditional Morality - Traditional/religious moral values, family values, religious institutions vs. modern values
38: Multiculturalism - Cultural diversity, immigrant integration approaches, indigenous rights
40: Civic Mindedness - National solidarity, civil society, volunteering, public spirit
43: Law and Order - Law enforcement, crime control, police resources, penalties
45: National Way of Life - National pride, patriotism, national ideas, immigration policies

**SOCIAL GROUPS:**
8: Labour Groups - Working class, trade unions, employment conditions, wages
10: Underprivileged Minority Groups - Handicapped, minorities, immigrants not defined economically/demographically
11: Middle Class and Professional Groups - Middle class, professionals, white collar workers, service sector
30: Non-economic Demographic Groups - Women, students, age groups, demographic special interests
41: Agriculture and Farmers - Agricultural policies, farmer support, rural interests

**INSTRUCTIONS:**
- Read the sentence carefully and identify its main policy focus
- Consider both explicit policy statements and implicit political messages
- If multiple categories seem relevant, choose the most specific and central theme
- Categories aggregate both positive and negative stances (e.g., Military includes both pro-military and anti-military statements)
- Respond with only the number (0-45) of the most relevant category

Classification: """
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Choose the single category that best fits the political manifesto sentence below. Select only one of these labels: Technology and Infrastructure, Welfare State Expansion, Controlled Economy, Education Expansion, Equality, Federalism, Economic Orthodoxy, Economic Growth, Labour Groups, Military/Defence, Underprivileged Minority Groups, Middle Class and Professional Groups, Culture, Economic Goals, Protectionism, Foreign Special Relationships, Keynesian Demand Management, Incentives, Traditional Morality, Corporatism/Mixed Economy, Internationalism, Environmental Protection, Constitutionalism, Governmental and Administrative Efficiency, Political Corruption, Peace, Free Market Economy, Market Regulation, European Community/Union, Economic Planning, Non-economic Demographic Groups, Welfare State Limitation, Education Limitation, Nationalisation, Anti-Growth Economy, Freedom and Human Rights, Democracy, Marxist Analysis, Multiculturalism, Political Authority, Civic Mindedness, Agriculture and Farmers, Anti-Imperialism, Law and Order, Centralisation, National Way of Life.\n\nProvide only the chosen label with no additional text.\n\n---\n\nSentence: {text}\n\nLabel:"
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short] with numbers',
                'original_description': '[zero-shot-short] with numbers',
                'compatible_output_mapping': ['default output mapping with numbers'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Sentence to classify: {text}

Review the policy areas listed below and pick the single category that best matches the content of the sentence. Return only the associated number—no words or extra characters.

0: Technology and Infrastructure  
1: Welfare State Expansion  
2: Controlled Economy  
3: Education Expansion  
4: Equality  
5: Federalism  
6: Economic Orthodoxy  
7: Economic Growth  
8: Labour Groups  
9: Military/Defence  
10: Underprivileged Minority Groups  
11: Middle Class and Professional Groups  
12: Culture  
13: Economic Goals  
14: Protectionism  
15: Foreign Special Relationships  
16: Keynesian Demand Management  
17: Incentives  
18: Traditional Morality  
19: Corporatism/Mixed Economy  
20: Internationalism  
21: Environmental Protection  
22: Constitutionalism  
23: Governmental and Administrative Efficiency  
24: Political Corruption  
25: Peace  
26: Free Market Economy  
27: Market Regulation  
28: European Community/Union  
29: Economic Planning  
30: Non-economic Demographic Groups  
31: Welfare State Limitation  
32: Education Limitation  
33: Nationalisation  
34: Anti-Growth Economy  
35: Freedom and Human Rights  
36: Democracy  
37: Marxist Analysis  
38: Multiculturalism  
39: Political Authority  
40: Civic Mindedness  
41: Agriculture and Farmers  
42: Anti-Imperialism  
43: Law and Order  
44: Centralisation  
45: National Way of Life"""
                    }
                ],
            },
        ]
