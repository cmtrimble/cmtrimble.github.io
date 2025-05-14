# Create a mapping dictionary to align country names if there are discrepancies
country_mapping = {
    'United States': 'United States',
    'South Korea': 'Korea, Rep.',
    'North Korea': 'Korea, Dem. People’s Rep.',
    'DR Congo': 'Democratic Republic of Congo',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'Syria': 'Syrian Arab Republic',
    'Cape Verde': 'Cabo Verde',
    'Timor-Leste': 'East Timor',
    'Micronesia': 'Micronesia (country)',
    'Saint Kitts & Nevis': 'Saint Kitts and Nevis',
    'Sint Maarten': 'Sint Maarten (Dutch part)',
    'Saint Vincent & Grenadines': 'Saint Vincent and the Grenadines',
    'Curacao': 'Curaçao',
    'Czechia': 'Czech Republic (Czechia)',
    'Sao Tome and Principe': 'Sao Tome & Principe',
    'Palestine': 'State of Palestine',
    'Turks and Caicos Islands': 'Turks and Caicos',
    'Taiwan': 'Taiwan*',
    'United States of America': 'United States',
    'Greenland': 'Greenland*',
    'Korea, Dem. People’s Rep.': 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Reunion': 'Réunion',
    'Saint Vincent and the Grenadines': 'St. Vincent & Grenadines',
    'Cape Verde': 'Cabo Verde',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena',
    # More mappings as needed
}

# Manually add data for countries with missing entries after the merge
missing_countries = [
    'United States of America', 'Greenland', 'North Korea', 'South Korea', 'Taiwan', 'Sao Tome & Principe',
    'Guam', 'Gibraltar', 'Martinique', 'French Polynesia', 'Réunion', 'Western Sahara', 'Eritrea', 'Isle of Man',
    'Caribbean Netherlands', 'British Virgin Islands', 'Cook Islands', 'New Caledonia', 'Wallis & Futuna',
    'French Guiana', 'Liechtenstein', 'Mayotte', 'Saint Martin', 'Tokelau', 'Anguilla', 'Tonga', 'South Sudan',
    'Yemen', 'Falkland Islands', 'Syrian Arab Republic', 'Holy See', 'Saint Helena', 'Northern Mariana Islands',
    'San Marino', 'Niue', 'Saint Barthelemy', 'Guadeloupe', 'Lebanon', 'State of Palestine', 'U.S. Virgin Islands',
    'Saint Pierre & Miquelon', 'Venezuela', 'American Samoa', 'Montserrat', 'Cabo Verde', 'Bhutan'
]
for country in missing_countries:
    if country not in df['Country'].values:
        if country in df_pop['Country'].values:
            population = df_pop.loc[df_pop['Country'] == country, 'Population_2024'].values[0]
        else:
            population = None
        entity = country_mapping.get(country, country)
        if entity in df_gdp_2023['Entity'].values:
            gdp = df_gdp_2023.loc[df_gdp_2023['Entity'] == entity, 'GDP (constant 2015 USD)'].values[0]
        else:
            gdp = None
        row = pd.DataFrame({
            'Country': [country],
            'Population_2024': [population],
            'Yearly_Change': [None],
            'Net_Change': [None],
            'Entity': [entity],
            'Code': [None],
            'Year': [2023],
            'GDP (constant 2015 USD)': [gdp]
        })
        # Filter out empty or all-NA entries before concatenation
        row = row.dropna(how='all')
        if not row.empty:  # Add this check to avoid FutureWarning
            df = pd.concat([df, row], ignore_index=True)
