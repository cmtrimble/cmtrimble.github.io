# GDP source: World Bank and OECD (2025) – with minor processing by Our World in Data.
# “Gross domestic product (GDP) – World Bank – In constant US$” [dataset]. World Bank and OECD,
# “World Development Indicators” [original data].

# Population Source: Samithsachidanandan. (2025, February 6). Countries in the world by population (2025). Kaggle.
# https://www.kaggle.com/code/samithsachidanandan/countries-in-the-world-by-population-2025

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import matplotlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import findspark
import geopandas as gpd

# Initialize findspark
findspark.init()

# Set environment variables
os.environ['PYSPARK_PYTHON'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Scripts/python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Scripts/python.exe'
os.environ['HADOOP_HOME'] = 'D:/MediaDocs_Caleb/Documents/School/DSC400/hadoop-3.0.0'
os.environ['SPARK_HOME'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Lib/site-packages/pyspark'

# Increase memory allocation for Spark
spark = SparkSession.builder \
    .appName("PopulationGDPAnalysis") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load the population and GDP datasets
df_pop = pd.read_csv('C:\\Users\\caleb\\PycharmProjects\\DSC400\\Project\\World_Population_Data.csv', encoding='latin1')
df_gdp = pd.read_csv('C:\\Users\\caleb\\PycharmProjects\\DSC400\\Project\\gdp-worldbank-constant-usd.csv',
                     encoding='latin1')

# Clean the column names and apply country mapping
df_pop.columns = df_pop.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Â', '')
country_mapping = {
    'United States': 'United States', 'South Korea': 'Korea, Rep.', 'North Korea': 'Korea, Dem. People’s Rep.',
    'DR Congo': 'Democratic Republic of Congo', "Côte d'Ivoire": "Cote d'Ivoire", 'Syria': 'Syrian Arab Republic',
    'Cape Verde': 'Cabo Verde', 'Timor-Leste': 'East Timor', 'Micronesia': 'Micronesia (country)',
    'Saint Kitts & Nevis': 'Saint Kitts and Nevis', 'Sint Maarten': 'Sint Maarten (Dutch part)',
    'Saint Vincent & Grenadines': 'Saint Vincent and the Grenadines', 'Curacao': 'Curaçao', 'Czechia': 'Czech Republic (Czechia)',
    'Sao Tome and Principe': 'Sao Tome & Principe', 'Palestine': 'State of Palestine', 'Turks and Caicos Islands': 'Turks and Caicos',
    'Taiwan': 'Taiwan*', 'Greenland': 'Greenland*', 'Korea, Dem. People’s Rep.': 'North Korea', 'Korea, Rep.': 'South Korea',
    'Reunion': 'Réunion', 'Saint Vincent and the Grenadines': 'St. Vincent & Grenadines', 'Cape Verde': 'Cabo Verde',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena', 'Venezuela': 'Venezuela, RB',
    'Democratic Republic of Congo': 'Congo, Dem. Rep.', 'Western Sahara': 'Western Sahara'
}
df_pop['Country'] = df_pop['Country'].replace(country_mapping)
df_pop['Population_2024'] = pd.to_numeric(df_pop['Population_2024'].astype(str).str.replace(',', ''), errors='coerce')

# Clean and standardize the GDP dataset for 2023
df_gdp_2023 = df_gdp[df_gdp['Year'] == 2023].dropna(subset=['GDP (constant 2015 USD)'])
df_gdp_2023['GDP (constant 2015 USD)'] = pd.to_numeric(df_gdp_2023['GDP (constant 2015 USD)'], errors='coerce')

# Merge the population and GDP datasets
df = pd.merge(df_pop, df_gdp_2023, left_on='Country', right_on='Entity', how='left')

# Drop unnecessary columns
columns_to_drop = ['Rank', 'Density_P/Km²', 'Land_Area_Km²', 'Migrants_net', 'Fert._Rate', 'Med._Age', 'Urban_Pop_%', 'World_Share']
df_pop = df_pop.drop(columns=columns_to_drop, errors='ignore')

df_pop = df_pop.replace([np.inf, -np.inf], np.nan)
df_pop = df_pop.dropna()

scaler = StandardScaler()
df_pop['Population_2024'] = scaler.fit_transform(df_pop[['Population_2024']])

# Merge datasets on country name using a left merge
df = pd.merge(df_pop, df_gdp_2023, left_on='Country', right_on='Entity', how='left')

# Check for missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Population_2024', 'GDP (constant 2015 USD)'], inplace=True)
df = df[df['Population_2024'] != 0]
df = df[df['GDP (constant 2015 USD)'] != 0]

# Normalize the data
min_max_scaler = MinMaxScaler()
df[['Population_2024', 'GDP (constant 2015 USD)']] = min_max_scaler.fit_transform(df[['Population_2024', 'GDP (constant 2015 USD)']])

# Prepare data for regression analysis
X = df['Population_2024']
y = df['GDP (constant 2015 USD)']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

# Prepare data for deep learning model
X = df[['Population_2024']].values  # Features (Population)
y = df['GDP (constant 2015 USD)'].values  # Target (GDP)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Model Building ###

# Initialize the neural network model
model = Sequential()

# Add layers (including dropout for regularization)
model.add(Dense(128, input_dim=1, activation='relu'))  # First hidden layer (increased complexity)
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(32, activation='relu'))  # Third hidden layer
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(1))  # Output layer

# Compile the model with an adjusted learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual GDP (constant 2015 USD)")
plt.ylabel("Predicted GDP (constant 2015 USD)")
plt.title("Actual vs Predicted GDP")
plt.show()

### PySpark Integration ###

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Assemble features (Population_2024 as a feature)
assembler = VectorAssembler(inputCols=['Population_2024'], outputCol='features')
spark_df = assembler.transform(spark_df)

# Linear regression model with PySpark
lr = LinearRegression(featuresCol='features', labelCol='GDP (constant 2015 USD)')
lr_model = lr.fit(spark_df)

# Print model coefficients
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Stop the Spark session
spark.stop()

### Streamlit Integration ###
import streamlit as st
import plotly.express as px
import pandas as pd
import findspark
import os
from pyspark.sql import SparkSession

# Initialize findspark
findspark.init()

# Set environment variables
os.environ['PYSPARK_PYTHON'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Scripts/python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Scripts/python.exe'
os.environ['HADOOP_HOME'] = 'D:/MediaDocs_Caleb/Documents/School/DSC400/hadoop-3.0.0'
os.environ['SPARK_HOME'] = 'C:/Users/caleb/PycharmProjects/dsc360/new_env/Lib/site-packages/pyspark'

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CountryStatisticsDashboard") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load the datasets
try:
    df_pop = pd.read_csv('C:\\Users\\caleb\\PycharmProjects\\DSC400\\Project\\World_Population_Data.csv', encoding='latin1')
    df_gdp = pd.read_csv('C:\\Users\\caleb\\PycharmProjects\\DSC400\\Project\\gdp-worldbank-constant-usd.csv',
                         encoding='latin1')
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    st.stop()

# Clean the column names and apply country mapping
df_pop.columns = df_pop.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace(
    'Â', '')
country_mapping = {
    'United States': 'United States', 'South Korea': 'Korea, Rep.', 'North Korea': 'Korea, Dem. People’s Rep.',
    'DR Congo': 'Democratic Republic of Congo', "Côte d'Ivoire": "Cote d'Ivoire", 'Syria': 'Syrian Arab Republic',
    'Cape Verde': 'Cabo Verde', 'Timor-Leste': 'East Timor', 'Micronesia': 'Micronesia (country)',
    'Saint Kitts & Nevis': 'Saint Kitts and Nevis', 'Sint Maarten': 'Sint Maarten (Dutch part)',
    'Saint Vincent & Grenadines': 'Saint Vincent and the Grenadines', 'Curacao': 'Curaçao',
    'Czechia': 'Czech Republic (Czechia)',
    'Sao Tome and Principe': 'Sao Tome & Principe', 'Palestine': 'State of Palestine',
    'Turks and Caicos Islands': 'Turks and Caicos',
    'Taiwan': 'Taiwan*', 'Greenland': 'Greenland*', 'Korea, Dem. People’s Rep.': 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Reunion': 'Réunion', 'Saint Vincent and the Grenadines': 'St. Vincent & Grenadines', 'Cape Verde': 'Cabo Verde',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena', 'Venezuela': 'Venezuela, RB',
    'Democratic Republic of Congo': 'Congo, Dem. Rep.', 'Western Sahara': 'Western Sahara'
}
df_pop['Country'] = df_pop['Country'].replace(country_mapping)
df_pop['Population_2024'] = pd.to_numeric(df_pop['Population_2024'].astype(str).str.replace(',', ''), errors='coerce')

# Merge the population and GDP datasets
df_combined = pd.merge(df_pop, df_gdp, left_on='Country', right_on='Entity', how='left')

# Title of the app
st.title("Country Statistics Dashboard")

# Sidebar for user inputs
st.sidebar.title("GDP Year Selection")
selected_year = st.sidebar.selectbox("Select Year", df_gdp['Year'].unique())

# Filter data based on user input for the GDP plot
filtered_gdp_data = df_combined[df_combined['Year'] == selected_year]

# Interactive map
st.write("### Interactive Map")
fig_map = px.choropleth(filtered_gdp_data, locations="Country", locationmode='country names',
                        color="GDP (constant 2015 USD)",
                        hover_name="Country",
                        hover_data={"GDP (constant 2015 USD)": True, "Population_2024": True, "Rank": True},
                        projection="natural earth",
                        title=f'World GDP in {selected_year}')
st.plotly_chart(fig_map)

# GDP Plot
st.write(f"### GDP Plot ({selected_year})")
fig_gdp = px.scatter(filtered_gdp_data, x="Country", y="GDP (constant 2015 USD)",
                     color="GDP (constant 2015 USD)",
                     hover_name="Country",
                     hover_data={"GDP (constant 2015 USD)": True, "Population_2024": True, "Rank": True},
                     title=f'GDP of Countries in {selected_year}')
st.plotly_chart(fig_gdp)

# Population Plot (for 2024)
st.write("### Population Plot (2024)")
fig_population = px.scatter(df_pop, x="Country", y="Population_2024",
                            color="Population_2024",
                            hover_name="Country",
                            hover_data={"Population_2024": True},
                            title='Population of Countries in 2024')
st.plotly_chart(fig_population)

# Checkbox for raw data
if st.checkbox('See Raw Data'):
    st.write("### Raw Data")
    st.write(filtered_gdp_data)

# Checkbox for statistical and processing information
if st.checkbox('See Statistical and Processing Information'):
    st.markdown("""
    ### Statistical and Processing Information

    #### OLS Regression Results
    - Dep. Variable: **GDP (constant 2015 USD)**
    - Model: **OLS**
    - Method: **Least Squares**
    - No. Observations: **183**
    - Df Residuals: **181**
    - Df Model: **1**
    - Covariance Type: **nonrobust**
    - R-squared: **0.364**
    - Adj. R-squared: **0.360**
    - F-statistic: **103.5**
    - Prob (F-statistic): **1.62e-19**
    - Log-Likelihood: **209.50**
    - AIC: **-415.0**
    - BIC: **-408.6**

    ## const (Y-intercept)
    - Coefficient:0.0061
    - Standard Error: 0.006
    - t-value: 1.019
    - P>|t|: 0.310
    - 0.025|0.975 - -0.006|0.018

    ## Population_2024
    - Coefficient:0.5454
    - Standard Error: 0.054
    - t-value: 10.175
    - P>|t|: 0.000
    - 0.025|0.975 - -0.440|0.651

    - **Omnibus:** 293.393 <br>
    - **Durbin-Watson:** 1.769 <br>
    - **Prob (Omnibus):** 0.000 <br>
    - **Jarque-Bera (JB):** 61306.359 <br>
    - **Skew:** 7.170 <br>
    - **Prob(JB):** 0.000 <br>
    - **Kurtosis:** 91.513 <br>
    - **Cond. No.:** 9.37 <br>

    #### Deep Learning Model Training Results
    - Epoch 1/100: **Loss:** 0.0118, **Val Loss:** 5.8803e-04
    - Epoch 2/100: **Loss:** 0.0106, **Val Loss:** 5.2374e-04
    - Epoch 29/100: **Loss:** 0.0085, **Val Loss:** 4.4542e-04
    - **Final Model Loss:** 0.0004210729675833136
    - **Coefficients:** [0.5453914575490807]
    - **Intercept:** 0.0060564106630136595
    """)

# Caveats and Source Information
st.markdown("""
### Caveats
- The population data is for the year 2024 only.
- The GDP data covers multiple years, but only one year can be selected at a time.
- Not all countries have data due to varying reasons. These countries cannot be populated within the interactive map.

### Source Information 
- Population data: [World Population Data](https://www.kaggle.com/code/samithsachidanandan/countries-in-the-world-by-population-2025) 
- GDP data: [World Bank and OECD (2025) – with minor processing by Our World in Data](https://ourworldindata.org/grapher/gdp-worldbank-constant-usd?form=MG0AV3) 
""")


