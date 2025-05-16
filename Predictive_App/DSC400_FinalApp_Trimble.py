import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import urllib.request
import spacy
import subprocess

# Install the model if missing
try:
    nlp_md = spacy.load("en_core_web_md")
except OSError:
    print("🔧 en_core_web_md not found, installing now...")
    print("Installation successful!")
    nlp_md = spacy.load("en_core_web_md")

### Load Population & GDP Datasets ###
@st.cache_data
def load_data():
    pop_url = "https://raw.githubusercontent.com/cmtrimble/cmtrimble.github.io/main/Predictive_App/population.csv"
    gdp_url = "https://raw.githubusercontent.com/cmtrimble/cmtrimble.github.io/main/Predictive_App/gdp-worldbank-constant-usd.csv"

    try:
        df_pop = pd.read_csv(pop_url, encoding='latin1')
        df_gdp = pd.read_csv(gdp_url, encoding='latin1')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None

    return df_pop, df_gdp

### Data Cleaning & Preprocessing ###
def preprocess_data(df_pop, df_gdp):
    df_pop.columns = df_pop.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Â', '')

    # Rename for consistency
    df_pop.rename(columns={'Population_(historical)': 'Population_historical'}, inplace=True)

    # Filter by year
    latest_year = df_pop['Year'].max()
    df_pop_filtered = df_pop[df_pop['Year'] >= 1950].copy()

    # Compute population growth trends
    df_pop_filtered['Pop_Growth'] = df_pop_filtered.groupby('Entity')['Population_historical'].pct_change().fillna(0)

    # Clean GDP data
    df_gdp_latest = df_gdp[df_gdp['Year'] == df_gdp['Year'].max()].dropna(subset=['GDP (constant 2015 USD)'])
    df_gdp_latest['GDP (constant 2015 USD)'] = pd.to_numeric(df_gdp_latest['GDP (constant 2015 USD)'], errors='coerce')

    # Merge population and GDP data
    df_combined = pd.merge(df_pop_filtered, df_gdp_latest, left_on=['Entity', 'Year'], right_on=['Entity', 'Year'], how='left')

    # **Check for Missing or Infinite Values**
    df_combined.dropna(subset=['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)'], inplace=True)
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)

    # Normalize Data
    scaler = MinMaxScaler()
    df_combined[['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)']] = scaler.fit_transform(df_combined[['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)']])

    return df_combined

### Regression Analysis ###
def run_regression(df_combined):
    X = df_combined[['Population_historical', 'Pop_Growth']]
    y = df_combined['GDP (constant 2015 USD)']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    return X, y, model

### 🚀 Neural Network Training ###
@st.cache_data
def train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Input(shape=(2,)),  # Fixes TensorFlow `input_dim` warning
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    return model

### 🚀 Streamlit Controls for Running Modules ###
st.title("Modularized Debugging")

# Ensure data is loaded before preprocessing
if st.sidebar.button("Load Data"):
    df_pop, df_gdp = load_data()

    if df_pop is None or df_gdp is None:
        st.error("🚨 Error: Population or GDP data failed to load.")
    else:
        st.session_state["df_pop"] = df_pop  # Store `df_pop` globally
        st.session_state["df_gdp"] = df_gdp  # Store `df_gdp` globally
        st.write("Data Loaded Successfully!")

# **Preprocess Data**
if st.sidebar.button("Preprocess Data"):
    if "df_pop" not in st.session_state or "df_gdp" not in st.session_state:
        st.error("🚨 Error: Load data first before preprocessing.")
    else:
        df_combined = preprocess_data(st.session_state["df_pop"], st.session_state["df_gdp"])

        # Store the preprocessed data globally in session state
        st.session_state["df_combined"] = df_combined

        st.write("Data Preprocessed Successfully!")

# **Run Regression**
if st.sidebar.button("Run Regression"):
    if "df_combined" not in st.session_state:
        st.error("🚨 Error: Preprocess data first before running regression.")
    else:
        X, y, model = run_regression(st.session_state["df_combined"])

# **Train Neural Network Model**
if st.sidebar.button("Train Model"):
    if "df_combined" not in st.session_state:
        st.error("🚨 Error: Preprocess data first before training.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["df_combined"][['Population_historical', 'Pop_Growth']],
            st.session_state["df_combined"]['GDP (constant 2015 USD)'],
            test_size=0.2,
            random_state=42
        )

        # Store split datasets in session state
        st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
        st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test

        trained_model = train_model(X_train, y_train, X_test, y_test)

        st.session_state["trained_model"] = trained_model
        st.write("Model Training Completed!")

# **Run Predictions**
if st.sidebar.button("Generate Predictions"):
    if "trained_model" not in st.session_state or "X_test" not in st.session_state:
        st.error("🚨 Error: Train the model first before generating predictions!")
    else:
        trained_model = st.session_state["trained_model"]
        predictions = trained_model.predict(st.session_state["X_test"]).flatten().tolist()

        # Store predictions globally for easy access
        st.session_state["predictions"] = predictions[:10]

        st.write("Predictions Generated!")

# **Visualizations**
if st.sidebar.button("Generate Visualizations", key="generate_visualizations"):
    if "trained_model" not in st.session_state or "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.error("🚨 Error: Train the model first before generating predictions!")
    else:
        trained_model = st.session_state["trained_model"]
        predictions = trained_model.predict(st.session_state["X_test"]).flatten().tolist()

        st.session_state["predictions"] = predictions[:10]

        # Retrieve y_test for visualization
        y_test = st.session_state["y_test"]

        # Generate scatter plot
        fig, ax1 = plt.subplots()
        ax1.scatter(y_test, predictions)
        ax1.set_xlabel("Actual GDP")
        ax1.set_ylabel("Predicted GDP")
        ax1.set_title("Actual vs. Predicted GDP")

        st.pyplot(fig)

st.sidebar.title("GDP Year Selection")

# Ensure GDP data is loaded before filtering
if "df_gdp" in st.session_state:
    selected_year = st.sidebar.selectbox("Select Year", st.session_state["df_gdp"]["Year"].unique())

    # Filter data for the selected year
    filtered_gdp_data = st.session_state["df_combined"][st.session_state["df_combined"]["Year"] == selected_year]

    # Store selection in session state
    st.session_state["selected_year"] = selected_year
else:
    st.sidebar.warning("⚠ Load data first to enable year selection.")


# Streamlit Integration
import streamlit as st
import plotly.express as px
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import spacy
import subprocess

# Ensure SpaCy model installation
try:
    nlp_md = spacy.load("en_core_web_md")
except OSError:
    print("🔧 Model not found, installing now...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
    nlp_md = spacy.load("en_core_web_md")
    print("Model installed successfully!")

### 🚀 Caching Functions ###

@st.cache_data
def load_population_data():
    pop_url = "https://raw.githubusercontent.com/cmtrimble/cmtrimble.github.io/main/Predictive_App/population.csv"
    df = pd.read_csv(pop_url, encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Â', '')

    # **Fix: Correct Filtering**
    df_filtered = df[df['Year'] >= 1950].copy()

    # **Fix Column Name Consistency**
    df_filtered.rename(columns={'Population_(historical)': 'Population_historical'}, inplace=True)

    # **Compute Population Growth Trends Correctly**
    df_filtered['Pop_Growth'] = df_filtered.groupby('Entity')['Population_historical'].pct_change().fillna(0)

    return df_filtered

@st.cache_data
def load_gdp_data():
    gdp_url = "https://raw.githubusercontent.com/cmtrimble/cmtrimble.github.io/main/Predictive_App/gdp-worldbank-constant-usd.csv"
    return pd.read_csv(gdp_url, encoding='latin1')

# Load cached datasets
df_pop = load_population_data()
df_gdp = load_gdp_data()

### **Data Preprocessing Fixes** ###
df_gdp_latest = df_gdp[df_gdp['Year'] == df_gdp['Year'].max()].dropna(subset=['GDP (constant 2015 USD)'])
df_gdp_latest['GDP (constant 2015 USD)'] = pd.to_numeric(df_gdp_latest['GDP (constant 2015 USD)'], errors='coerce')

df_combined = pd.merge(df_pop, df_gdp_latest, left_on=['Entity', 'Year'], right_on=['Entity', 'Year'], how='left')

df_combined.dropna(subset=['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)'], inplace=True)

scaler = MinMaxScaler()
df_combined[['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)']] = scaler.fit_transform(df_combined[['Population_historical', 'Pop_Growth', 'GDP (constant 2015 USD)']])

### **Streamlit Sidebar (Year Selection)** ###
st.sidebar.title("GDP Year Selection")

# Ensure GDP data is loaded before filtering
if "df_gdp" in st.session_state:
    selected_year = st.sidebar.selectbox("Select Year", df_gdp["Year"].unique())

    # Filter data for the selected year
    filtered_gdp_data = df_combined[df_combined["Year"] == selected_year]

    # Store selection in session state
    st.session_state["selected_year"] = selected_year
else:
    st.sidebar.warning("Load data first to enable year selection.")

### 🚀 **Interactive Map Visualization** ###
if "selected_year" in st.session_state:
    st.write(f"### Interactive Map for {st.session_state['selected_year']}")

    fig_map = px.choropleth(filtered_gdp_data, locations="Entity", locationmode='country names',
                            color="GDP (constant 2015 USD)",
                            hover_name="Entity",
                            hover_data={"GDP (constant 2015 USD)": True, "Population_historical": True, "Pop_Growth": True},
                            projection="natural earth",
                            title=f'World GDP in {st.session_state["selected_year"]}')
    st.plotly_chart(fig_map)
else:
    st.warning("Select a year to visualize GDP data.")

### **Fix: ML Model Training with Arguments** ###
@st.cache_data
def train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(128, input_dim=2, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    return model

# **Ensure `X_train, X_test` is created before training**
X = df_combined[['Population_historical', 'Pop_Growth']].values
y = df_combined['GDP (constant 2015 USD)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Train the model with correct parameters**
trained_model = train_model(X_train, y_train, X_test, y_test)

# **Generate predictions safely**
predictions = trained_model.predict(X_test).flatten().tolist()

print("Raw Predictions:", predictions[:10])  # Debugging output

# **Model Predictions**
if predictions and len(predictions) > 0:
    st.write("### Model Predictions (GDP)")
    st.write(predictions)
else:
    st.write("No predictions were generated. Check model training.")

# **Checkbox for Raw Data**
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

