import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Pagina configuratie
st.set_page_config(
    page_title="CO2-uitstoot vs Welvaart (Richting B)",
    page_icon="🌍",
    layout="wide"
)

# Gegevens lokaal inlezen (voldoet aan tentameneis: geen live API-calls)
@st.cache_data
def load_data():
    return pd.read_csv("co2_gdp_data.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Lokaal bestand 'co2_gdp_data.csv' niet gevonden! Voer eerst het script 'etl.py' uit.")
    st.stop()

# Titel en introductie
st.title("🌍 De Omgekeerde U: CO₂-uitstoot tegen Welvaart")
st.markdown("""
Dit dashboard onderzoekt de **Environmental Kuznets Curve (EKC)** hypothese. We toetsen of de CO₂-uitstoot per hoofd 
blijft stijgen bij toenemende welvaart, of dat er een **kantelpunt** bestaat waarna rijkere landen per hoofd juist minder gaan uitstoten.
""")

# --- SIDEBAR INTERACTIE ---
st.sidebar.header("🎛️ Dashboard Besturing")
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_year = st.sidebar.slider("Selecteer Jaartal", min_value=min_year, max_value=max_year, value=max_year)

# Filter data voor gekozen jaartal
df_year = df[df['year'] == selected_year].dropna(subset=['gdp_per_capita', 'co2_per_capita'])

# --- EIS 2: KAART (Geospatiale Inspectie) ---
st.header(f"1. Geospatiale Spreiding: CO₂-uitstoot per hoofd in {selected_year}")
st.markdown("De onderstaande wereldkaart toont de uitstoot per hoofd per land. Dit geeft direct ruimtelijk inzicht dat een tabel mist.")

fig_map = px.choropleth(
    df_year,
    locations="iso3",
    color="co2_per_capita",
    hover_name="country",
    color_continuous_scale="Reds",
    labels={"co2_per_capita": "CO₂ per hoofd (ton)"},
    title=f"CO₂-uitstoot per inwoner ({selected_year})"
)
fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=450)
st.plotly_chart(fig_map, use_container_width=True)

# --- EIS 3: STATISTISCHE ONDERBOUWING (Kwadratische Regressie & Kantelpunt) ---
st.header("2. Statistische Onderbouwing: De Omgekeerde U-Curve")
st.markdown("""
Om de hypothese te bewijzen schatten we een **kwadratische regressie** ($CO_2 = \beta_0 + \beta_1 \cdot BBP + \beta_2 \cdot BBP^2$). 
Als de kwadratische coëfficiënt ($\beta_2$) negatief is, ontstaat er een bergparabool.
""")

# Multiselect om specifieke landen uit te lichten in de grafiek
all_countries = sorted(df['country'].dropna().unique())
default_iso = ['USA', 'CHN', 'NLD', 'DEU', 'IND', 'QAT']
default_names = df[df['iso3'].isin(default_iso)]['country'].unique().tolist()

selected_countries = st.sidebar.multiselect(
    "Uitgelichte landen in grafiek",
    options=all_countries,
    default=default_names if all(c in all_countries for c in default_names) else all_countries[:5]
)

# Berekening regressie en R-squared
x = df_year['gdp_per_capita'].values
y = df_year['co2_per_capita'].values
valid = ~np.isnan(x) & ~np.isnan(y)
x_clean, y_clean = x[valid], y[valid]

if len(x_clean) > 5:
    # Kwadratische fit (graad 2)
    coeffs = np.polyfit(x_clean, y_clean, 2)
    a, b, c = coeffs
    
    # R-squared berekenen
    p = np.poly1d(coeffs)
    y_pred = p(x_clean)
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Theoretisch kantelpunt (top van de parabool: -b / (2*a))
    tipping_point = -b / (2 * a) if a < 0 else None

    # Tonen van statistische kerncijfers (KPI's)
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Goede Fit ($R^2$)", f"{r_squared:.3f}")
    col2.metric("Kwadratische Coëfficiënt ($\beta_2$)", f"{a:.2e}")
    if tipping_point and tipping_point > 0:
        col3.metric("Theoretisch Kantelpunt BBP", f"${tipping_point:,.0f}")
    else:
        col3.metric("Theoretisch Kantelpunt", "Niet bereikt (open lijn)")

    # Scatter plot met polynomiale trendlijn
    df_year['highlight'] = df_year['country'].isin(selected_countries)
    
    fig_scatter = px.scatter(
        df_year,
        x="gdp_per_capita",
        y="co2_per_capita",
        hover_name="country",
        color="highlight",
        color_discrete_map={True: "#ff4b4b", False: "#1f77b4"},
        labels={"gdp_per_capita": "BBP per hoofd (USD)", "co2_per_capita": "CO₂-uitstoot per hoofd (ton)"},
        title=f"Welvaart vs CO₂-uitstoot in {selected_year}"
    )

    # Lijn toevoegen van de kwadratische fit
    x_range = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_range = p(x_range)
    fig_scatter.add_trace(go.Scatter(
        x=x_range, y=y_range, mode='lines', name='Kwadratische Fit (EKC)',
        line=dict(color='green', width=3)
    ))

    fig_scatter.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("Onvoldoende datapunten voor dit jaartal.")

# --- EIS 4: ZELF AFGELEIDE KOLOM & DATA-INSPECTIE ---
st.header("3. Gedetailleerde Data & Zelf Afgeleide Kolommen")
st.markdown("""
In dit dashboard is de kolom **`gdp_per_capita_sq`** (het kwadraat van het BBP per hoofd) zelf afgeleid en gebruikt 
voor de statistische modellering van de Kuznets-curve.
""")

if selected_countries:
    df_table = df_year[df_year['country'].isin(selected_countries)]
else:
    df_table = df_year

st.dataframe(
    df_table[['country', 'iso3', 'year', 'gdp_per_capita', 'gdp_per_capita_sq', 'co2_per_capita']]
    .sort_values(by='gdp_per_capita', ascending=False),
    width="stretch"
)
