import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CO2-uitstoot vs Welvaart Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Eenduidig kleurenpalet & thema voor alle visualisaties
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
THEME_TEMPLATE = "plotly_white"

@st.cache_data
def load_data():
    return pd.read_csv("co2_gdp_data.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Lokaal bestand 'co2_gdp_data.csv' niet gevonden! Voer eerst het script 'etl.py' uit.")
    st.stop()

st.title("🌍 CO₂-Uitstoot & Economische Welvaart")
st.markdown("""
Dit interactieve dashboard toetst de **Environmental Kuznets Curve (EKC)** hypothese. 
Gebruik de verfijnde besturing in de sidebar om de analyse te focussen en navigeer door de tabbladen voor de visualisaties.
""")

# --- STRAKKE EN INTUTIEVE SIDEBAR BEDIENING ---
st.sidebar.header("🎛️ Dashboard Besturing")
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_year = st.sidebar.slider("📅 Selecteer Jaartal", min_value=min_year, max_value=max_year, value=max_year)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Data Opschoning & Filters")
filter_outliers = st.sidebar.checkbox("Verwijder uitschieters (1.5 × IQR regel)", value=True)

# Selectie van specifieke regio/landen of focus in sidebar
analysis_focus = st.sidebar.selectbox(
    "🔍 Focus Analyse",
    ["Wereldwijd (Alle landen)", "Alleen G20 / Grote Economieën", "Focus op Europa"]
)

# Data filteren op basis van jaartal
df_year = df[df['year'] == selected_year].dropna(subset=['gdp_per_capita', 'co2_per_capita'])

# Focus toepassen
if "G20" in analysis_focus:
    g20_isos = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'IND', 'FRA', 'ITA', 'BRA', 'CAN', 'RUS', 'AUS', 'KOR', 'MEX', 'IDN', 'TUR', 'SAU', 'ARG', 'ZAF']
    df_year = df_year[df_year['iso3'].isin(g20_isos)]
elif "Europa" in analysis_focus:
    eu_isos = ['NLD', 'DEU', 'FRA', 'BEL', 'ITA', 'ESP', 'GBR', 'SWE', 'NOR', 'DNK', 'AUT', 'CHE', 'POL', 'FIN', 'PRT', 'IRL', 'GRC', 'CZE', 'ROU', 'HUN']
    df_year = df_year[df_year['iso3'].isin(eu_isos)]

# IQR uitbijter filter
if filter_outliers and len(df_year) > 5:
    q1_co2 = df_year['co2_per_capita'].quantile(0.25)
    q3_co2 = df_year['co2_per_capita'].quantile(0.75)
    iqr_co2 = q3_co2 - q1_co2
    
    q1_gdp = df_year['gdp_per_capita'].quantile(0.25)
    q3_gdp = df_year['gdp_per_capita'].quantile(0.75)
    iqr_gdp = q3_gdp - q1_gdp

    df_year = df_year[
        (df_year['co2_per_capita'] >= q1_co2 - 1.5 * iqr_co2) & 
        (df_year['co2_per_capita'] <= q3_co2 + 1.5 * iqr_co2) &
        (df_year['gdp_per_capita'] >= q1_gdp - 1.5 * iqr_gdp) & 
        (df_year['gdp_per_capita'] <= q3_gdp + 1.5 * iqr_gdp)
    ]

# --- TABBLADEN STRUCTUUR ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Wereldkaart & Globale KPI's", 
    "📈 Statistische Kuznets-Curve & Model", 
    "📦 Uitgebreide Distributie & Boxplotten", 
    "🔘 Categorie Analyse & Toplijsten"
])

# --- TAB 1: KAART & KPI'S ---
with tab1:
    st.header(f"Geospatiale Wereldspreiding in {selected_year}")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Actieve Landen", len(df_year))
    kpi2.metric("Gemiddeld BBP per hoofd", f"${df_year['gdp_per_capita'].mean():,.0f}" if len(df_year) > 0 else "$0")
    kpi3.metric("Gemiddelde CO₂ per hoofd", f"{df_year['co2_per_capita'].mean():.2f} ton" if len(df_year) > 0 else "0 ton")
    kpi4.metric("Totale Uitstoot", f"{df_year['co2_per_capita'].sum():,.0f} ton" if len(df_year) > 0 else "0 ton")

    st.markdown("---")
    
    col_map, col_rank = st.columns([2, 1])
    with col_map:
        fig_map = px.choropleth(
            df_year,
            locations="iso3",
            color="co2_per_capita",
            hover_name="country",
            color_continuous_scale="Blues",
            template=THEME_TEMPLATE,
            labels={"co2_per_capita": "CO₂ per hoofd (ton)"},
            title=f"CO₂-uitstoot per inwoner ({selected_year})"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=480)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_rank:
        st.subheader("🌱 Schoonste Top 5")
        if len(df_year) >= 5:
            cleanest = df_year.sort_values(by="co2_per_capita", ascending=True).head(5)
            for _, row in cleanest.iterrows():
                st.markdown(f"• **{row['country']}**: `{row['co2_per_capita']:.2f} t`")
        else:
            st.info("Onvoldoende data.")

        st.subheader("🏭 Hoogste Uitstoters Top 5")
        if len(df_year) >= 5:
            highest = df_year.sort_values(by="co2_per_capita", ascending=False).head(5)
            for _, row in highest.iterrows():
                st.markdown(f"• **{row['country']}**: `{row['co2_per_capita']:.2f} t`")
        else:
            st.info("Onvoldoende data.")

# --- TAB 2: STATISTISCHE ONDERBOUWING & MODEL (Schoon en zonder overbodige tekst) ---
with tab2:
    st.header("Statistische Onderbouwing: Kwadratische Regressie & Modelanalyse")
    st.markdown("""
    Hier toetsen we de **Environmental Kuznets Curve (EKC)** via een kwadratisch model ($CO_2 = \\beta_0 + \\beta_1 \\cdot BBP + \\beta_2 \\cdot BBP^2$). 
    De grafiek visualiseert de berekende trendlijn en spreiding voor het geselecteerde jaartal.
    """)

    x = df_year['gdp_per_capita'].values
    y = df_year['co2_per_capita'].values
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[valid], y[valid]

    if len(x_clean) > 5:
        coeffs = np.polyfit(x_clean, y_clean, 2)
        a, b, c = coeffs
        
        p = np.poly1d(coeffs)
        y_pred = p(x_clean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        tipping_point = -b / (2 * a) if a < 0 else None

        # KPI Metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Model Goede Fit ($R^2$)", f"{r_squared:.3f}")
        mcol2.metric("Kwadratische Coëfficiënt ($\beta_2$)", f"{a:.2e}")
        if tipping_point and tipping_point > 0:
            mcol3.metric("Theoretisch Kantelpunt BBP", f"${tipping_point:,.0f}")
        else:
            mcol3.metric("Theoretisch Kantelpunt", "Niet bereikt")

        st.markdown("---")

        # Scatterplot met fit
        fig_scatter = px.scatter(
            df_year,
            x="gdp_per_capita",
            y="co2_per_capita",
            hover_name="country",
            template=THEME_TEMPLATE,
            labels={"gdp_per_capita": "BBP per hoofd (USD)", "co2_per_capita": "CO₂ per hoofd (ton)"},
            title=f"Economische Welvaart versus CO₂-uitstoot ({selected_year})"
        )
        fig_scatter.update_traces(marker=dict(color=PRIMARY_COLOR, size=8, opacity=0.8))

        x_range = np.linspace(x_clean.min(), x_clean.max(), 200)
        fig_scatter.add_trace(go.Scatter(
            x=x_range, y=p(x_range), mode='lines', name='Kwadratische Fit (EKC)',
            line=dict(color=ACCENT_COLOR, width=3)
        ))
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Onvoldoende datapunten voor dit jaartal en de actieve filters.")

# --- TAB 3: DISTRIBUTIE & BOXPLOTTEN ---
with tab3:
    st.header("📦 Uitgebreide Distributie & Uitbijter Analyse")
    st.markdown("""
    Dit tabblad combineert **boxplots** (voor het visualiseren van de medianen, spreiding en uitschieters via de IQR-methode) 
    met **statistische samenvattingen** per variabele.
    """)

    bcol1, bcol2 = st.columns(2)
    
    with bcol1:
        st.subheader("CO₂-uitstoot per hoofd")
        fig_box_co2 = px.box(
            df_year, 
            y="co2_per_capita", 
            template=THEME_TEMPLATE,
            title=f"Boxplot CO₂ ({selected_year})",
            labels={"co2_per_capita": "CO₂ per hoofd (ton)"},
            color_discrete_sequence=[PRIMARY_COLOR]
        )
        fig_box_co2.update_layout(height=350)
        st.plotly_chart(fig_box_co2, use_container_width=True)

        co2_stats = df_year['co2_per_capita'].describe()
        st.markdown(f"""
        * **Gemiddelde:** `{co2_stats['mean']:.2f} ton`
        * **Mediaan (Q2):** `{co2_stats['50%']:.2f} ton`
        * **Interkwartielafstand (IQR):** `{co2_stats['75%'] - co2_stats['25%']:.2f} ton`
        """)

    with bcol2:
        st.subheader("BBP per hoofd")
        fig_box_gdp = px.box(
            df_year, 
            y="gdp_per_capita", 
            template=THEME_TEMPLATE,
            title=f"Boxplot BBP ({selected_year})",
            labels={"gdp_per_capita": "BBP per hoofd (USD)"},
            color_discrete_sequence=[PRIMARY_COLOR]
        )
        fig_box_gdp.update_layout(height=350)
        st.plotly_chart(fig_box_gdp, use_container_width=True)

        gdp_stats = df_year['gdp_per_capita'].describe()
        st.markdown(f"""
        * **Gemiddelde:** `${gdp_stats['mean']:,.0f}`
        * **Mediaan (Q2):** `${gdp_stats['50%']:,.0f}`
        * **Interkwartielafstand (IQR):** `${gdp_stats['75%'] - gdp_stats['25%']:,.0f}`
        """)

# --- TAB 4: CATEGORIE & TOP-LANDEN ---
with tab4:
    st.header("Gerichte Categorie-Analyse")
    st.markdown("Schakel eenvoudig tussen economische klassen met de knoppen hieronder om de top-uitstoters per categorie te inspecteren.")

    categorie = st.radio(
        "Kies een economische categorie:",
        ["Rijke Economieën (BBP > $40k)", "Opkomende Markten ($10k - $40k)", "Ontwikkelingslanden (BBP < $10k)"],
        horizontal=True
    )

    if "Rijke" in categorie:
        df_filtered = df_year[df_year['gdp_per_capita'] > 40000]
    elif "Opkomende" in categorie:
        df_filtered = df_year[(df_year['gdp_per_capita'] >= 10000) & (df_year['gdp_per_capita'] <= 40000)]
    else:
        df_filtered = df_year[df_year['gdp_per_capita'] < 10000]

    st.info(f"Aantal landen binnen deze categorie in **{selected_year}**: **{len(df_filtered)}** landen")

    if len(df_filtered) > 0:
        fig_bar = px.bar(
            df_filtered.sort_values(by="co2_per_capita", ascending=False).head(10),
            x="country",
            y="co2_per_capita",
            template=THEME_TEMPLATE,
            labels={"country": "Land", "co2_per_capita": "CO₂ per hoofd (ton)"},
            title=f"Top 10 hoogste uitstoters binnen deze categorie ({selected_year})"
        )
        fig_bar.update_traces(marker_color=PRIMARY_COLOR)
        fig_bar.update_layout(height=450)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Geen landen gevonden binnen deze selectie.")