import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf

st.set_page_config(
    page_title="CO2-uitstoot vs Welvaart Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Eenduidig kleurenpalet
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#d62728"
THEME_TEMPLATE = "plotly_white"

@st.cache_data
def load_data():
    return pd.read_csv("co2_gdp_data.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Lokaal bestand 'co2_gdp_data.csv' niet gevonden! Voer eerst het script 'etl.py' uit.")
    st.stop()

# --- HOOFDBOODSCHAP / HOOFDCONCLUSIE BOVENAAN ---
st.title("🌍 CO₂-Uitstoot & Economische Welvaart (EKC Analyse)")
st.info("""
**Hoofdboodschap:** Dit dashboard toetst of economische groei automatisch leidt tot vergroening (de Environmental Kuznets Curve). 
**Belangrijkste inzicht:** Hoewel rijke landen per inwoner meer uitstoten dan arme landen, vlakt de stijging af. 
Toch blijkt uit de statistische toetsing dat economische groei alleen **niet voldoende** is om CO₂-emissies tijdig te laten dalen.
""")

# --- SIDEBAR BEDIENING ---
st.sidebar.header("🎛️ Dashboard Besturing")
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_year = st.sidebar.slider("📅 Selecteer Jaartal", min_value=min_year, max_value=max_year, value=max_year)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Regressie Opties")
filter_outliers_regression = st.sidebar.checkbox(
    "Verwijder uitschieters (1.5 × IQR) ALLEEN voor het regressiemodel", 
    value=False,
    help="De wereldkaart blijft de volledige dataset tonen om de data-integriteit te bewaken."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Categorie Drempels (Instelbaar)")
low_thresh = st.sidebar.slider("Drempel Ontwikkeling vs Opkomend ($)", 2000, 15000, 10000, 1000)
high_thresh = st.sidebar.slider("Drempel Opkomend vs Rijk ($)", 20000, 60000, 40000, 2000)

# Filter dataset op gekozen jaartal (Volledige dataset voor kaart/kpi)
df_year_full = df[df['year'] == selected_year].dropna(subset=['gdp_per_capita', 'co2_per_capita']).copy()

# Dynamische her-categorisering op basis van sidebar sliders
def dynamic_categorize(gdp):
    if gdp > high_thresh:
        return f"Rijk (> ${high_thresh/1000:.0f}k)"
    elif gdp >= low_thresh:
        return f"Opkomend (${low_thresh/1000:.0f}k - ${high_thresh/1000:.0f}k)"
    else:
        return f"Ontwikkeling (< ${low_thresh/1000:.0f}k)"

df_year_full['dynamic_income_group'] = df_year_full['gdp_per_capita'].apply(dynamic_categorize)

# Gefilterde dataset voor statistische regressie (indien IQR aangevinkt)
df_year_reg = df_year_full.copy()
if filter_outliers_regression and len(df_year_reg) > 10:
    q1_co2 = df_year_reg['co2_per_capita'].quantile(0.25)
    q3_co2 = df_year_reg['co2_per_capita'].quantile(0.75)
    iqr_co2 = q3_co2 - q1_co2
    
    q1_gdp = df_year_reg['gdp_per_capita'].quantile(0.25)
    q3_gdp = df_year_reg['gdp_per_capita'].quantile(0.75)
    iqr_gdp = q3_gdp - q1_gdp

    df_year_reg = df_year_reg[
        (df_year_reg['co2_per_capita'] >= q1_co2 - 1.5 * iqr_co2) & 
        (df_year_reg['co2_per_capita'] <= q3_co2 + 1.5 * iqr_co2) &
        (df_year_reg['gdp_per_capita'] >= q1_gdp - 1.5 * iqr_gdp) & 
        (df_year_reg['gdp_per_capita'] <= q3_gdp + 1.5 * iqr_gdp)
    ]

# OLS Regressie berekenen via Statsmodels (Gebruikt de afgeleide kolom gdp_per_capita_sq)
model_quad = None
model_lin = None
r2_quad, r2_lin, p_val_sq, tipping_point, n_obs = None, None, None, None, len(df_year_reg)

if n_obs > 10:
    # Model 1: Kwadratisch (EKC) - Expliciet met gdp_per_capita_sq uit ETL!
    model_quad = smf.ols('co2_per_capita ~ gdp_per_capita + gdp_per_capita_sq', data=df_year_reg).fit()
    r2_quad = model_quad.rsquared
    p_val_sq = model_quad.pvalues.get('gdp_per_capita_sq', 1.0)
    
    b1 = model_quad.params.get('gdp_per_capita', 0)
    b2 = model_quad.params.get('gdp_per_capita_sq', 0)
    if b2 < 0:
        tipping_point = -b1 / (2 * b2)

    # Model 2: Lineair (om te vergelijken wat een rechte lijn zou missen)
    model_lin = smf.ols('co2_per_capita ~ gdp_per_capita', data=df_year_reg).fit()
    r2_lin = model_lin.rsquared

# Markeer landen ten opzichte van kantelpunt op de wereldkaart
if tipping_point and tipping_point > 0:
    df_year_full['ekc_status'] = df_year_full['gdp_per_capita'].apply(
        lambda x: "Boven Kantelpunt" if x >= tipping_point else "Onder Kantelpunt"
    )
else:
    df_year_full['ekc_status'] = "Onder Kantelpunt"

# --- TABBLADEN STRUCTUUR ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Wereldkaart & Globale KPI's", 
    "📈 Regressie & Hypothesetoetsing", 
    "📦 Verdeling & Welvaartsgroepen", 
    "🔘 Dynamic Categorie Analyse"
])

# --- TAB 1: KAART & KPI'S ---
with tab1:
    st.header(f"Geospatiale Wereldspreiding in {selected_year}")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Aantal Landen (Volledig)", len(df_year_full))
    kpi2.metric("Gemiddeld BBP per hoofd", f"${df_year_full['gdp_per_capita'].mean():,.0f}")
    kpi3.metric("Gemiddelde CO₂ per hoofd", f"{df_year_full['co2_per_capita'].mean():.2f} ton")
    kpi4.metric("Mediaan CO₂ per hoofd", f"{df_year_full['co2_per_capita'].median():.2f} ton")

    st.markdown("---")
    
    col_map_opt, _ = st.columns([1, 2])
    with col_map_opt:
        map_color_mode = st.radio("Kleurkaart op basis van:", ["Absolute CO₂-uitstoot", "Positie t.o.v. EKC-Kantelpunt"], horizontal=True)

    if map_color_mode == "Absolute CO₂-uitstoot":
        fig_map = px.choropleth(
            df_year_full,
            locations="iso3",
            color="co2_per_capita",
            hover_name="country",
            color_continuous_scale="Blues",
            template=THEME_TEMPLATE,
            labels={"co2_per_capita": "CO₂ per hoofd (ton)"},
            title=f"CO₂-uitstoot per inwoner ({selected_year})"
        )
    else:
        fig_map = px.choropleth(
            df_year_full,
            locations="iso3",
            color="ekc_status",
            hover_name="country",
            color_discrete_map={"Onder Kantelpunt": PRIMARY_COLOR, "Boven Kantelpunt": ACCENT_COLOR},
            template=THEME_TEMPLATE,
            title=f"Landen gekleurd naar EKC-kantelpunt status ({selected_year})"
        )

    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 2: STATISTISCHE HYPOTHESETOETSING ---
with tab2:
    st.header("Statistische Hypothesetoetsing: Kwadratische OLS Regressie")
    st.markdown(r"""
    We toetsen de hypothese: $CO_2 = \beta_0 + \beta_1 \cdot BBP + \beta_2 \cdot BBP^2 + \epsilon$.
    De afgeleide kolom `gdp_per_capita_sq` uit de ETL wordt hier expliciet meegegeven aan de **Statsmodels OLS-engine**.
    """)

    if model_quad is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Aantal Observaties (n)", f"{n_obs}")
        m2.metric(r"Kwadratisch Model $R^2$", f"{r2_quad:.3f}")
        m3.metric(r"Lineair Model $R^2$", f"{r2_lin:.3f}", delta=f"{r2_quad - r2_lin:+.3f} winst")
        
        p_status = "Significant (p < 0.05)" if p_val_sq < 0.05 else "Niet Significant (p ≥ 0.05)"
        m4.metric(r"p-waarde $\beta_2$", f"{p_val_sq:.4f}", delta=p_status, delta_color="normal")

        st.markdown("---")

        col_scatter, col_stats = st.columns([2, 1])

        with col_scatter:
            fig_scatter = px.scatter(
                df_year_reg,
                x="gdp_per_capita",
                y="co2_per_capita",
                hover_name="country",
                template=THEME_TEMPLATE,
                labels={"gdp_per_capita": "BBP per hoofd (USD)", "co2_per_capita": "CO₂ per hoofd (ton)"},
                title=f"EKC Kwadratische Regressielijn vs. Lineaire Fit ({selected_year})"
            )

            # Fit curves berekenen
            x_vals = np.linspace(df_year_reg['gdp_per_capita'].min(), df_year_reg['gdp_per_capita'].max(), 200)
            df_pred = pd.DataFrame({'gdp_per_capita': x_vals, 'gdp_per_capita_sq': x_vals**2})
            
            y_quad = model_quad.predict(df_pred)
            y_lin = model_lin.predict(df_pred)

            fig_scatter.add_trace(go.Scatter(x=x_vals, y=y_quad, mode='lines', name='Kwadratisch Model (EKC)', line=dict(color=ACCENT_COLOR, width=3)))
            fig_scatter.add_trace(go.Scatter(x=x_vals, y=y_lin, mode='lines', name='Rechte Lijn (Lineair)', line=dict(color='gray', dash='dash')))

            fig_scatter.update_layout(height=480)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_stats:
            st.subheader("📋 Regressieresultaten")
            conf_int = model_quad.conf_int().loc['gdp_per_capita_sq']
            
            st.markdown(f"""
            * **Kwadratische Term ($\beta_2$):** `{model_quad.params['gdp_per_capita_sq']:.2e}`
            * **95% Betrouwbaarheidsinterval:** 
              `[{conf_int[0]:.2e}, {conf_int[1]:.2e}]`
            * **Berekend Kantelpunt:** {f'${tipping_point:,.0f}' if tipping_point else 'Geen omgekeerde U'}
            """)

            st.markdown("---")
            st.markdown("**Wat mist een rechte lijn?**")
            st.write(f"Een lineair model heeft een $R^2$ van **{r2_lin:.3f}**. Het kwadratische model verhoogt de verklarende kracht naar **{r2_quad:.3f}**. Een rechte lijn veronderstelt dat CO₂ oneindig blijft stijgen bij meer welvaart, wat de werkelijkheid vertekent.")
    else:
        st.warning("Onvoldoende datapunten voor betrouwbare regressie.")

# --- TAB 3: VERDELING & BOXPLOTTEN ---
with tab3:
    st.header("📦 Datadistributie & Spreiding")
    st.markdown("Inzage in de verdeling van de variabelen per economische welvaartsklasse.")

    fig_box_cat = px.box(
        df_year_full,
        x="dynamic_income_group",
        y="co2_per_capita",
        color="dynamic_income_group",
        template=THEME_TEMPLATE,
        title=f"CO₂-uitstoot per inwoner verdeeld over Inkomensklassen ({selected_year})",
        labels={"dynamic_income_group": "Inkomensklasse", "co2_per_capita": "CO₂ per hoofd (ton)"}
    )
    fig_box_cat.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_box_cat, use_container_width=True)

# --- TAB 4: CATEGORIE ANALYSE ---
with tab4:
    st.header("🔘 Dynamische Categorie Analyse")
    st.markdown("Analyseer de hoogste uitstoters binnen de instelbare economische drempels uit de sidebar.")

    selected_cat = st.radio(
        "Selecteer een categorie om te inspecteren:",
        df_year_full['dynamic_income_group'].unique(),
        horizontal=True
    )

    df_cat_filtered = df_year_full[df_year_full['dynamic_income_group'] == selected_cat]

    st.info(f"Aantal landen in **{selected_cat}**: **{len(df_cat_filtered)}** landen")

    if len(df_cat_filtered) > 0:
        fig_bar = px.bar(
            df_cat_filtered.sort_values(by="co2_per_capita", ascending=False).head(10),
            x="country",
            y="co2_per_capita",
            template=THEME_TEMPLATE,
            labels={"country": "Land", "co2_per_capita": "CO₂ per hoofd (ton)"},
            title=f"Top 10 Hoogste Uitstoters binnen {selected_cat} ({selected_year})"
        )
        fig_bar.update_traces(marker_color=PRIMARY_COLOR)
        fig_bar.update_layout(height=450)
        st.plotly_chart(fig_bar, use_container_width=True)