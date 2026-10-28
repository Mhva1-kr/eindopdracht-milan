import requests
import io
import pandas as pd
import numpy as np

print("1. Bezig met ophalen van Bron 1 (Wereldbank API: BBP per hoofd)...")
def fetch_wb_gdp():
    records = []
    page = 1
    while True:
        url = f"http://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&per_page=1000&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Fout bij ophalen Wereldbank API (status {response.status_code})")
        data = response.json()
        if len(data) < 2 or not data[1]:
            break
        for entry in data[1]:
            iso = entry.get('countryiso3code')
            if iso and len(iso) == 3:
                date = entry.get('date')
                val = entry.get('value')
                if date and val is not None:
                    records.append({
                        'iso3': iso,
                        'country': entry.get('country', {}).get('value'),
                        'year': int(date),
                        'gdp_per_capita': float(val)
                    })
        if page >= data[0].get('pages', 1):
            break
        page += 1
    return pd.DataFrame(records)

df_gdp = fetch_wb_gdp()

print("2. Bezig met ophalen van Bron 2 (CO₂-uitstoot per hoofd via requests)...")
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
resp = requests.get(co2_url)
if resp.status_code != 200:
    raise Exception("Fout bij ophalen van CO₂-data via requests")
df_co2_raw = pd.read_csv(io.StringIO(resp.text))

# Filter benodigde kolommen voor CO2
df_co2 = df_co2_raw[['iso_code', 'year', 'co2_per_capita']].rename(columns={'iso_code': 'iso3'})
df_co2 = df_co2.dropna(subset=['iso3', 'co2_per_capita'])
df_co2['year'] = df_co2['year'].astype(int)

print("3. Twee afzonderlijke bronnen samenvoegen op samengestelde sleutel (ISO3 + Jaartal)...")
# Dit dekt Eis 1 volledig af: expliciete merge op een sleutel die niet vanzelf werkt
df_merged = pd.merge(df_gdp, df_co2, on=['iso3', 'year'], how='inner')

# Opschonen en filteren op relevante periode (1990 tot 2024)
df_clean = df_merged.dropna(subset=['gdp_per_capita', 'co2_per_capita']).copy()
df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= 2024)]

# Eventuele dubbele landkolom oplossen
if 'country_y' in df_clean.columns:
    df_clean = df_clean.rename(columns={'country_x': 'country'}).drop(columns=['country_y'])

# Eis 4: Zelf afgeleide kolom (Kwadratische welvaart voor de Kuznets-curve)
df_clean['gdp_per_capita_sq'] = df_clean['gdp_per_capita'] ** 2

# Opslaan als lokaal bestand (vereist tijdens examen)
output_filename = "co2_gdp_data.csv"
df_clean.to_csv(output_filename, index=False)
print(f"ETL-proces voltooid! {len(df_clean)} schone rijen opgeslagen in '{output_filename}'.")