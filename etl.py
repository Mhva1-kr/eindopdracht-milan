import requests
import io
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# 1. Vastleggen peildatum en metadata voor data-verantwoording
peildatum = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"==================================================")
print(f"START ETL PROCES - Peildatum: {peildatum}")
print(f"==================================================")

# 2. Bron 1: Wereldbank API (BBP per hoofd) via HTTPS & Timeout
print("\n1. Bezig met ophalen van Bron 1 (Wereldbank API: BBP per hoofd)...")
def fetch_wb_gdp():
    records = []
    page = 1
    while True:
        url = f"https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&per_page=1000&page={page}"
        response = requests.get(url, timeout=15)
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
count_gdp_raw = len(df_gdp)
print(f"   -> Bron 1 (Wereldbank) opgehaald: {count_gdp_raw} observaties ({df_gdp['iso3'].nunique()} unieke ISO3-codes).")

# 3. Bron 2: Our World in Data (CO2-uitstoot)
print("\n2. Bezig met ophalen van Bron 2 (Our World in Data CO₂)...")
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
resp = requests.get(co2_url, timeout=15)
if resp.status_code != 200:
    raise Exception("Fout bij ophalen van CO₂-data via requests")

df_co2_raw = pd.read_csv(io.StringIO(resp.text))
df_co2 = df_co2_raw[['iso_code', 'year', 'co2_per_capita']].rename(columns={'iso_code': 'iso3'})
df_co2 = df_co2.dropna(subset=['iso3', 'co2_per_capita'])
df_co2['year'] = df_co2['year'].astype(int)
count_co2_raw = len(df_co2)
print(f"   -> Bron 2 (OWID) opgehaald: {count_co2_raw} observaties ({df_co2['iso3'].nunique()} unieke ISO3-codes).")

# 4. Anti-Join Analyse voor Data-Verantwoording (Welke ISO's vallen uit?)
iso_gdp_set = set(df_gdp['iso3'].unique())
iso_co2_set = set(df_co2['iso3'].unique())

only_in_gdp = iso_gdp_set - iso_co2_set
only_in_co2 = iso_co2_set - iso_gdp_set

print(f"\n--- ANTI-JOIN ANALYSE (Transparantie over uitvallers) ---")
print(f"   -> ISO3-codes wel in Wereldbank, NIEUWS in OWID ({len(only_in_gdp)}): {sorted(list(only_in_gdp))[:10]}...")
print(f"   -> ISO3-codes wel in OWID, NIET in Wereldbank ({len(only_in_co2)}): {sorted(list(only_in_co2))[:10]}...")

# 5. Inner Merge op ISO3 + Jaartal
print("\n3. Twee afzonderlijke bronnen samenvoegen op samengestelde sleutel (ISO3 + Jaartal)...")
df_merged = pd.merge(df_gdp, df_co2, on=['iso3', 'year'], how='inner')
print(f"   -> Rijen direct na inner merge: {len(df_merged)}")

# 6. Filteren & Opschonen
df_clean = df_merged.dropna(subset=['gdp_per_capita', 'co2_per_capita']).copy()
df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= 2024)]

# 7. Zelf afgeleide kolommen (Eis 4)
# (A) Kwadraat van BBP voor expliciete OLS regressie
df_clean['gdp_per_capita_sq'] = df_clean['gdp_per_capita'] ** 2
# (B) Log BBP vanwege scheve verdeling
df_clean['log_gdp_per_capita'] = np.log(df_clean['gdp_per_capita'])
# (C) Economische Categorieën direct in ETL afleiden
def categorize_gdp(gdp):
    if gdp > 40000:
        return "Rijk (> $40k)"
    elif gdp >= 10000:
        return "Opkomend ($10k - $40k)"
    else:
        return "Ontwikkeling (< $10k)"

df_clean['income_group'] = df_clean['gdp_per_capita'].apply(categorize_gdp)

output_filename = "co2_gdp_data.csv"
df_clean.to_csv(output_filename, index=False)

print("\n==================================================")
print(f"ETL-PROCES VOLTOOID!")
print(f"Eindtotaal: {len(df_clean)} schone rijen opgeslagen in '{output_filename}'.")
print(f"Aantal unieke landen in eindrapport: {df_clean['iso3'].nunique()}")
print(f"==================================================")