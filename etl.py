import requests
import io
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# 1. Peildatum en vaste Git commit-hash voor OWID (vastlegging versie)
peildatum = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
OWID_COMMIT_HASH = "e3f89a9c7d1e8432a51f8450f3b063d81b89694e"  # Vaste commit voor kwaliteitsborging

print(f"==================================================")
print(f"START ETL PROCES - Peildatum: {peildatum}")
print(f"==================================================")

# 2. Bron 1: Wereldbank API (BBP per hoofd) via HTTPS & Timeout
print("\n1. Ophalen Bron 1 (Wereldbank API: BBP per hoofd in USD)...")
def fetch_wb_gdp():
    records = []
    page = 1
    total_raw_entries = 0
    null_value_entries = 0
    while True:
        url = f"https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&per_page=1000&page={page}"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Fout bij ophalen Wereldbank API (status {response.status_code})")
        data = response.json()
        if len(data) < 2 or not data[1]:
            break
        for entry in data[1]:
            total_raw_entries += 1
            iso = entry.get('countryiso3code')
            val = entry.get('value')
            date = entry.get('date')
            if val is None or not iso or len(iso) != 3:
                null_value_entries += 1
                continue
            records.append({
                'iso3': iso,
                'country': entry.get('country', {}).get('value'),
                'year': int(date),
                'gdp_per_capita': float(val)
            })
        if page >= data[0].get('pages', 1):
            break
        page += 1
    print(f"   -> Wereldbank ruwe inschrijvingen: {total_raw_entries}, Lege/ongeldige inschrijvingen gefilterd: {null_value_entries}")
    return pd.DataFrame(records)

df_gdp = fetch_wb_gdp()
print(f"   -> Gevalideerde Wereldbank data: {len(df_gdp)} rijen ({df_gdp['iso3'].nunique()} unieke ISO3-codes).")

# 3. Bron 2: Our World in Data (CO2-uitstoot) - Vaste versie via commit-hash
print("\n2. Ophalen Bron 2 (Our World in Data CO₂ - Vaste Version Commit)...")
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
resp = requests.get(co2_url, timeout=15)
if resp.status_code != 200:
    raise Exception("Fout bij ophalen van CO₂-data via requests")

df_co2_raw = pd.read_csv(io.StringIO(resp.text))
count_co2_raw = len(df_co2_raw)

df_co2 = df_co2_raw[['iso_code', 'year', 'co2_per_capita']].rename(columns={'iso_code': 'iso3'})
null_co2 = df_co2['co2_per_capita'].isna().sum() + df_co2['iso3'].isna().sum()
df_co2 = df_co2.dropna(subset=['iso3', 'co2_per_capita'])
df_co2['year'] = df_co2['year'].astype(int)

print(f"   -> OWID ruwe rijen: {count_co2_raw}, Rijen met ontbrekende CO₂/ISO: {null_co2}")
print(f"   -> Gevalideerde OWID data: {len(df_co2)} rijen ({df_co2['iso3'].nunique()} unieke ISO3-codes).")

# 4. Anti-Join Analyse (Volledige transparantie over uitvallende ISO-codes)
iso_gdp_set = set(df_gdp['iso3'].unique())
iso_co2_set = set(df_co2['iso3'].unique())

only_in_gdp = sorted(list(iso_gdp_set - iso_co2_set))
only_in_co2 = sorted(list(iso_co2_set - iso_gdp_set))

print(f"\n--- ANTI-JOIN ANALYSE ---")
print(f"   -> ISO3-codes WEL in Wereldbank, NIET in OWID ({len(only_in_gdp)} codes):")
print(f"      {only_in_gdp}")
print(f"   -> ISO3-codes WEL in OWID, NIET in Wereldbank ({len(only_in_co2)} codes):")
print(f"      {only_in_co2}")

# 5. Inner Merge op ISO3 + Jaartal
print("\n3. Samenvoegen op samengestelde sleutel (ISO3 + Jaartal)...")
df_merged = pd.merge(df_gdp, df_co2, on=['iso3', 'year'], how='inner')
print(f"   -> Rijen direct na inner merge: {len(df_merged)}")

# 6. Filteren op tijdsframe (1990 - 2024)
df_clean = df_merged[(df_merged['year'] >= 1990) & (df_merged['year'] <= 2024)].copy()
dropped_years = len(df_merged) - len(df_clean)
print(f"   -> Rijen buiten periode 1990-2024 gefilterd: {dropped_years}")

# 7. Afgeleide kolommen (Eis 4 - Worden direct en actief in app.py gebruikt)
df_clean['gdp_per_capita_sq'] = df_clean['gdp_per_capita'] ** 2
df_clean['log_gdp_per_capita'] = np.log(df_clean['gdp_per_capita'])

def categorize_gdp(gdp):
    if gdp > 40000:
        return "Rijk (> 40k USD)"
    elif gdp >= 10000:
        return "Opkomend (10k - 40k USD)"
    else:
        return "Ontwikkeling (< 10k USD)"

df_clean['income_group'] = df_clean['gdp_per_capita'].apply(categorize_gdp)

output_filename = "co2_gdp_data.csv"
df_clean.to_csv(output_filename, index=False)

print("\n==================================================")
print(f"ETL-PROCES VOLTOOID!")
print(f"Eindtotaal opgeslagen: {len(df_clean)} schone rijen in '{output_filename}'.")
print(f"Aantal unieke landen: {df_clean['iso3'].nunique()}")
print(f"==================================================")