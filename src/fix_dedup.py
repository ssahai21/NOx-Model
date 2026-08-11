import pandas as pd

df = pd.read_csv('data/df_final.csv')
print(f"Before dedup: {len(df)} rows")

df = df.drop_duplicates(subset=['plot_id', 'fire_name', 'date'])
print(f"After dedup:  {len(df)} rows")

print("\nRows per fire:")
print(df['fire_name'].value_counts())

df.to_csv('data/nox_clean.csv', index=False)
print("\nSaved to data/nox_clean.csv")