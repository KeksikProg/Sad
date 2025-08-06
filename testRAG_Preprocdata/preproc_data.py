import pandas as pd
from tools import extract_cleaned_text
import re

df = pd.read_excel('./data/very_very_low_data.xlsx')
df.columns = ['code', 'text']
df['text'] = df['text'].apply(extract_cleaned_text)
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
df['text'] = df['text'].str.replace(r'_x000C_', '', regex=False)
df['text'] = df['text'].str.replace(r'•', '', regex=False)
df['text'] = df['text'].str.replace(r';', '.', regex=False)
df['text'] = df['text'].str.replace(r'', '', regex=False)
df['text'] = df['text'].str.replace(r'\uf0b7', '', regex=False)
df['text'] = df['text'].str.strip()
df = df.dropna(subset=['text'])
df = df.drop_duplicates()

print(df['code'].value_counts())
df.to_csv('data/clear_data.csv', index=False)
