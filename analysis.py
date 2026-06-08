# ── E-Commerce Customer Behaviour & Sales Performance Analysis ──
# Tools: Python, Pandas, NumPy, Matplotlib, SQL-style analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_excel('online_retail.xlsx')
print("Dataset loaded!")
print("Shape:", df.shape)
print(df.head())

# ─────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────
df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['Revenue']     = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month']       = df['InvoiceDate'].dt.to_period('M')

print("\nCleaned shape:", df.shape)
print("Total Revenue: £{:,.2f}".format(df['Revenue'].sum()))

# ─────────────────────────────────────────
# 3. MONTHLY REVENUE TREND
# ─────────────────────────────────────────
monthly_revenue = df.groupby('Month')['Revenue'].sum().reset_index()
monthly_revenue['Month'] = monthly_revenue['Month'].astype(str)

plt.figure(figsize=(14, 5))
plt.plot(monthly_revenue['Month'], monthly_revenue['Revenue'],
         marker='o', color='#1B3A5C', linewidth=2)
plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
plt.tight_layout()
plt.savefig('monthly_revenue.png', dpi=150)
plt.show()
print("Monthly revenue chart saved!")

# ─────────────────────────────────────────
# 4. TOP 10 PRODUCTS BY REVENUE
# ─────────────────────────────────────────
top_products = (df.groupby('Description')['Revenue']
                .sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(12, 5))
top_products.plot(kind='barh', color='#1B3A5C')
plt.title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Revenue (£)')
plt.gca().xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
plt.tight_layout()
plt.savefig('top_products.png', dpi=150)
plt.show()
print("Top products chart saved!")

# ─────────────────────────────────────────
# 5. TOP 10 COUNTRIES BY REVENUE
# ─────────────────────────────────────────
top_countries = (df.groupby('Country')['Revenue']
                 .sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(12, 5))
top_countries.plot(kind='bar', color='#1B3A5C')
plt.title('Top 10 Countries by Revenue', fontsize=14, fontweight='bold')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
plt.tight_layout()
plt.savefig('top_countries.png', dpi=150)
plt.show()
print("Top countries chart saved!")

# ─────────────────────────────────────────
# 6. RFM CUSTOMER SEGMENTATION
# ─────────────────────────────────────────
snapshot_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('InvoiceNo',   'nunique'),
    Monetary  = ('Revenue',     'sum')
).reset_index()

rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1,2,3,4])
rfm['RFM_Score'] = (rfm['R_Score'].astype(int) +
                    rfm['F_Score'].astype(int) +
                    rfm['M_Score'].astype(int))

def segment(score):
    if score >= 10: return 'Champions'
    elif score >= 7: return 'Loyal Customers'
    elif score >= 5: return 'At Risk'
    else: return 'Lost'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)
print("\nCustomer Segments:")
print(rfm['Segment'].value_counts())

rfm['Segment'].value_counts().plot(
    kind='bar', color=['#1B3A5C','#2E6DA4','#6FA3D0','#B8D3EC'])
plt.title('Customer Segments', fontsize=14, fontweight='bold')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('customer_segments.png', dpi=150)
plt.show()
print("Customer segments chart saved!")

# ─────────────────────────────────────────
# 7. KEY BUSINESS INSIGHT — PARETO 80/20
# ─────────────────────────────────────────
rfm_sorted  = rfm.sort_values('Monetary', ascending=False)
top_20_cut  = int(len(rfm_sorted) * 0.20)
top_20_rev  = rfm_sorted.head(top_20_cut)['Monetary'].sum()
total_rev   = rfm_sorted['Monetary'].sum()
pct         = (top_20_rev / total_rev) * 100

print(f"\nTotal Customers : {len(rfm_sorted)}")
print(f"Top 20%         : {top_20_cut} customers")
print(f"Their Revenue   : £{top_20_rev:,.2f}")
print(f"% of Total      : {pct:.1f}%")

# ─────────────────────────────────────────
# 8. EXPORT CSVs FOR POWER BI
# ─────────────────────────────────────────
df.to_csv('ecommerce_cleaned.csv', index=False)
rfm.to_csv('rfm_segments.csv', index=False)
monthly_revenue.to_csv('monthly_revenue.csv', index=False)

print("\n✅ All done! 3 CSV files exported for Power BI:")
print("   → ecommerce_cleaned.csv")
print("   → rfm_segments.csv")
print("   → monthly_revenue.csv")
print("\n✅ 4 chart images saved:")
print("   → monthly_revenue.png")
print("   → top_products.png")
print("   → top_countries.png")
print("   → customer_segments.png")