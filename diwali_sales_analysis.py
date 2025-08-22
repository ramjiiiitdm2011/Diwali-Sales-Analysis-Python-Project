import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for saving images
if not os.path.exists("images"):
    os.makedirs("images")

# ✅ Load CSV directly from GitHub (raw link)
csv_url = "https://raw.githubusercontent.com/ramjiiiitdm2011/Diwali-Sales-Analysis-Python-Project/main/Diwali%20Sales%20Data.csv"
df = pd.read_csv(csv_url, encoding='ISO-8859-1')

# Display the first 10 rows of the dataset
print(df.head(10))

# Print the shape of the dataset
print(df.shape)

# Display info about the dataset
print("Info about the CSV data file:")
df.info()

# Drop unrelated or blank columns
df.drop(['Status', 'unnamed1'], axis=1, inplace=True)

# Confirm shape and info after dropping columns
print(df.shape)
df.info()

# Check for null values
print(pd.isnull(df).sum())

# Drop rows with null values
df.dropna(inplace=True)

# Confirm no null values remain
print(pd.isnull(df).sum())
print(df.shape)

# Convert the 'Amount' column to integer type
df['Amount'] = df['Amount'].astype('int')
print(df['Amount'].dtypes)

# Display column names
print(df.columns)

# Rename 'Marital_Status' to 'Shaadi' (not in-place)
df.rename(columns={'Marital_Status': 'Shaadi'})
print(df.columns)

# Display descriptive statistics of the dataset
print(df.describe())

# ==============================
# Exploratory Data Analysis (EDA)
# ==============================

# Gender-wise count plot
palette = {'M': 'blue', 'F': 'deeppink'}
ax = sns.countplot(x='Gender', data=df, palette=palette)
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Gender Distribution')
plt.savefig("images/gender_distribution.png", bbox_inches="tight")
plt.show()

# Total sales by gender
sales_gen = df.groupby(['Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(x='Gender', y='Amount', data=sales_gen)
plt.title('Total Sales by Gender')
plt.savefig("images/total_sales_by_gender.png", bbox_inches="tight")
plt.show()

# Count plot of Age Group by Gender
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, x='Age Group', hue='Gender')
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.0f')
plt.xticks(rotation=45)
plt.title('Age Group Distribution by Gender')
plt.tight_layout()
plt.savefig("images/age_group_distribution.png", bbox_inches="tight")
plt.show()

# Total sales by age group
sales_age = df.groupby('Age Group', as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(x='Age Group', y='Amount', data=sales_age, palette='Set2')
plt.title('Total Sales by Age Group')
plt.savefig("images/sales_by_age_group.png", bbox_inches="tight")
plt.show()

# Top 10 states by number of orders
sales_state = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)
sns.set(rc={'figure.figsize': (15, 5)})
sns.barplot(data=sales_state, x='State', y='Orders', palette='Set2')
plt.title('Top 10 States by Orders')
plt.savefig("images/top_states_by_orders.png", bbox_inches="tight")
plt.show()

# Top 10 states by total sales
sales_state = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
sns.set(rc={'figure.figsize': (15, 5)})
sns.barplot(data=sales_state, x='State', y='Amount', palette='Set2')
plt.title('Top 10 States by Sales')
plt.savefig("images/top_states_by_sales.png", bbox_inches="tight")
plt.show()

# Count plot of Marital Status
df['Marital_Status'] = df['Marital_Status'].astype(str)  # Ensure type for plotting
sns.set(rc={'figure.figsize': (7, 5)})
ax = sns.countplot(data=df, x='Marital_Status', palette='Set3')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Marital Status Distribution')
plt.savefig("images/marital_status_distribution.png", bbox_inches="tight")
plt.show()

# Total sales by marital status and gender
sales_state = df.groupby(['Marital_Status', 'Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.set(rc={'figure.figsize': (6, 5)})
sns.barplot(data=sales_state, x='Marital_Status', y='Amount', hue='Gender', palette='Set1')
plt.title('Sales by Marital Status and Gender')
plt.savefig("images/sales_by_marital_status_gender.png", bbox_inches="tight")
plt.show()

# Total sales by occupation
sales_state = df.groupby(['Occupation'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.set(rc={'figure.figsize': (20, 5)})
sns.barplot(data=sales_state, x='Occupation', y='Amount', palette='Set1')
plt.title('Sales by Occupation')
plt.savefig("images/sales_by_occupation.png", bbox_inches="tight")
plt.show()

# Count plot of product categories
sns.set(rc={'figure.figsize': (20, 5)})
ax = sns.countplot(data=df, x='Product_Category', palette='Set1')
for bars in ax.containers:
    ax.bar_label(bars)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.title('Product Category Distribution')
plt.savefig("images/product_category_distribution.png", bbox_inches="tight")
plt.show()

# Top 10 product categories by total sales
sales_state = df.groupby(['Product_Category'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
sns.set(rc={'figure.figsize': (20, 5)})
sns.barplot(data=sales_state, x='Product_Category', y='Amount', palette='Set1')
plt.title('Top 10 Product Categories by Sales')
plt.xticks(rotation=45, ha='right')
plt.savefig("images/top_product_categories_by_sales.png", bbox_inches="tight")
plt.show()

# Top 10 products by number of orders
sales_state = df.groupby(['Product_ID'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)
sns.set(rc={'figure.figsize': (20, 5)})
sns.barplot(data=sales_state, x='Product_ID', y='Orders', palette='Set1')
plt.title('Top 10 Products by Orders')
plt.xticks(rotation=45, ha='right')
plt.savefig("images/top_products_by_orders.png", bbox_inches="tight")
plt.show()

# Bar chart of top 10 most sold products
fig1, ax1 = plt.subplots(figsize=(12, 7))
df.groupby('Product_ID')['Orders'].sum().nlargest(10).sort_values(ascending=False).plot(kind='bar')
plt.title('Top 10 Most Sold Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig("images/top_most_sold_products.png", bbox_inches="tight")
plt.show()
