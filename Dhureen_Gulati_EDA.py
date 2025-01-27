import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

transac_path = '/Users/dhureengulati/Downloads/Transactions.csv'
prod_path = '/Users/dhureengulati/Downloads/Products.csv'
cust_path = '/Users/dhureengulati/Downloads/Customers.csv'

# Loading data in dataframe 
transac_df = pd.read_csv(transac_path)
prod_df = pd.read_csv(prod_path)
cust_df = pd.read_csv(cust_path)

# Print first few columns of data to verify the data is clean
transactions_head = transac_df.head()
products_head = prod_df.head()
customers_head = cust_df.head()

transactions_info = transac_df.info()
products_info = prod_df.info()
customers_info = cust_df.info()

# Dates -> datetime format
cust_df["SignupDate"] = pd.to_datetime(cust_df['SignupDate'])
transac_df['TransactionDate'] = pd.to_datetime(transac_df['TransactionDate'])


customers_summary = cust_df.describe(include='all')
products_summary = prod_df.describe(include='all')
transactions_summary = transac_df.describe()

#customer distribution by region
plt.figure(figsize=(8, 5))
sns.countplot(data=cust_df, x='Region', order=cust_df['Region'].value_counts().index) #occurence per region
plt.title('Customer Distribution by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#product by category
plt.figure(figsize=(8, 5))
sns.countplot(data=prod_df, y='Category', order=prod_df['Category'].value_counts().index) #occurence per category sorted by freq
plt.title('Product Distribution by Category')
plt.tight_layout()
plt.show()

# Plot monthly sales trend
transac_df['TransactionMonth'] = transac_df['TransactionDate'].dt.to_period('M')
monthly_sales = transac_df.groupby('TransactionMonth')['TotalValue'].sum()

plt.figure(figsize=(12, 5))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Value')
plt.grid()
plt.tight_layout()
plt.show()

# Plot top 10 customers by total spending
top_customers = transac_df.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_customers.index, y=top_customers.values, palette='viridis')
plt.title('Top 10 Customers by Total Spending')
plt.xlabel('CustomerID')
plt.ylabel('Total Spending (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Popular Products by Quantity Sold
popular_products = transac_df.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=popular_products.index, y=popular_products.values, palette='coolwarm')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('ProductID')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#merge transac_df,cust_df and prod_df
merged_data=transac_df.merge(cust_df, on='CustomerID').merge(prod_df,on='ProductID')
reg_sales=merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=reg_sales.index, y=reg_sales.values, palette='Blues_d')
plt.title('Regional Sales Distribution')
plt.ylabel('Total Sales')
plt.show()

# Sales by Product Category
category_sales = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=category_sales.index, y=category_sales.values, palette='Oranges')
plt.title('Total Sales by Product Category')
plt.ylabel('Total Sales')
plt.show()

# Output the summaries
customers_summary, products_summary, transactions_summary
