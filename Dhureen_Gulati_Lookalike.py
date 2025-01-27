import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_customer_data(transaction_path, product_path, customer_path):
    """
    Load and combine customer, transaction, and product data from CSV files.
    Returns a merged dataset with all relevant information.
    """
    # Read individual data files
    transactions = pd.read_csv(transaction_path)
    products = pd.read_csv(product_path)
    customers = pd.read_csv(customer_path)
    
    # Combine all data into a single comprehensive dataset
    return transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

def create_customer_profiles(merged_data):
    """
    Build detailed profiles for each customer based on their shopping behavior.
    Includes total spending, number of purchases, and category preferences.
    """
    # Create a summary of each customer's shopping behavior
    customer_profiles = merged_data.groupby('CustomerID').agg({
        'TotalValue': 'sum',         # How much they've spent in total
        'TransactionID': 'count',    # How many times they've shopped
        'Category': lambda x: x.value_counts().to_dict()  # What types of products they buy
    }).reset_index()
    
    # Convert shopping categories into separate columns for easier comparison
    product_categories = merged_data['Category'].unique()
    for category in product_categories:
        customer_profiles[category] = customer_profiles['Category'].apply(
            lambda x: x.get(category, 0)  # How many items bought in each category
        )
    
    # Remove the original category column since we've broken it out
    customer_profiles.drop(columns=['Category'], inplace=True)
    
    return customer_profiles, product_categories

def normalize_customer_data(customer_profiles, numerical_columns):
    """
    Scale all numerical values to be between 0 and 1 for fair comparison.
    This ensures that different metrics (like spending and visit count) are comparable.
    """
    scaler = MinMaxScaler()
    customer_profiles[numerical_columns] = scaler.fit_transform(customer_profiles[numerical_columns])
    return customer_profiles

def find_similar_customers(customer_profiles, numerical_columns, top_n=3):
    """
    Find the most similar customers for each customer based on their shopping patterns.
    Uses Euclidean distance to measure similarity - smaller distance means more similar.
    """
    formatted_results = []
    
    # Compare each customer with every other customer
    for i, current_customer in enumerate(customer_profiles['CustomerID']):
        distances = []
        current_profile = customer_profiles.loc[i, numerical_columns].values
        
        # Calculate similarity with all other customers
        for j, other_customer in enumerate(customer_profiles['CustomerID']):
            if current_customer != other_customer:
                other_profile = customer_profiles.loc[j, numerical_columns].values
                similarity = np.sqrt(np.sum((current_profile - other_profile) ** 2))
                distances.append((other_customer, similarity))
        
        # Get the top most similar customers
        most_similar = sorted(distances, key=lambda x: x[1])[:top_n]
        
        # Format the results in a readable way
        result = {
            'CustomerID': current_customer
        }
        for i, (similar_customer, similarity_score) in enumerate(most_similar, 1):
            result[f'Similar_Customer_{i}'] = similar_customer
            result[f'Similarity_Score_{i}'] = round(similarity_score, 4)
            
        formatted_results.append(result)
    
    return pd.DataFrame(formatted_results)

def main():
    # File paths for our data
    transaction_path = '/Users/dhureengulati/Downloads/Transactions.csv'
    product_path = '/Users/dhureengulati/Downloads/Products.csv'
    customer_path = '/Users/dhureengulati/Downloads/Customers.csv'
    
    # Step 1: Load and combine all our data
    print("Loading and combining data...")
    merged_data = load_customer_data(transaction_path, product_path, customer_path)
    
    # Step 2: Create detailed customer profiles
    print("Creating customer profiles...")
    customer_profiles, product_categories = create_customer_profiles(merged_data)
    
    # Step 3: Prepare data for comparison
    numerical_columns = ['TotalValue', 'TransactionID'] + list(product_categories)
    customer_profiles = normalize_customer_data(customer_profiles, numerical_columns)
    
    # Step 4: Find similar customers
    print("Finding similar customers...")
    similar_customers = find_similar_customers(customer_profiles, numerical_columns)
    
    # Step 5: Filter for first 20 customers and save results
    final_results = similar_customers[similar_customers['CustomerID'] <= 'C0020']
    final_results.to_csv('Lookalike.csv', index=False)
    
    print("\nAnalysis complete! Here are the first few results:")
    print(final_results.head())

if __name__ == "__main__":
    main()