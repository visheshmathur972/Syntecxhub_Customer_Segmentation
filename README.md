# Customer Segmentation using K-Means Clustering

## Project Overview

This project performs customer segmentation using K-Means Clustering.

The Mall Customers dataset is analyzed using customer age, annual income, and spending score to identify different customer groups.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses the Mall Customers dataset.

Main features used:

- Age
- Annual Income (k$)
- Spending Score (1-100)

CustomerID was removed because it is not useful for clustering.

## Project Steps

1. Load the dataset
2. Check dataset shape and missing values
3. Remove CustomerID
4. Perform data visualization
5. Select clustering features
6. Scale the data using StandardScaler
7. Find the optimal number of clusters using the Elbow Method
8. Apply K-Means Clustering
9. Visualize customer clusters
10. Profile the customer segments
11. Save the final segmented dataset

## Customer Segments

The customers were divided into 4 clusters:

- Older Medium-Spending Customers
- High Income High Spending Customers
- Young High-Spending Customers
- High Income Low-Spending Customers

## Output

The project generates:

- Age distribution graph
- Income distribution graph
- Spending score distribution graph
- Elbow Method graph
- Customer cluster visualization
- Final customer segments CSV file

## Conclusion

K-Means clustering successfully groups customers based on their age, annual income, and spending behavior. These segments can help businesses understand customer behavior and develop targeted marketing strategies.