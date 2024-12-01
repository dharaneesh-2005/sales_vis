import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('vgsales.csv')

# Data Cleaning
# Fill missing 'Year' with median and 'Publisher' with 'Unknown'
df.loc[:, 'Year'] = df['Year'].fillna(df['Year'].median())
df.loc[:, 'Publisher'] = df['Publisher'].fillna('Unknown')


# Exploratory Data Analysis (EDA)
# 1. Global Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Global_Sales'], bins=30, kde=True)
plt.title('Distribution of Global Sales (in millions)')
plt.show()

# 2. Sales by Genre
plt.figure(figsize=(10, 6))
sns.barplot(x='Genre', y='Global_Sales', data=df, estimator=sum)
plt.xticks(rotation=45)
plt.title('Total Global Sales by Genre')
plt.show()

# 3. Sales by Platform
plt.figure(figsize=(12, 6))
sns.barplot(x='Platform', y='Global_Sales', data=df, estimator=sum)
plt.xticks(rotation=45)
plt.title('Total Global Sales by Platform')
plt.show()

# 4. Sales Trend Over Years
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Global_Sales', data=df, estimator=sum)
plt.title('Global Sales Over the Years')
plt.show()

# 5. Top Publishers by Sales
top_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(10)
top_publishers.plot(kind='bar', figsize=(10, 5))
plt.title('Top 10 Publishers by Global Sales')
plt.show()

# Model Building
# Select features and target
df = pd.get_dummies(df, columns=['Platform', 'Genre', 'Publisher'], drop_first=True)
X = df.drop(columns=['Name', 'Global_Sales', 'Rank'])
y = df['Global_Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.show()
