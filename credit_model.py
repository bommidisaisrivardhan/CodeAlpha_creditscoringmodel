import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop unnecessary columns
drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Month']
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Drop rows with missing values
train_df = train_df.dropna()

# Convert 'Credit_History_Age' to numeric (in months)
def convert_age_to_months(age_str):
    try:
        years, months = 0, 0
        if 'Years' in age_str:
            years = int(age_str.split('Years')[0].strip())
        if 'Months' in age_str:
            months = int(age_str.split('Years')[-1].split('Months')[0].strip())
        return years * 12 + months
    except:
        return 0

train_df['Credit_History_Age'] = train_df['Credit_History_Age'].apply(convert_age_to_months)

# Encode categorical features
categorical_cols = ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Mix']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Define features and target
features = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Mix',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_Credit_Inquiries'
]

target = 'Credit_Score'

# Drop rows where target is missing
train_df = train_df.dropna(subset=[target])

# Encode target
train_df[target] = LabelEncoder().fit_transform(train_df[target])

print("\n📊 Feature types:\n", train_df[features].dtypes)
# Ensure all feature columns are numeric
for col in features:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

# Drop rows where feature columns have missing values after conversion
train_df.dropna(subset=features, inplace=True)

# Train/test split
X = train_df[features]
y = train_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'))
print("✅ Credit Scoring Model Finished.")
