"""
✅ COMPLETE SALES PREDICTION SCRIPT - FINAL VERSION
⚠️ DELETE YOUR OLD sales_prediction.py AND COPY THIS ENTIRE SCRIPT!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD YOUR CSV FILE
# ============================================
df = pd.read_csv('Amazon Sale Report.csv')  # CHANGE THIS TO YOUR FILE NAME
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# ============================================
# 2. STORE ORIGINAL VALUES (BEFORE ANY CHANGES)
# ============================================
original_order_id = df['Order ID'].copy() if 'Order ID' in df.columns else None
original_state = df['ship-state'].copy() if 'ship-state' in df.columns else None

# ============================================
# 3. REMOVE USELESS COLUMNS
# ============================================
columns_to_drop = ['ASIN', 'SKU', 'ship-postal-code', 'Unnamed: 22', 'index']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# ============================================
# 3.5 STANDARDIZE TEXT COLUMNS (FIX DUPLICATES)
# ============================================
# Convert all text columns to uppercase to fix duplicates (e.g., "Maharashtra" vs "MAHARASHTRA")
text_cols_to_standardize = ['ship-state', 'ship-city', 'ship-country', 'Status', 'Fulfilment', 
                             'Sales Channel ', 'ship-service-level', 'Category', 'Size', 
                             'Courier Status', 'fulfilled-by', 'currency', 'Style']

for col in text_cols_to_standardize:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()

print("\nColumns after cleanup:")
print(df.columns.tolist())

# ============================================
# 4. HANDLE MISSING VALUES
# ============================================
print("\nMissing values BEFORE:")
print(df.isnull().sum())

df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Amount'].fillna(df['Amount'].median(), inplace=True)

df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
df['Qty'].fillna(0, inplace=True)

df['Courier Status'].fillna('Unknown', inplace=True)
df['fulfilled-by'].fillna('Unknown', inplace=True)
df['promotion-ids'].fillna('No_Promotion', inplace=True)

print("\nMissing values AFTER:")
print(df.isnull().sum())

# ============================================
# 5. FEATURE ENGINEERING - DATES
# ============================================
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter

# ============================================
# 6. CREATE CLASSIFICATION TARGET (HIGH/LOW AMOUNT)
# ============================================
amount_median = df['Amount'].median()
print(f"\nAmount median (threshold for HIGH/LOW): ₹{amount_median:.2f}")

df['Amount_Category'] = (df['Amount'] > amount_median).astype(int)
print(f"HIGH orders: {(df['Amount_Category'] == 1).sum()}")
print(f"LOW orders: {(df['Amount_Category'] == 0).sum()}")

# ============================================
# 7. ENCODE TEXT COLUMNS (SAVE MAPPINGS)
# ============================================
le_dict = {}
text_columns = ['Status', 'Fulfilment', 'Sales Channel ', 'ship-service-level', 'Category', 'Size', 
                'ship-state', 'ship-city', 'ship-country', 'B2B', 'Courier Status', 'fulfilled-by', 'currency', 'Style']

for col in text_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Handle promotion-ids
df['Has_Promotion'] = (df['promotion-ids'] != 'No_Promotion').astype(int)
df = df.drop('promotion-ids', axis=1)

print("\nColumns after encoding:")
print(df.columns.tolist())

# ============================================
# 8. PREPARE FEATURES (X) AND TARGETS (Y)
# ============================================
columns_to_drop_final = ['Date', 'Order ID', 'Amount', 'Amount_Category']
features = [col for col in df.columns if col not in columns_to_drop_final]
features = [col for col in features if col in df.columns and df[col].dtype in ['int64', 'float64']]

# ================== CRITICAL FIX ==================
# Remove rows where regression target is missing
df_reg = df.dropna(subset=['Amount']).copy()

X = df_reg[features].copy()
y_regression = df_reg['Amount'].copy()
y_classification = df_reg['Amount_Category'].copy()

# Safety check (optional but recommended)
print("NaN in regression target:", y_regression.isna().sum())


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print(f"\nFeatures used: {features}")
print(f"Number of features: {len(features)}")

# ============================================
# 9. SPLIT DATA (80% train, 20% test)
# ============================================
X_train, X_test, y_reg_train, y_reg_test, idx_train, idx_test = train_test_split(
    X_scaled, y_regression, X_scaled.index, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    X_scaled, y_classification, test_size=0.2, random_state=42
)

# Get original values for test set
test_order_ids = original_order_id.iloc[idx_test].values if original_order_id is not None else None
test_states_encoded = original_state.iloc[idx_test].values if original_state is not None else None

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================================
# 10. MODEL A: REGRESSION (Predict EXACT Amount)
# ============================================
print("\n" + "="*70)
print("MODEL A: REGRESSION - Predict EXACT Individual Order Amount")
print("="*70)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_train, y_reg_train)

y_reg_pred = model_reg.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

feature_importance_reg = pd.DataFrame({
    'Feature': features,
    'Importance': model_reg.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Important Features (Regression):")
print(feature_importance_reg.head(10).to_string())

# ============================================
# 11. MODEL B: CLASSIFICATION (Predict HIGH/LOW)
# ============================================
print("\n" + "="*70)
print("MODEL B: CLASSIFICATION - Predict HIGH or LOW Order Amount")
print("="*70)

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_train, y_clf_train)

y_clf_pred = model_clf.predict(X_test)

accuracy = accuracy_score(y_clf_test, y_clf_pred)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_clf_test, y_clf_pred, target_names=['LOW', 'HIGH']))

feature_importance_clf = pd.DataFrame({
    'Feature': features,
    'Importance': model_clf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Important Features (Classification):")
print(feature_importance_clf.head(10).to_string())

# ============================================
# 12. TIME SERIES: PREDICT NEXT 12 MONTHS BY INDIVIDUAL STATE
# ============================================
print("\n" + "="*70)
print("MODEL C: TIME SERIES - Monthly Sales Forecast by Individual State")
print("="*70)

state_le = le_dict.get('ship-state', None)

# Create monthly sales by state
df['Year_Month'] = df['Date'].dt.to_period('M')
state_monthly = df.groupby(['ship-state', 'Year_Month'])['Amount'].sum().reset_index()
state_monthly['Year_Month'] = state_monthly['Year_Month'].dt.to_timestamp()

# Get top 10 states by total sales
top_states = df.groupby('ship-state')['Amount'].sum().nlargest(10).index.tolist()

forecast_all_states = []

print(f"\n📊 FORECASTING TOP 10 STATES FOR NEXT 12 MONTHS:\n")

for state_code in top_states:
    state_data = state_monthly[state_monthly['ship-state'] == state_code].sort_values('Year_Month').reset_index(drop=True)
    
    # Get state name
    if state_le is not None:
        state_name = state_le.inverse_transform([state_code])[0]
    else:
        state_name = f"State_{state_code}"
    
    if len(state_data) < 2:
        continue
    
    # Time series forecasting
    X_ts = np.arange(len(state_data)).reshape(-1, 1)
    y_ts = state_data['Amount'].values
    
    model_ts = LinearRegression()
    model_ts.fit(X_ts, y_ts)
    
    # Predict next 12 months
    future_months = np.arange(len(state_data), len(state_data) + 12).reshape(-1, 1)
    future_sales = model_ts.predict(future_months)
    
    # Create future dates
    last_date = state_data['Year_Month'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    # Calculate trend
    current_avg = state_data['Amount'].tail(3).mean()
    future_avg = future_sales[:3].mean()
    trend = "📈 UP" if future_avg > current_avg else "📉 DOWN"
    
    print(f"{state_name.upper()}")
    print(f"Current 3-month Avg: ₹{current_avg:.2f} | Next 3-month Forecast: ₹{future_avg:.2f} | Trend: {trend}")
    print(f"{'Month':<12} {'Forecast':>15} {'Trend':>10}")
    print("-" * 40)
    
    for i, (date, sales) in enumerate(zip(future_dates, future_sales)):
        month_trend = "📈" if sales > current_avg else "📉"
        print(f"{date.strftime('%Y-%m'):<12} ₹{sales:>13,.0f} {month_trend:>10}")
        
        forecast_all_states.append({
            'State_Name': state_name,
            'Year_Month': date.strftime('%Y-%m'),
            'Predicted_Sales': sales,
            'Trend': trend
        })
    
    print("\n")

# Create forecast dataframe
forecast_df = pd.DataFrame(forecast_all_states)

# Summary by state
print("\n" + "="*70)
print("SUMMARY: TOTAL FORECASTED SALES BY STATE (Next 12 Months)")
print("="*70)

state_forecast_summary = forecast_df.groupby(['State_Name', 'Trend'])['Predicted_Sales'].sum().reset_index()
state_forecast_summary = state_forecast_summary.sort_values('Predicted_Sales', ascending=False)
state_forecast_summary.columns = ['State_Name', 'Overall_Trend', 'Total_Forecasted_Sales_12M']

print("\n")
print(state_forecast_summary.to_string(index=False))

# ============================================
# 13. MODEL D: ANALYZE DECLINING STATES & ROOT CAUSES
# ============================================
print("\n" + "="*70)
print("MODEL D: DECLINING STATES ANALYSIS - Root Cause Analysis")
print("="*70)

# Split data into two periods (early vs recent)
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
mid_date = df['Date'].median()

early_period = df[df['Date'] < mid_date].copy()
recent_period = df[df['Date'] >= mid_date].copy()

state_le = le_dict.get('ship-state', None)

# Analyze each state
state_analysis = []

for state_code in df['ship-state'].unique():
    early_state = early_period[early_period['ship-state'] == state_code]
    recent_state = recent_period[recent_period['ship-state'] == state_code]
    
    if len(early_state) == 0 or len(recent_state) == 0:
        continue
    
    # Get state name
    if state_le is not None:
        state_name = state_le.inverse_transform([state_code])[0]
    else:
        state_name = f"State_{state_code}"
    
    # Calculate metrics for both periods
    early_sales = early_state['Amount'].sum()
    recent_sales = recent_state['Amount'].sum()
    sales_change = ((recent_sales - early_sales) / early_sales * 100) if early_sales > 0 else 0
    
    early_orders = len(early_state)
    recent_orders = len(recent_state)
    order_change = ((recent_orders - early_orders) / early_orders * 100) if early_orders > 0 else 0
    
    early_cancelled = (early_state['Status'] == 'Cancelled').sum()
    recent_cancelled = (recent_state['Status'] == 'Cancelled').sum()
    early_cancel_rate = (early_cancelled / len(early_state) * 100) if len(early_state) > 0 else 0
    recent_cancel_rate = (recent_cancelled / len(recent_state) * 100) if len(recent_state) > 0 else 0
    cancel_rate_change = recent_cancel_rate - early_cancel_rate
    
    early_avg_order = early_state['Amount'].mean()
    recent_avg_order = recent_state['Amount'].mean()
    avg_order_change = ((recent_avg_order - early_avg_order) / early_avg_order * 100) if early_avg_order > 0 else 0
    
    early_promo = (early_state['Has_Promotion'] == 1).sum() / len(early_state) * 100 if len(early_state) > 0 else 0
    recent_promo = (recent_state['Has_Promotion'] == 1).sum() / len(recent_state) * 100 if len(recent_state) > 0 else 0
    promo_change = recent_promo - early_promo
    
    early_b2b = (early_state['B2B'] == 1).sum() / len(early_state) * 100 if len(early_state) > 0 else 0
    recent_b2b = (recent_state['B2B'] == 1).sum() / len(recent_state) * 100 if len(recent_state) > 0 else 0
    b2b_change = recent_b2b - early_b2b
    
    state_analysis.append({
        'State': state_name,
        'Sales_Change_%': sales_change,
        'Order_Count_Change_%': order_change,
        'Avg_Order_Change_%': avg_order_change,
        'Cancellation_Rate_Change_%': cancel_rate_change,
        'Promotion_Usage_Change_%': promo_change,
        'B2B_Orders_Change_%': b2b_change,
        'Early_Total_Sales': early_sales,
        'Recent_Total_Sales': recent_sales,
        'Early_Avg_Order': early_avg_order,
        'Recent_Avg_Order': recent_avg_order,
        'Early_Cancel_Rate_%': early_cancel_rate,
        'Recent_Cancel_Rate_%': recent_cancel_rate,
        'Early_Promo_%': early_promo,
        'Recent_Promo_%': recent_promo
    })

analysis_df = pd.DataFrame(state_analysis)

# Identify declining states (negative sales change)
declining_states = analysis_df[analysis_df['Sales_Change_%'] < 0].sort_values('Sales_Change_%')

print(f"\n🔴 DECLINING STATES (Sales Going DOWN):\n")

if len(declining_states) == 0:
    print("✅ No declining states! All states showing growth or stable sales.")
else:
    for idx, row in declining_states.head(10).iterrows():
        print(f"\n{'='*70}")
        print(f"STATE: {str(row['State']).upper()}")
        print(f"{'='*70}")
        print(f"Sales Decline: {row['Sales_Change_%']:.2f}%")
        print(f"  Early Period: ₹{row['Early_Total_Sales']:,.0f}")
        print(f"  Recent Period: ₹{row['Recent_Total_Sales']:,.0f}")
        
        print(f"\n📊 ROOT CAUSE ANALYSIS:")
        print(f"-" * 70)
        
        # Identify main reasons
        reasons = []
        
        # Reason 1: Fewer orders
        if row['Order_Count_Change_%'] < -5:
            reasons.append(f"❌ ORDER DECLINE: {row['Order_Count_Change_%']:.1f}% fewer orders")
        elif row['Order_Count_Change_%'] > 5:
            reasons.append(f"✅ More orders: +{row['Order_Count_Change_%']:.1f}%")
        else:
            reasons.append(f"➡️ Orders stable: {row['Order_Count_Change_%']:.1f}% change")
        
        # Reason 2: Lower average order value
        if row['Avg_Order_Change_%'] < -5:
            reasons.append(f"❌ LOWER VALUES: Avg order ↓{abs(row['Avg_Order_Change_%']):.1f}%")
            reasons.append(f"   (₹{row['Early_Avg_Order']:.0f} → ₹{row['Recent_Avg_Order']:.0f})")
        elif row['Avg_Order_Change_%'] > 5:
            reasons.append(f"✅ Higher order values: +{row['Avg_Order_Change_%']:.1f}%")
        else:
            reasons.append(f"➡️ Order values stable: {row['Avg_Order_Change_%']:.1f}% change")
        
        # Reason 3: Increased cancellations
        if row['Cancellation_Rate_Change_%'] > 2:
            reasons.append(f"❌ CANCELLATIONS UP: +{row['Cancellation_Rate_Change_%']:.1f}%")
            reasons.append(f"   (Cancel rate: {row['Early_Cancel_Rate_%']:.1f}% → {row['Recent_Cancel_Rate_%']:.1f}%)")
        elif row['Cancellation_Rate_Change_%'] < -2:
            reasons.append(f"✅ Fewer cancellations: {row['Cancellation_Rate_Change_%']:.1f}%")
        
        # Reason 4: Less promotions
        if row['Promotion_Usage_Change_%'] < -5:
            reasons.append(f"❌ LESS PROMOTIONS: Usage ↓{abs(row['Promotion_Usage_Change_%']):.1f}%")
            reasons.append(f"   ({row['Early_Promo_%']:.1f}% → {row['Recent_Promo_%']:.1f}%)")
        elif row['Promotion_Usage_Change_%'] > 5:
            reasons.append(f"✅ More promotions: +{row['Promotion_Usage_Change_%']:.1f}%")
        
        # Reason 5: B2B decline
        if row['B2B_Orders_Change_%'] < -5:
            reasons.append(f"❌ B2B DECLINE: {row['B2B_Orders_Change_%']:.1f}%")
        elif row['B2B_Orders_Change_%'] > 5:
            reasons.append(f"✅ More B2B orders: +{row['B2B_Orders_Change_%']:.1f}%")
        
        for reason in reasons:
            print(f"{reason}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"-" * 70)
        
        # Generate recommendations based on root causes
        recommendations = []
        
        if row['Order_Count_Change_%'] < -5:
            recommendations.append("• Increase marketing efforts in this state")
            recommendations.append("• Launch targeted campaigns & discounts")
            recommendations.append("• Improve delivery speed & reduce shipping costs")
        
        if row['Avg_Order_Change_%'] < -5:
            recommendations.append("• Offer bundle deals to increase basket size")
            recommendations.append("• Run upselling campaigns for premium products")
            recommendations.append("• Create value packs for popular categories")
        
        if row['Cancellation_Rate_Change_%'] > 2:
            recommendations.append("• Improve product descriptions & images")
            recommendations.append("• Ensure faster delivery to reduce cancellations")
            recommendations.append("• Check fulfillment quality in this region")
            recommendations.append("• Verify payment gateway issues")
        
        if row['Promotion_Usage_Change_%'] < -5:
            recommendations.append("• Design more attractive promotional offers")
            recommendations.append("• Use SMS/email to notify customers of deals")
            recommendations.append("• Partner with local influencers for promotions")
        
        if row['B2B_Orders_Change_%'] < -5:
            recommendations.append("• Create B2B-specific discounts & packages")
            recommendations.append("• Establish B2B partnerships in this region")
            recommendations.append("• Offer payment terms & bulk order benefits")
        
        if not recommendations:
            recommendations.append("• Monitor this state closely for further changes")
            recommendations.append("• Conduct customer surveys to understand needs")
            recommendations.append("• Consider temporary incentives to boost sales")
        
        for rec in list(set(recommendations))[:5]:
            print(rec)

# Save analysis
analysis_df.to_csv('state_analysis_detailed.csv', index=False)
declining_states.to_csv('declining_states_analysis.csv', index=False)

print(f"\n{'='*70}\n")

# ============================================
# 14. STATE-WISE SALES PREDICTIONS (UP/DOWN)
# ============================================
print("\n" + "="*70)
print("STATE-WISE SALES PREDICTIONS (UP/DOWN Trend)")
print("="*70)

state_le = le_dict.get('ship-state', None)

# State-wise sales analysis
state_sales = df.groupby('ship-state').agg({
    'Amount': ['sum', 'count', 'mean']
}).reset_index()
state_sales.columns = ['state_encoded', 'Total_Sales', 'Order_Count', 'Avg_Order']

# Decode state names
state_sales['State_Name'] = state_le.inverse_transform(state_sales['state_encoded'])

# Sort by total sales
state_sales = state_sales.sort_values('Total_Sales', ascending=False)

print(f"\nTop 15 States by Total Sales:")
print(state_sales[['State_Name', 'Total_Sales', 'Order_Count', 'Avg_Order']].head(15).to_string(index=False))

# Predict state-wise trend using overall median
overall_median = state_sales['Total_Sales'].median()
state_sales['Trend'] = state_sales['Total_Sales'].apply(
    lambda x: "📈 UP" if x > overall_median else "📉 DOWN"
)

print(f"\n✅ STATE-WISE SALES TREND:")
print(state_sales[['State_Name', 'Total_Sales', 'Trend']].head(15).to_string(index=False))

# ============================================
# 15. INDIVIDUAL PREDICTIONS WITH STATE NAMES & ORDER IDS
# ============================================
print("\n" + "="*70)
print("MODEL A DETAILED: INDIVIDUAL ORDER PREDICTIONS")
print("(With State Names, Order IDs, and Predictions)")
print("="*70)

# test_states_encoded already contains original state names (no conversion needed!)
predictions_df_a = pd.DataFrame({
    'Order_ID': test_order_ids,
    'State_Name': test_states_encoded,
    'Actual_Amount': y_reg_test.values,
    'Predicted_Amount': y_reg_pred,
    'Error': abs(y_reg_test.values - y_reg_pred)
})

predictions_df_a = predictions_df_a.sort_values('Predicted_Amount', ascending=False)

print("\nTop 20 Predicted Orders (Highest to Lowest):")
print(predictions_df_a[['Order_ID', 'State_Name', 'Actual_Amount', 'Predicted_Amount', 'Error']].head(20).to_string(index=False))

predictions_df_a.to_csv('MODEL_A_predictions.csv', index=False)

# ============================================
# 16. CLASSIFICATION PREDICTIONS WITH STATE NAMES & ORDER IDS
# ============================================
print("\n" + "="*70)
print("MODEL B DETAILED: CLASSIFICATION PREDICTIONS")
print("(With State Names, Order IDs, HIGH/LOW, and Rest)")
print("="*70)

predictions_df_b = pd.DataFrame({
    'Order_ID': test_order_ids,
    'State_Name': test_states_encoded,
    'Actual_Amount': y_reg_test.values,
    'Category': ['HIGH' if x == 1 else 'LOW' for x in y_clf_pred],
    'Actual_Category': ['HIGH' if x == 1 else 'LOW' for x in y_clf_test.values],
    'Predicted_Amount': y_reg_pred
})

predictions_df_b = predictions_df_b.sort_values('Predicted_Amount', ascending=False)

print("\nTop 20 Classification Predictions:")
print(predictions_df_b[['Order_ID', 'State_Name', 'Actual_Amount', 'Category', 'Actual_Category', 'Predicted_Amount']].head(20).to_string(index=False))

predictions_df_b.to_csv('MODEL_B_predictions.csv', index=False)

# ============================================
# SAVE ALL FILES
# ============================================
forecast_df.to_csv('state_wise_monthly_forecast.csv', index=False)
state_sales.to_csv('state_wise_current_sales.csv', index=False)

print("\n" + "="*70)
print("💾 SAVING MODELS FOR WEB APP...")
print("="*70)

import joblib

joblib.dump(model_reg, 'model_regression.pkl')
print("✅ Saved: model_regression.pkl")

joblib.dump(model_clf, 'model_classification.pkl')
print("✅ Saved: model_classification.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("✅ Saved: scaler.pkl")

joblib.dump(le_dict, 'label_encoders.pkl')
print("✅ Saved: label_encoders.pkl")

print("="*70)
print("✅ ALL MODELS SAVED! Ready for web app!")
print("="*70)