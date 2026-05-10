import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title = 'FraudGuard',
    page_icon = '🛡️',
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Section 2
@st.cache_data
def load_data():
    df = pd.read_csv('../data/fraud_clean.csv')
    return df
@st.cache_resource
def build_model(df):
    feature_cols = [c for c in df.columns if c!='isFraud']
    target_col = 'isFraud'
    df = df.sort_values('tx_day_elapsed').reset_index(drop = True)
    split_idx = int(len(df) * 0.80)
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df[target_col].iloc[:split_idx]
    X_test = df[feature_cols].iloc[split_idx:]
    y_test = df[target_col].iloc[split_idx:]

    # SMOTE
    smote = SMOTE(sampling_strategy = 0.1, random_state = 42, k_neighbors = 5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train,y_train)

    # Train Model
    model = XGBClassifier(n_estimators = 300, max_depth = 6,
                          learning_rate = 0.05, subsample = 0.8,
                          colsample_bytree = 0.8, reg_alpha = 0.1,
                          reg_lambda = 1.0, random_state = 42,
                          eval_metric = 'aucpr', verbosity = 0)
    model.fit(X_train_sm,y_train_sm)
    return model, feature_cols, X_test, y_test
df = load_data()
model, feature_cols, X_test,y_test = build_model(df)

# Define global metrics — available to all pages
total_tx = len(df)
total_fraud = df['isFraud'].sum()
fraud_rate = df['isFraud'].mean() * 100
avg_fraud_amt = df[df['isFraud']==1]['TransactionAmt'].mean()

# Section 3
# Sidebar
st.sidebar.markdown("## 🛡️ FraudGuard")
st.sidebar.markdown("---")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Select Page",
    ["Transaction Overview", "Fraud Pattern Analysis", "Transaction Scorer"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Dataset:** IEEE-CIS Fraud Detection
    **Transactions:** 590,540
    **Fraud Rate:** 3.5%
    **Model:** XGBoost + SMOTE
    **ROC-AUC:** 0.896
    """
)

# PAGE 1: TRANSACTION OVERVIEW ─────────────────────────────
if page == "Transaction Overview":

    st.title("🛡️ Transaction Fraud Overview")
    st.markdown("Executive summary of fraud across the portfolio.")
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Fraudulent", f"{total_fraud:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:,.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Fraud Rate by Product Type")

        # Map encoded values back to labels
        prod_map = {0: 'C', 1: 'H', 2: 'R', 3: 'S', 4: 'W'}
        df['ProductCD_label'] = df['ProductCD'].map(prod_map)

        prod_analysis = df.groupby(
            'ProductCD_label')['isFraud'].agg(
            total='count', fraud='sum'
        ).reset_index()
        prod_analysis['fraud_rate'] = (
            prod_analysis['fraud'] /
            prod_analysis['total'] * 100
        ).round(2)
        prod_analysis = prod_analysis.sort_values(
            'fraud_rate', ascending=False)

        fig_prod = px.bar(
            prod_analysis,
            x='ProductCD_label',
            y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale='Reds',
            labels={'fraud_rate': 'Fraud Rate %',
                    'ProductCD_label': 'Product Code'},
            text='fraud_rate'
        )
        fig_prod.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig_prod.add_hline(
            y=fraud_rate,
            line_dash='dash',
            line_color='white',
            annotation_text=f'Avg {fraud_rate:.1f}%'
        )
        fig_prod.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            yaxis=dict(range=[0,
                prod_analysis['fraud_rate'].max() + 3])
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_right:
        st.subheader("Fraud Rate by Card Type")

        # Map encoded values back to labels
        card_map = {0: 'charge card', 1: 'credit',
                    2: 'debit', 3: 'debit or credit'}
        df['card6_label'] = df['card6'].map(card_map)

        card_analysis = df.groupby(
            'card6_label')['isFraud'].agg(
            total='count', fraud='sum'
        ).reset_index()
        card_analysis['fraud_rate'] = (
            card_analysis['fraud'] /
            card_analysis['total'] * 100
        ).round(2)
        card_analysis = card_analysis.sort_values(
            'fraud_rate', ascending=False)

        fig_card = px.bar(
            card_analysis,
            x='card6_label',
            y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale='Oranges',
            labels={'fraud_rate': 'Fraud Rate %',
                    'card6_label': 'Card Type'},
            text='fraud_rate'
        )
        fig_card.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_card.update_layout(
            showlegend=False, coloraxis_showscale=False,
            height=400, yaxis=dict(range=[0,
                card_analysis['fraud_rate'].max() + 3])
        )
        st.plotly_chart(fig_card, use_container_width=True)
        st.caption("⚠️ Charge card and Debit or credit categories "
           "have <10 transactions — rates statistically "
           "unreliable.")

# PAGE 2: FRAUD PATTERN ANALYSIS ───────────────────────────
elif page == "Fraud Pattern Analysis":
    st.title("📊 Fraud Pattern Analysis")
    st.markdown("Transaction patterns and behivioural signals "
                "for fraud analyst briefings.")
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Fraud Rate by Hour of Day")
        hour_analysis = df.groupby('tx_hour')['isFraud'].agg(
            total = 'count', fraud = 'sum'
        ).reset_index()
        hour_analysis['fraud_rate'] = (hour_analysis['fraud']/hour_analysis['total']*100).round(2)

        fig_hour = px.line(hour_analysis, x = 'tx_hour', y = 'fraud_rate', 
                           markers = True, labels = {'fraud_rate': 'Fraud Rate %',
                                                     'tx_hour': 'Hour of the Day'},
        )
        fig_hour.add_hline(y = fraud_rate, line_dash = 'dash',
                           line_color = 'white', annotation_text = f'Avg {fraud_rate:.1f}%')
        fig_hour.update_layout(height = 400)
        st.plotly_chart(fig_hour, use_container_width = True)

    with col_right:
        st.subheader("Transaction Amount - Fraud vs Legitimate")
        fig_amt = px.histogram(df.sample(10000, random_state = 42), 
                               x = 'TransactionAmt', color = 'isFraud', nbins = 50, 
                               barmode = 'overlay', opacity = 0.7, range_x = [0,500],
                               color_discrete_map = {0: '#2ecc71', 1: '#e74c3c'},
                               labels = {'TransactionAmt': 'Transaction Amount ($)', 'isFraud': 'Is Fraud'}
    
        )
        fig_amt.update_layout(height = 400)
        st.plotly_chart(fig_amt,use_container_width = True)

    st.markdown("---")

    # Model Performance section
    st.subheader("Model Performance - Precision Recall Curve")
    y_pred_proba = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    col1,col2,col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{auc:.4f}")
    col2.metric("Average Precision", f"{avg_prec:.4f}")
    col3.metric("Fraud Rate", f"{y_test.mean()*100:.2f}%")

    fig_pr = px.line(x = recall, y = precision, labels = {
        'x': 'Recall (Fraud Caught Rate)',
        'y': 'Precision (Fraud Flag Accuracy)'
    })        
    fig_pr.add_hline(
        y = y_test.mean(),
        line_dash = 'dash', line_color = 'gray', 
        annotation_text = 'Random baseline'
    )
    fig_pr.update_layout(height = 400, title = 'Precision - Recall Curve')
    st.plotly_chart(fig_pr,use_container_width = True)

# PAGE 3: TRANSACTION SCORER ───────────────────────────────
elif page == "Transaction Scorer":
    st.title("🔍 Transaction Scorer")
    st.markdown("Enter transaction details to get an instant "
                " fraud probability score.")
    st.markdown("---")
    st.markdown("### Transaction Details")
    st.caption("input fields reflect the model's strongest "
               "predictive features.")
    col1, col2 = st.columns(2)

    with col1:
        trans_amt = st.number_input("Transaction Amount($)", min_value = 100.0, step = 10.0)
        product_cd = st.selectbox("Product Code", options = [0,1,2,3,4], format_func = lambda x:{
            0:'C - CNP Credit',
            1:'H - Hosted', 
            2:'R - Recurring',
            3:'S - Store',
            4:'W - Web'}[x])
        card_type = st.selectbox("Card Type", options = [0,1,2,3], format_func = lambda x:{
            0:'Charge Card', 1:'Credit', 2:'Debit', 3:'Debit or Credit'}[x])
        tx_hour = st.slider("Hour of transaction (0-23)", min_value = 0, max_value = 23, value = 12)
        is_night = st.selectbox("Hour Transaction (11pm-6am)?", options = [0,1], format_func = lambda x:
                                'Yes' if x == 1 else 'No')
    
    with col2:
        card_tx_count = st.number_input("Total Card Transactions (card activity)",
                                        min_value = 1, max_value = 10000, value = 500)
        card_mean_amt = st.number_input("Card Mean Transaction Amount ($)", min_value = 1.0, max_value = 1000.0,
                                        value = 100.0, step = 10.0)
        is_weekend = st.selectbox("Weekend Transaction?", options = [0,1], format_func = lambda x:
                                  'Yes' if x == 1 else 'No')
        tx_day = st.slider("Day of Week (0=Mon, 6=Sun)", min_value = 0, max_value = 6, value = 2)
        tx_day_elapsed = st.number_input("Day in Dataset (1-182)", min_value = 1, max_value = 182, value = 90)

    st.markdown("---")

    if st.button("🔍 Score Transaction", type="primary"):

        # Calculate derived features
        amt_deviation = (trans_amt - card_mean_amt)/(card_mean_amt + 1)
        amt_ratio = trans_amt/(card_mean_amt + 1)

        # Build input with all 60 features
        input_data = pd.DataFrame([np.zeros(len(feature_cols))], columns = feature_cols)

        # Set known input values
        input_data['TransactionAmt'] = trans_amt
        input_data['ProductCD'] = product_cd
        input_data['card6'] = card_type
        input_data['tx_hour'] = tx_hour
        input_data['is_night'] = is_night
        input_data['is_weekend'] = is_weekend
        input_data['tx_day'] = tx_day
        input_data['tx_day_elapsed'] = tx_day_elapsed
        input_data['card_tx_count'] = card_tx_count
        input_data['card_mean_amt'] = card_mean_amt
        input_data['amt_deviation'] = amt_deviation
        input_data['amt_ratio'] = amt_ratio

        # Score
        fraud_prob = model.predict_proba(input_data)[0][1]

        # Risk Tier
        if fraud_prob >= 0.5:
            tier = "🔴 High Risk"
            colour = "red"
            action = "Block and escalate to Fraud Operations"
        elif fraud_prob >= 0.143:
            tier = "🟡 Suspicious"
            colour = "orange"
            action = "Flag for manual review"
        else:
            tier = "🟢 Low Risk"
            colour = "green"
            action = "Approve — monitor for pattern changes"

        # Display Results
        st.markdown("### Scoring Result")
        res1,res2,res3 = st.columns(3)
        res1.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
        res2.metric("Risk Tier", tier)
        res3.metric("Amt vs Card baseline", f"{amt_deviation*100:+.1f}%")

        st.markdown(f"**Recommended Action:** {action}")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(mode = "gauge + number", value = fraud_prob * 100, 
                                           title = {'text':"Fraud probability (%)"}, 
                                           gauge = {'axis': {'range':[0,100]},
                                                    'bar': {'color': colour},
                                                    'steps': [
                                                        {'range': [0,14.3],
                                                         'color': 'rgba(46,204,113,0.3)'},
                                                        {'range': [14.3,50],
                                                         'color': 'rgba(243,156,18,0.3)'},
                                                        {'range': [50,100],
                                                         'color': 'rgba(231,76,60,0.3)'} 
                                                    ]  
                                                }
                                            )
        )
        fig_gauge.update_layout(height = 300)
        st.plotly_chart(fig_gauge, use_container_width = True)
