import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, precision_recall_curve, PrecisionRecallDisplay)
from sklearn.ensemble import IsolationForest
import shap

import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

@st.cache_resource #this is to make sure that the doesnt reload every time the page is refreshed
def get_styling(style_path):
    with open(style_path) as style_file:
        st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)

get_styling("assets/app_styling.css")


@st.cache_data
def load_model():
    with open("AI_Model_Code/insider_threat_model.pkl", "rb") as file:
        model_package = pickle.load(file)
    if 'shap_explainer' not in model_package:
        model_package['shap_explainer'] = shap.TreeExplainer(model_package['xgb_model'])
    return model_package


def validate_input_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            "Please provide a CSV with the expected employee / threat feature columns."
        )


def prepare_features(df, model_package):
    cat_cols = model_package['cat_cols']
    bin_cols = model_package['bin_cols']
    num_cols = model_package['num_cols']

    engineered_cols = ['print_ratio', 'file_ratio', 'risk_ratio', 'access_ratio', 'afterhrs_ratio']
    required_raw = [
        col for col in model_package['feature_columns']
        if col not in engineered_cols + ['isolation_forest_anomaly_score']
    ]

    validate_input_columns(df, required_raw)

    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str).apply(lambda s: s.str.strip())
    raw_num_cols = [col for col in num_cols if col in df.columns]
    df[raw_num_cols] = df[raw_num_cols].apply(pd.to_numeric, errors='coerce')
    df[raw_num_cols] = df[raw_num_cols].fillna(df[raw_num_cols].median())

    df['print_ratio'] = df['total_printed_pages'] / df['num_printed_pages_off_hours']
    df['file_ratio'] = df['total_files_burned'] / df['num_printed_pages_off_hours']
    df['risk_ratio'] = (
        df['has_criminal_record'] + df['is_contractor'] + df['has_foreign_citizenship']
    )
    df['access_ratio'] = df['num_printed_pages_off_hours'] * df['entry_during_weekend']
    df['afterhrs_ratio'] = df['late_exit_flag'] * df['num_printed_pages_off_hours']

    df[engineered_cols] = df[engineered_cols].replace([np.inf, -np.inf], np.nan)
    df[engineered_cols] = df[engineered_cols].fillna(df[engineered_cols].median())

    for col in cat_cols:
        le = model_package['label_encoders'][col]
        values = df[col].astype(str).str.strip()
        unseen = ~values.isin(le.classes_)
        if unseen.any():
            raise ValueError(
                f"Column '{col}' contains unseen categories: "
                f"{sorted(values[unseen].unique())}."
            )
        df[col] = le.transform(values)

    df[num_cols] = model_package['scaler'].transform(df[num_cols])

    feature_cols = model_package['feature_columns'][:-1]
    x_for_iso = df[feature_cols].to_numpy()
    iso_scores = model_package['iso_forest'].decision_function(x_for_iso).reshape(-1, 1)
    x_append = np.hstack((x_for_iso, iso_scores))

    return df, x_append, iso_scores

def results_explainability(model_package, x_append_row, iso_score, original_row, feature_names, threshold):
    xgb_model = model_package['xgb_model']
    explainer = model_package['shap_explainer']
    
    # SHAP values for this single instance
    shap_vals_list = explainer.shap_values(x_append_row.reshape(1, -1))
    # For binary classification, shap_vals_list is [class0, class1]; we want class1 (malicious)
    if isinstance(shap_vals_list, list) and len(shap_vals_list) == 2:
        shap_values = shap_vals_list[1][0]       # first (only) row of positive class
    else:
        shap_values = shap_vals_list[0]          # fallback
    
    # Probability and prediction
    prob = xgb_model.predict_proba(x_append_row.reshape(1, -1))[0, 1]
    pred = "Malicious" if prob >= threshold else "Normal"
    confidence = prob if pred == "Malicious" else 1 - prob
    
    # Feature contributions
    feature_contrib = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values
    }).sort_values('shap_value', ascending=False)
    
    # Top features pushing toward malicious (positive SHAP)
    top_malicious = feature_contrib[feature_contrib['shap_value'] > 0].head(5)
    # Top features pushing toward normal (negative SHAP)
    top_normal = feature_contrib[feature_contrib['shap_value'] < 0].head(5).sort_values('shap_value')
    
    # Get original values for those features (human-readable)
    readable_explanation = []
    for _, row in top_malicious.iterrows():
        feat = row['feature']
        val = original_row.get(feat, 'N/A')
        if isinstance(val, (np.generic, np.ndarray)):
            val = val.item()
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        readable_explanation.append(f"• **{feat}** = {val}  →  increases risk")
    
    for _, row in top_normal.iterrows():
        feat = row['feature']
        val = original_row.get(feat, 'N/A')
        if isinstance(val, (np.generic, np.ndarray)):
            val = val.item()
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        readable_explanation.append(f"• **{feat}** = {val}  →  reduces risk")
    
    return pred, prob, confidence, readable_explanation, feature_contrib


def main():

    st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">🔍 ThreatFind</h1>
        <p class="hero-subtitle">
            Snuff out any potential insider threats before they can cause harm 🚨
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown(f"### 🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Organisational Search via CSV", 
         "🎯 Single Search", 
         "📊 Exploratory Data Analysis"]
    )
    st.divider()

    if page == "🔍 Organisational Search via CSV":
        st.markdown('<h2 class="centered-header">🏢 Organisational Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("👆 **Upload your CSV file** containing employee data, logs, or threat indicators")
        with col2:
            st.metric("Files Analyzed", "0", help="Number of CSV files processed")

        file_upload = st.file_uploader("📁 Select Your CSV File", type="csv")
        
        if file_upload is not None:
            try:
                df = pd.read_csv(file_upload)
                st.success(f"Loaded **{len(df):,}** records successfully!")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Total Records", len(df))
                with col2:
                    if 'employee_campus' in df.columns:
                        st.metric("Unique Campuses", df["employee_campus"].nunique())
                
                if st.button("🚀 Run Threat Detection", type="primary"):
                    with st.spinner("Analyzing all records..."):
                        model = load_model()
                        try:
                            # Prepare features and get predictions
                            processed_df, x_append, iso_scores = prepare_features(df, model)
                            xgb_model = model['xgb_model']
                            threshold = model['best_threshold']
                            probs = xgb_model.predict_proba(x_append)[:, 1]
                            preds = (probs >= threshold).astype(int)
                            
                            # Build results dataframe
                            results_df = df.copy()
                            results_df["Prediction"] = ["Malicious" if p == 1 else "Normal" for p in preds]
                            results_df["Risk_Prob"] = probs
                            results_df["Anomaly_Score"] = iso_scores
                            results_df["Confidence"] = np.where(preds == 1, probs, 1 - probs).astype(float)
                            
                            # ========== OVERALL STATISTICS ==========
                            st.subheader("📊 Organisational Threat Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            total = len(results_df)
                            mal_count = (results_df["Prediction"] == "Malicious").sum()
                            norm_count = total - mal_count
                            col1.metric("Total Employees", total)
                            col2.metric("⚠️ Malicious", mal_count, delta=f"{mal_count/total:.1%}" if total>0 else "0")
                            col3.metric("✅ Normal", norm_count, delta=f"{norm_count/total:.1%}" if total>0 else "0")
                            col4.metric("Avg. Confidence", f"{results_df['Confidence'].mean():.2%}")
                            
                            # ========== GRAPHS ==========
                            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Bar chart of predictions
                            pred_counts = results_df["Prediction"].value_counts()
                            axes[0].bar(pred_counts.index, pred_counts.values, color=['#d9534f', '#5bc0de'])
                            axes[0].set_title("Threat Prediction Count")
                            axes[0].set_ylabel("Number of employees")
                            
                            # Histogram of risk probabilities
                            axes[1].hist(results_df["Risk_Prob"], bins=30, color='darkorange', edgecolor='black')
                            axes[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
                            axes[1].set_title("Risk Probability Distribution")
                            axes[1].set_xlabel("Probability of being malicious")
                            axes[1].set_ylabel("Frequency")
                            axes[1].legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)# After loading model and before using feature_names
                            
                            feature_names_full = model['feature_columns']
                            if len(feature_names_full) == 0:
                                st.error("Model has no feature columns defined.")
                                st.stop()
                            # Exclude the isolation forest score column (last column)
                            feature_names = feature_names_full[:-1] if len(feature_names_full) > 1 else feature_names_full
                                                        

                            st.subheader("📈 Global Feature Importance (Top 15)")
                            feature_names = model['feature_columns'][:-1]  # exclude isolation forest score
                            importance = xgb_model.feature_importances_[:len(feature_names)]
                            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False).head(15)
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            ax2.barh(imp_df['feature'], imp_df['importance'], color='teal')
                            ax2.set_xlabel("Importance (F-score)")
                            ax2.set_title("Which features drive malicious predictions across the organisation?")
                            ax2.invert_yaxis()
                            st.pyplot(fig2)
                            plt.close(fig2)
                            
                            # Optional: SHAP summary for the whole dataset (use a sample if too large)
      # ... after global feature importance plot ...

                            # SHAP summary for the whole dataset (sample up to 100 records)
                            st.subheader("🔎 Global SHAP Explanation (sample of 100 records)")

                            # Use the full feature list (including isolation_forest_anomaly_score)
                            full_feature_names = model['feature_columns']   # already contains the iso score column

                            # Sample the data to keep SHAP computation fast
                            if len(x_append) > 100:
                                sample_idx = np.random.choice(len(x_append), 100, replace=False)
                                x_sample = x_append[sample_idx]
                            else:
                                x_sample = x_append

                            explainer = model['shap_explainer']
                            shap_values_sample = explainer.shap_values(x_sample)

                            # Handle binary classification output (list of two arrays)
                            if isinstance(shap_values_sample, list):
                                if len(shap_values_sample) == 2:
                                    shap_vals = shap_values_sample[1]   # positive class (malicious)
                                else:
                                    shap_vals = shap_values_sample[0]   # fallback
                            else:
                                shap_vals = shap_values_sample

                            # Now shap_vals has shape (n_samples, n_features)
                            # Plot with the full feature names
                            fig3, ax3 = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_vals, x_sample, feature_names=full_feature_names,
                                            show=False, max_display=15)
                            st.pyplot(fig3)
                            plt.close(fig3)
                            
                            # ========== OVERALL EXPLANATION TEXT ==========
                            st.subheader("📝 Organisational Risk Insight")
                            if mal_count > 0:
                                st.warning(f"**{mal_count} employees ({mal_count/total:.1%})** exhibit malicious patterns. "
                                          f"The highest‑risk features across the organisation are: "
                                          f"{', '.join(imp_df.head(3)['feature'].values)}.")
                            else:
                                st.success("✅ No malicious employees detected. The organisation appears clean.")
                            
                            # ========== DETAILED RESULTS TABLE ==========
                            with st.expander("📋 Detailed Results Table (all employees)"):
                                display_cols = ["Prediction", "Confidence", "Risk_Prob", "Anomaly_Score"] + \
                                              [c for c in df.columns if c in model['feature_columns']][:5]
                                st.dataframe(results_df[display_cols], use_container_width=True)
                                
                                # Download button
                                csv_download = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="⬇️ Download results as CSV",
                                    data=csv_download,
                                    file_name="threat_analysis_results.csv",
                                    mime="text/csv"
                                )
                            
                            # ========== PER‑RECORD EXPLANATION (optional) ==========
                            with st.expander("🔍 Explain a specific employee (SHAP per instance)"):
                                record_options = [
                                    f"Employee {i} – {row['Prediction']} (Conf: {row['Confidence']:.2%})"
                                    for i, row in results_df.iterrows()
                                ]
                                selected_idx = st.selectbox("Select a record to explain", range(len(record_options)), 
                                                           format_func=lambda i: record_options[i])
                                
                                original_row = df.iloc[selected_idx].to_dict()
                                x_row = x_append[selected_idx]
                                iso = iso_scores[selected_idx]
                                
                                pred_text, prob_score, conf, explanation_list, feat_contrib = results_explainability(
                                    model, x_row, iso, original_row, feature_names, threshold
                                )
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Prediction", pred_text)
                                col2.metric("Confidence", f"{conf:.2%}")
                                col3.metric("Anomaly Score", f"{iso:.3f}")
                                
                                st.markdown("**Why? (Human‑readable risk indicators)**")
                                for bullet in explanation_list[:8]:
                                    st.write(bullet)
                                
                                # SHAP bar for this employee
                                fig4, ax4 = plt.subplots(figsize=(8, 4))
                                top_n = feat_contrib.head(10)
                                colors = ['red' if x > 0 else 'green' for x in top_n['shap_value']]
                                ax4.barh(top_n['feature'], top_n['shap_value'], color=colors)
                                ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
                                ax4.set_xlabel("SHAP value (pushes toward Malicious →)")
                                ax4.set_title("Top 10 features influencing this employee")
                                st.pyplot(fig4)
                                plt.close(fig4)
                            
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    elif page == "Single Record Search":
        st.markdown('<h2 class="centered-header">Single Entity Search</h2>', unsafe_allow_html=True)
        st.info("This is the Place to Search for a Single Employee or Entity.")
        
    elif page == "📊 Exploratory Data Analysis":
        st.markdown('<h2 class="centered-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        st.info("🔬 **Coming soon**: Interactive visualizations & anomaly detection")

if __name__ == "__main__":
    main()