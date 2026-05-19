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
    
    #SHAP values for this single instance
    shap_vals_list = explainer.shap_values(x_append_row.reshape(1, -1))
    #for binary classification, shap_vals_list is [class0, class1]; i need class1 (malicious)
    if isinstance(shap_vals_list, list) and len(shap_vals_list) == 2:
        shap_values = shap_vals_list[1][0]       # first (only) row of positive class
    else:
        shap_values = shap_vals_list[0]          # fallback
    
    #probability and prediction
    prob = xgb_model.predict_proba(x_append_row.reshape(1, -1))[0, 1]
    pred = "Malicious" if prob >= threshold else "Normal"
    confidence = prob if pred == "Malicious" else 1 - prob
    
    #feature contributions
    feature_contrib = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values
    }).sort_values('shap_value', ascending=False)
    
    #top features pushing toward malicious (positive SHAP)
    top_malicious = feature_contrib[feature_contrib['shap_value'] > 0].head(5)
    #top features pushing toward normal (negative SHAP)
    top_normal = feature_contrib[feature_contrib['shap_value'] < 0].head(5).sort_values('shap_value')
    
    #get original values for those features (human-readable)
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
        <h1 class="hero-title">ThreatDetect</h1>
        <p class="hero-subtitle">
            Snuff out any potential insider threats before they can cause harm 
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown(f"### Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Organisational Search via CSV", 
         "Exploratory Data Analysis"]
    )
    st.divider()

    if page == "Organisational Search via CSV":
        st.markdown('<h2 class="centered-header">Organisational Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("**Upload your CSV file** containing employee data, logs, or threat indicators")
        with col2:
            st.metric("Files Analyzed", "0", help="Number of CSV files processed")

        file_upload = st.file_uploader("Select Your CSV File", type="csv")
        
        if file_upload is not None:
            try:
                df = pd.read_csv(file_upload)
                st.success(f"Loaded **{len(df):,}** records successfully!")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    if 'employee_campus' in df.columns:
                        st.metric("Unique Campuses", df["employee_campus"].nunique())
                
                if st.button("Run Threat Detection", type="primary"):
                    with st.spinner("Analyzing all records..."):
                        model = load_model()
                        try:
                            #prepare features and get predictions
                            processed_df, x_append, iso_scores = prepare_features(df, model)
                            xgb_model = model['xgb_model']
                            threshold = model['best_threshold']
                            probs = xgb_model.predict_proba(x_append)[:, 1]
                            preds = (probs >= threshold).astype(int)
                            
                            #create results dataframe
                            results_df = df.copy()
                            results_df["Prediction"] = ["Malicious" if p == 1 else "Normal" for p in preds]
                            results_df["Risk_Prob"] = probs
                            results_df["Anomaly_Score"] = iso_scores
                            results_df["Confidence"] = np.where(preds == 1, probs, 1 - probs).astype(float)
                            
                            #keeps analysis results in session state to avoid recalculation
                            st.session_state.analysis_complete = True
                            st.session_state.model = model
                            st.session_state.x_append = x_append
                            st.session_state.iso_scores = iso_scores
                            st.session_state.results_df = results_df
                            st.session_state.threshold = threshold
                            st.session_state.df_original = df
                            st.session_state.probs = probs
                            st.session_state.preds = preds
                            st.success("Analysis complete! Results are displayed below.")
                            
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                
                #show analysis results if they exist in session state (it kept on restarting)
                if "analysis_complete" in st.session_state and st.session_state.analysis_complete:
                    model = st.session_state.model
                    x_append = st.session_state.x_append
                    iso_scores = st.session_state.iso_scores
                    results_df = st.session_state.results_df
                    threshold = st.session_state.threshold
                    df = st.session_state.df_original
                    probs = st.session_state.probs
                    preds = st.session_state.preds
                    xgb_model = model['xgb_model']
                    feature_names_full = model['feature_columns']
                    
                    #statics of orgnaization
                    st.subheader("Organisational Threat Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    total = len(results_df)
                    mal_count = (results_df["Prediction"] == "Malicious").sum()
                    norm_count = total - mal_count
                    col1.metric("Total Employees", total)
                    col2.metric("Malicious", mal_count, delta=f"{mal_count/total:.1%}" if total>0 else "0")
                    col3.metric("Normal", norm_count, delta=f"{norm_count/total:.1%}" if total>0 else "0")
                    col4.metric("Avg. Confidence", f"{results_df['Confidence'].mean():.2%}")
                    
                    #graphing implementation
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    #bar chart of predictions
                    pred_counts = results_df["Prediction"].value_counts()
                    axes[0].bar(pred_counts.index, pred_counts.values, color=['#d9534f', '#5bc0de'])
                    axes[0].set_title("Threat Prediction Count")
                    axes[0].set_ylabel("Number of employees")
                    
                    #histogram of risk probabilities
                    axes[1].hist(results_df["Risk_Prob"], bins=30, color='darkorange', edgecolor='black')
                    axes[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
                    axes[1].set_title("Risk Probability Distribution")
                    axes[1].set_xlabel("Probability of being malicious")
                    axes[1].set_ylabel("Frequency")
                    axes[1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader("📈 Global Feature Importance (Top 15)")
                    importance_feature_names = model['feature_columns'][:-1]
                    importance = xgb_model.feature_importances_[:len(importance_feature_names)]
                    imp_df = pd.DataFrame({'feature': importance_feature_names, 'importance': importance}).sort_values('importance', ascending=False).head(15)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(imp_df['feature'], imp_df['importance'], color='teal')
                    ax2.set_xlabel("Importance (F-score)")
                    ax2.set_title("Which features drive malicious predictions across the organisation?")
                    ax2.invert_yaxis()
                    st.pyplot(fig2)
                    plt.close(fig2)

                    st.subheader("Global SHAP Explanation (sample of 100 records)")

                    full_feature_names = model['feature_columns']

                    #sample the data to keep SHAP computation fast
                    if len(x_append) > 100:
                        sample_idx = np.random.choice(len(x_append), 100, replace=False)
                        x_sample = x_append[sample_idx]
                    else:
                        x_sample = x_append

                    explainer = model['shap_explainer']
                    shap_values_sample = explainer.shap_values(x_sample)

                    #work and handle binary classification result
                    if isinstance(shap_values_sample, list):
                        if len(shap_values_sample) == 2:
                            shap_vals = shap_values_sample[1]
                        else:
                            shap_vals = shap_values_sample[0]
                    else:
                        shap_vals = shap_values_sample

                    #plot with the full feature names
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_vals, x_sample, feature_names=full_feature_names,
                                    show=False, max_display=15)
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
                    #EXPLAINABILITY
                    st.subheader("Organisational Risk Insight")
                    high_risk_threshold = min(0.95, threshold + 0.15)
                    moderate_mask = (probs >= threshold) & (probs < high_risk_threshold)
                    high_risk_count = int((probs >= high_risk_threshold).sum())
                    moderate_risk_count = int(moderate_mask.sum())
                    low_risk_count = int((probs < threshold).sum())
                    avg_anomaly_score = results_df["Anomaly_Score"].mean()
                    most_anomalous = results_df.nsmallest(3, "Anomaly_Score")["Anomaly_Score"].tolist()
                    avg_malicious_confidence = results_df.loc[results_df["Prediction"] == "Malicious", "Confidence"].mean() if mal_count > 0 else 0.0
                    top_features = ', '.join(imp_df.head(3)['feature'].values)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Malicious", f"{mal_count}", delta=f"{mal_count/total:.1%}" if total > 0 else "0")
                    col2.metric("High risk", f"{high_risk_count}", delta=f"{high_risk_count/total:.1%}" if total > 0 else "0")
                    col3.metric("Moderate risk", f"{moderate_risk_count}", delta=f"{moderate_risk_count/total:.1%}" if total > 0 else "0")
                    col4.metric("Avg. anomaly", f"{avg_anomaly_score:.3f}")

                    st.markdown(
                        f"**Threshold reference:** {threshold:.2f}  \n"
                        f"**High-risk boundary:** {high_risk_threshold:.2f}  \n"
                        f"**Average risk probability:** {results_df['Risk_Prob'].mean():.2%}"
                    )

                    st.markdown(
                        "#### Key organisational risk signals\n"
                        f"- **Top features driving malicious predictions:** {top_features}  \n"
                        f"- **High-risk employees:** {high_risk_count}  \n"
                        f"- **Moderate-risk employees:** {moderate_risk_count}  \n"
                        f"- **Low-risk employees:** {low_risk_count}  \n"
                        f"- **Average anomaly score:** {avg_anomaly_score:.3f}  \n"
                        f"- **Most anomalous scores:** {', '.join([f'{score:.3f}' for score in most_anomalous])}"
                    )

                    st.markdown(
                        "#### What does this means\n"
                        "- A **Malicious** result means the model sees a strong pattern of risk in this group's data.\n"
                        "- A **High risk** employee has a very strong signal and should be reviewed first.\n"
                        "- A **Moderate risk** employee may be worth investigating, especially if multiple warning signs appear.\n"
                        "- A **Low risk** employee appears more typical compared with the rest of the dataset.\n"
                        "- The anomaly score shows how unusual the behavior is compared to others; lower numbers are more unusual.\n"
                        "#### What to do next\n"
                        "- Start with the high-risk employees and use the per-record explanation section to see the exact factors affecting the decision.\n"
                        "- Pay attention to the top features listed above, such as after-hours activity or unusual access patterns.\n"
                        "- If you are unsure, export the results and share them with your security or compliance team for a deeper review.\n"
                        "- If the dataset is incomplete, add more employee or activity details and rerun the analysis to improve confidence."
                    )

                    if mal_count > 0:
                        st.warning(
                            f"**{mal_count} employees ({mal_count/total:.1%})** exhibit malicious patterns. "
                            f"The highest-risk features across the organisation are: {top_features}.\n\n"
                            f"Average confidence for flagged employees is {avg_malicious_confidence:.1%}."
                        )
                    else:
                        st.success("No malicious employees detected. The organisation appears clean.")

                    #Results table and download
                    with st.expander("Detailed Results Table (all employees)"):
                        display_cols = ["Prediction", "Confidence", "Risk_Prob", "Anomaly_Score"] + \
                                      [c for c in st.session_state.df_original.columns if c in model['feature_columns']][:5]
                        st.dataframe(results_df[display_cols], use_container_width=True)
                        
                        #provide option to download results as CSV
                        csv_download = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download results as CSV",
                            data=csv_download,
                            file_name="threat_analysis_results.csv",
                            mime="text/csv"
                        )
                    
                    #per record explainability
                    with st.expander("Explain a specific employee (SHAP per instance)"):
                        record_options = [
                            f"Employee {i} – {row['Prediction']} (Conf: {row['Confidence']:.2%})"
                            for i, row in results_df.iterrows()
                        ]
                        st.write(f"**Enter an employee number (0 to {len(record_options) - 1})**")
                        selected_idx = st.number_input(
                            "Employee number",
                            min_value=0,
                            max_value=len(record_options) - 1,
                            step=1,
                            value=0,
                            label_visibility="collapsed"
                        )
                        selected_idx = int(selected_idx)
                        st.info(record_options[selected_idx])
                        
                        original_row = df.iloc[selected_idx].to_dict()
                        x_row = x_append[selected_idx]
                        iso = iso_scores[selected_idx][0]
                        
                        pred_text, prob_score, conf, explanation_list, feat_contrib = results_explainability(
                            model, x_row, iso, original_row, feature_names_full, threshold
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prediction", pred_text)
                        col2.metric("Confidence", f"{conf:.2%}")
                        col3.metric("Anomaly Score", f"{iso:.3f}")
                        
                        st.markdown("**Why? (Human‑readable risk indicators)**")
                        for bullet in explanation_list[:8]:
                            st.write(bullet)
                        
                        #SHAP bar for this employee
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
                st.error(f"Error reading file: {str(e)}")        
    elif page == "Exploratory Data Analysis":
        st.markdown('<h2 class="centered-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        st.info("Interactive visualizations & quick anomaly insights")

        #let user choose dataset source
        data_source = st.radio("Choose dataset:", ("Upload CSV", "Use sample dataset"))

        df = None
        if data_source == "Upload CSV":
            upload = st.file_uploader("Upload a CSV for EDA", type="csv")
            if upload is not None:
                try:
                    df = pd.read_csv(upload)
                    st.success(f"Loaded **{len(df):,}** records from uploaded file")
                except Exception as e:
                    st.error(f"Could not read uploaded CSV: {e}")
        else:
            #try to load the bundled sample dataset
            try:
                @st.cache_data
                def _load_sample():
                    return pd.read_csv("AI_Model_Code/insider_threat_clean_dataset.csv")
                df = _load_sample()
                st.success(f"Loaded sample dataset with **{len(df):,}** records")
            except Exception:
                st.warning("Sample dataset not available. Please upload a CSV.")

        if df is None:
            st.info("Upload a file or select the sample dataset to begin EDA.")
            return

        #show basic dataframe and shape
        st.subheader("Dataset Snapshot")
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        with st.expander("Preview data (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)


        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        #quick summary
        st.subheader("Quick Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.describe(include='all').T)
        with col2:
            missing = df.isnull().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            if len(missing):
                st.markdown("**Missing values (top)**")
                st.bar_chart(missing.head(20))
            else:
                st.markdown("**Missing values:** None detected")

        #detect candidate target column 
        possible_targets = [c for c in df.columns if c.lower() in ("is_malicious","malicious","label","target","is_threat","threat")]
        target_col = possible_targets[0] if possible_targets else None
        if target_col:
            st.markdown(f"**Detected target column:** `{target_col}`")
            st.write(df[target_col].value_counts(dropna=False))

        #istributions for numeric features
        if numeric_cols:
            st.subheader("Numeric Feature Exploration")
            num_sel = st.selectbox("Choose numeric column to inspect", [None] + numeric_cols)
            if num_sel:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(df[num_sel].dropna(), kde=True, ax=ax[0], color='cornflowerblue') # pyright: ignore[reportArgumentType]
                ax[0].set_title(f"Distribution of {num_sel}")
                sns.boxplot(x=df[num_sel], ax=ax[1], color='lightgreen')
                ax[1].set_title(f"Boxplot of {num_sel}")
                st.pyplot(fig)
                plt.close(fig)

            #scatter between two numeric features
            if len(numeric_cols) >= 2:
                st.subheader("Scatter / Relationship Explorer")
                x_col = st.selectbox("X axis", numeric_cols, index=0)
                y_col = st.selectbox("Y axis", numeric_cols, index=1)
                hue = st.selectbox("Color by (optional)", [None] + ([target_col] if target_col else []) + cat_cols)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                if hue and hue in df.columns:
                    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax2, palette='tab10', alpha=0.7)
                else:
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax2, color='purple', alpha=0.6)
                ax2.set_title(f"{y_col} vs {x_col}")
                st.pyplot(fig2)
                plt.close(fig2)

            #correlation heatmap
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Matrix (numeric)")
                corr = df[numeric_cols].corr()
                fig3, ax3 = plt.subplots(figsize=(10, max(4, len(numeric_cols)*0.25)))
                sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax3)
                st.pyplot(fig3)
                plt.close(fig3)

                if target_col and target_col in corr.columns:
                    st.markdown("**Top features correlated with target**")
                    corr_with_target = corr[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False)
                    st.write(corr_with_target.head(10))

        #categorical exploration
        if cat_cols:
            st.subheader("Categorical Feature Counts")
            cat_sel = st.selectbox("Choose categorical column (counts)", [None] + cat_cols)
            if cat_sel:
                counts = df[cat_sel].value_counts(dropna=False).head(30)
                fig4, ax4 = plt.subplots(figsize=(8, min(6, 0.35*len(counts))))
                sns.barplot(x=counts.values, y=counts.index, ax=ax4, palette='mako')
                ax4.set_xlabel("Count")
                ax4.set_title(f"Value counts for {cat_sel}")
                st.pyplot(fig4)
                plt.close(fig4)

        #quick anomaly hint using IsolationForest if num cols available
        if numeric_cols:
            st.subheader("Quick Anomaly Scan (IsolationForest)")
            if st.button("Run quick anomaly scan"):
                try:
                    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                    sample_df = df[numeric_cols].fillna(0).sample(n=min(2000, len(df)), random_state=42)
                    iso.fit(sample_df)
                    scores = iso.decision_function(sample_df)
                    outliers = (scores < np.quantile(scores, 0.01)).sum()
                    st.write(f"Approx. {outliers} outliers detected in sampled data (top 1%).")
                    fig5, ax5 = plt.subplots(figsize=(8, 3))
                    ax5.hist(scores, bins=50, color='salmon')
                    ax5.set_title("IsolationForest anomaly score distribution (sample)")
                    st.pyplot(fig5)
                    plt.close(fig5)
                except Exception as e:
                    st.error(f"Anomaly scan failed: {e}")

        st.success("Exploratory Data Analysis complete. Use the controls above to refine views.")

if __name__ == "__main__":
    main()