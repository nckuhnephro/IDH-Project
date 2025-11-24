# --- Core Libraries ---
import pandas as pd
import numpy as np
import copy
import joblib
import traceback

# --- Machine Learning & EBM ---
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, precision_recall_curve, auc
)

# --- Visualization ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

# --- 1. The Core EBM Wrapper Class (UNCHANGED) ---
class SmoothedEBM:
    # ... (The entire SmoothedEBM class is perfect and does not need any changes) ...
    def __init__(self, ebm_model):
        if not hasattr(ebm_model, 'feature_names_in_'): _ = ebm_model.explain_global()
        self.original_ebm = ebm_model
        self.intercept_ = ebm_model.intercept_[0]
        self.feature_names_in_ = ebm_model.feature_names_in_
        self.feature_types_in_ = ebm_model.feature_types_in_
        self.feature_bins_ = self._extract_all_bins(ebm_model)

    def _extract_all_bins(self, ebm_model):
        all_bins = {}
        global_explanation = ebm_model.explain_global()
        for i, name in enumerate(self.feature_names_in_):
            if self.feature_types_in_[i] == 'continuous':
                raw_data = global_explanation.data(i)
                if raw_data is None or 'scores' not in raw_data or raw_data['scores'] is None: continue
                scores = raw_data.get('scores')
                deviations = [np.nan] * len(scores)
                if 'error_bar' in raw_data: deviations = raw_data['error_bar']
                elif 'upper_bounds' in raw_data: deviations = np.abs(np.array(raw_data['upper_bounds']) - np.array(scores))
                fig_obj = global_explanation.visualize(i)
                if not fig_obj or not fig_obj.data: continue
                trace = fig_obj.data[0]
                lower_bounds = trace['x']
                if not (len(lower_bounds) == len(scores) == len(deviations)):
                    if len(lower_bounds) == len(scores) + 1:
                        upper_bounds = lower_bounds[1:]
                        lower_bounds = lower_bounds[:-1]
                    else: continue
                else:
                    upper_bounds = list(lower_bounds[1:]) + [raw_data['names'][-1]]
                all_bins[name] = pd.DataFrame({'lower_bound': lower_bounds, 'upper_bound': upper_bounds, 'score': scores,'score_std_dev': deviations})
        return all_bins
    
    def smooth_feature(self, feature_name, window_size=5, bin_indices=None):
        if feature_name not in self.feature_bins_: raise ValueError(f"Feature '{feature_name}' not found.")
        if window_size % 2 == 0: window_size += 1
        scores_to_modify = self.feature_bins_[feature_name]['score'].copy()
        if bin_indices is None:
            smoothed_scores = pd.Series(scores_to_modify).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
            self.feature_bins_[feature_name]['score'] = smoothed_scores
        else:
            globally_smoothed_series = pd.Series(scores_to_modify).rolling(window=window_size, center=True, min_periods=1).mean()
            scores_to_modify.iloc[bin_indices] = globally_smoothed_series.iloc[bin_indices]
            self.feature_bins_[feature_name]['score'] = scores_to_modify

    def apply_manual_adjustments(self, adjustments_dict):
        for feature_name, adjustments in adjustments_dict.items():
            if feature_name in self.feature_bins_:
                for bin_idx, new_score in adjustments.items():
                    self.feature_bins_[feature_name]['score'].iloc[bin_idx] = new_score

    def decision_function(self, X):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X, columns=self.feature_names_in_)
        total_scores = np.full(len(X), self.intercept_)
        for feature_name in self.feature_names_in_:
            if feature_name not in self.feature_bins_: continue
            bins_df = self.feature_bins_[feature_name]
            values_numeric = pd.to_numeric(X[feature_name], errors='coerce').fillna(X[feature_name].median())
            values = values_numeric.values
            lower_bounds = bins_df['lower_bound'].values
            bin_indices = np.clip(np.searchsorted(lower_bounds, values, side='right') - 1, 0, len(bins_df) - 1)
            total_scores += bins_df['score'].iloc[bin_indices].values
        return total_scores

    def predict_proba(self, X):
        log_odds = self.decision_function(X)
        probs = 1 / (1 + np.exp(-log_odds))
        return np.c_[1 - probs, probs]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

# --- 2. Backend Functions ---

# NEW Step 1 Function: Just stores the file paths
def handle_uploads(model_file, data_file):
    if not all([model_file, data_file]):
        return None, None, "Error: Please upload both files."
    # Return the temporary file paths created by Gradio
    return model_file.name, data_file.name, "Files uploaded. Ready to validate."

# MODIFIED Step 2 Function: Works with file paths from state
def load_and_validate_files(model_filepath, data_filepath, target_col):
    if not all([model_filepath, data_filepath, target_col]):
        return None, None, [], None, "Error: Files not uploaded or target not specified.", gr.update(interactive=False), gr.update(interactive=False)
    try:
        model = joblib.load(model_filepath)
        data_df = pd.read_csv(data_filepath, low_memory=False)
        
        # ... (The rest of the validation logic is the same and now robust) ...
        if not hasattr(model, 'feature_names_in_'):
            raise ValueError("Uploaded file is not a valid EBM model.")
        
        all_cols = data_df.columns.tolist()
        if target_col not in all_cols:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        initial_rows = len(data_df)
        if data_df[target_col].isnull().any():
            data_df.dropna(subset=[target_col], inplace=True)
        dropped_rows = initial_rows - len(data_df)
        
        model_features = set(model.feature_names_in_)
        data_features = set(all_cols)
        if not model_features.issubset(data_features):
            missing = model_features - data_features
            raise ValueError(f"Data is missing required features: {list(missing)}")
        
        features_to_use = model.feature_names_in_
        msg = f"Success! Found {len(features_to_use)} features. Dropped {dropped_rows} rows with missing targets."
        return model, data_df, features_to_use, target_col, msg, gr.update(choices=features_to_use, value=features_to_use[0], interactive=True), gr.update(interactive=True)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return None, None, [], None, error_trace, gr.update(interactive=False), gr.update(interactive=False)

# NEW Master update function (avoids .then() chaining issues)
# --- REPLACE the old function with this new, corrected version ---

def update_plot_and_metrics(model_obj, data_df, feature_cols, target_col, feature_name, 
                            window_size, bin_indices_str, view_options, manual_adjustments):
    """
    A single, robust function that generates the plot and all metrics at once.
    This avoids issues with chaining .then() events.
    """
    print("--- Running master update function ---")
    
    # --- THIS IS THE CORRECTED LOGIC ---
    # We now check for the DataFrame's validity using .empty
    if model_obj is None or data_df is None or data_df.empty or not feature_name:
        print("Master update function is exiting early: model, data, or feature name is missing.")
        empty_fig = go.Figure().update_layout(title="Waiting for data...")
        empty_df = pd.DataFrame()
        return empty_fig, empty_df, empty_fig, empty_df

    # --- The rest of the function is the same and will now work correctly ---
    
    # 1. First, generate the plot (logic from plot_and_compare_models)
    plot_fig = plot_and_compare_models(model_obj, feature_name, window_size, bin_indices_str, view_options, manual_adjustments)
    
    # 2. Second, generate all the metrics (logic from evaluate_models_on_data)
    full_metrics_df, cm_fig, light_metrics_df = evaluate_models_on_data(
        model_obj, data_df, feature_cols, target_col, feature_name, 
        window_size, bin_indices_str, manual_adjustments
    )
    
    # 3. Return all the results in the correct order for the UI
    return plot_fig, full_metrics_df, cm_fig, light_metrics_df

# (All other helper functions like plot_and_compare_models, evaluate_models_on_data, etc. are UNCHANGED)
def plot_and_compare_models(model_obj, feature_name, window_size, bin_indices_str, view_options, manual_adjustments):
    if not all([model_obj, feature_name]):
        return go.Figure(layout={'title': 'Upload a model and data to begin.'})
    wrapper = SmoothedEBM(model_obj)
    if feature_name not in wrapper.feature_bins_:
        return go.Figure(layout={'title': f"Feature '{feature_name}' not plottable."})
    
    smoothed_wrapper = copy.deepcopy(wrapper)
    try:
        bin_indices = [int(x.strip()) for x in bin_indices_str.split(',')] if bin_indices_str else None
        smoothed_wrapper.smooth_feature(feature_name, window_size=window_size, bin_indices=bin_indices)
    except Exception as e:
        print(f"Smoothing failed: {e}")
    
    adjusted_scores = smoothed_wrapper.feature_bins_[feature_name]['score'].copy()
    if feature_name in manual_adjustments:
        for idx, score in manual_adjustments[feature_name].items():
            adjusted_scores.iloc[idx] = score
    
    bins_df = wrapper.feature_bins_[feature_name]
    x_axis = (bins_df['lower_bound'] + bins_df['upper_bound']) / 2
    fig = go.Figure()
    if 'Show Confidence Interval' in view_options:
        upper = bins_df['score'] + 1.96 * bins_df['score_std_dev']
        lower = bins_df['score'] - 1.96 * bins_df['score_std_dev']
        fig.add_trace(go.Scatter(x=x_axis, y=upper, mode='lines', line={'width': 0}, showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=lower, mode='lines', line={'width': 0}, fill='tonexty', fillcolor='rgba(0,176,246,0.2)', name='95% CI'))
    if 'Show Original' in view_options:
        fig.add_trace(go.Scatter(x=x_axis, y=bins_df['score'], name='Original', mode='lines+markers', line={'color': 'rgb(0,176,246)'}))
    if 'Show Smoothed' in view_options:
        fig.add_trace(go.Scatter(x=x_axis, y=smoothed_wrapper.feature_bins_[feature_name]['score'], name='Smoothed', mode='lines', line={'color': 'rgb(255,127,14)', 'width': 3, 'dash': 'dash'}))
    if 'Show Manually Adjusted' in view_options:
        fig.add_trace(go.Scatter(x=x_axis, y=adjusted_scores, name='Adjusted', mode='lines+markers', line={'color': 'rgb(214, 39, 40)', 'width': 4}))
    fig.update_layout(title=f"Shape Function for '{feature_name}'", template="plotly_white")
    return fig

# --- REPLACE your old `evaluate_models_on_data` function with this corrected version ---

def evaluate_models_on_data(model_obj, data_df, feature_cols, target_col, feature_name, window_size, bin_indices_str, manual_adjustments):
    
    # --- THIS IS THE FIX ---
    # We now correctly check if the model or data are missing or if the DataFrame is empty.
    # This prevents the "ambiguous truth value" error.
    if model_obj is None or data_df is None or data_df.empty:
        return pd.DataFrame(), go.Figure(), pd.DataFrame()
    # --- END OF FIX ---

    X_eval, y_eval = data_df[feature_cols], data_df[target_col]
    
    smoothed = SmoothedEBM(model_obj)
    try:
        bin_indices = [int(x.strip()) for x in bin_indices_str.split(',')] if bin_indices_str else None
        smoothed.smooth_feature(feature_name, window_size=window_size, bin_indices=bin_indices)
    except Exception: pass
    
    adjusted = copy.deepcopy(smoothed)
    adjusted.apply_manual_adjustments(manual_adjustments)
    
    metrics = {
        "Original": calculate_all_metrics(model_obj, X_eval, y_eval),
        "Smoothed": calculate_all_metrics(smoothed, X_eval, y_eval),
        "Adjusted": calculate_all_metrics(adjusted, X_eval, y_eval)
    }
    
    cms = {k: v.pop("Confusion Matrix") for k, v in metrics.items()}
    metrics_df = pd.DataFrame(metrics).round(4).reset_index().rename(columns={'index': 'Metric'})
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(cms.keys()))
    colors = ['Blues', 'Greens', 'Reds']
    for i, (model_name, cm) in enumerate(cms.items()):
        # Correctly order for standard heatmap display (TN, FP, FN, TP)
        tp, fn = cm[0]
        fp, tn = cm[1]
        z = [[tn, fp], [fn, tp]]
        fig.add_trace(go.Heatmap(z=z, x=['Predicted Negative', 'Predicted Positive'], y=['Actual Negative', 'Actual Positive'], colorscale=colors[i], showscale=False, texttemplate="%{text}", text=z), 1, i+1)
    
    light_df = metrics_df[metrics_df['Metric'].isin(["ROC AUC", "PR Curve AUC", "F1 Score"])]
    return metrics_df, fig, light_df

def calculate_all_metrics(model, X, y):
    y_true = np.array(y).astype(float).astype(int)
    y_pred = np.array(model.predict(X)).astype(float).astype(int)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    return {"ROC AUC": roc_auc_score(y_true, y_proba), "Accuracy": accuracy_score(y_true, y_pred), "F1 Score": f1_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred), "Recall": recall_score(y_true, y_pred), "Specificity": tn/(tn+fp) if (tn+fp)>0 else 0, "PR Curve AUC": auc(rec, prec), "Confusion Matrix": [[tp, fn], [fp, tn]]}

def apply_manual_adjustment(adjustments, feature, index, score):
    if not all([feature, index]): return adjustments
    try: index = int(index)
    except (ValueError, TypeError): return adjustments
    if feature not in adjustments: adjustments[feature] = {}
    adjustments[feature][index] = score
    return adjustments

def reset_feature_adjustments(adjustments, feature):
    if feature in adjustments: del adjustments[feature]
    return adjustments

def export_model(model_obj, feature, window, bins, adjustments):
    if not model_obj: return None
    wrapper = SmoothedEBM(model_obj)
    try:
        bin_indices = [int(x.strip()) for x in bins.split(',')] if bins else None
        wrapper.smooth_feature(feature, window_size=window, bin_indices=bin_indices)
    except Exception as e:
        print(f"Export smoothing failed: {e}")
    wrapper.apply_manual_adjustments(adjustments)
    filepath = "adjusted_ebm_model.joblib"
    joblib.dump(wrapper, filepath)
    return filepath
# --- 3. The Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Universal EBM Workbench")
    
    # States for final, validated objects
    model_state = gr.State()
    data_state = gr.State()
    feature_cols_state = gr.State()
    target_col_state = gr.State()
    manual_adjustments_state = gr.State({})
    
    # NEW States for temporary file paths
    model_filepath_state = gr.State()
    data_filepath_state = gr.State()

    with gr.Accordion("Step 1: Upload and Validate Data", open=True):
        with gr.Row():
            model_upload = gr.File(label="Upload EBM Model (.joblib)", file_types=['.joblib'])
            data_upload = gr.File(label="Upload Dataset (.csv)", file_types=['.csv'])
        
        # NEW Two-button layout
        with gr.Row():
            upload_button = gr.Button("1. Stage Uploaded Files")
            target_col_textbox = gr.Textbox(label="2. Enter Target Column Name (Case-Sensitive)")
        
        load_button = gr.Button("3. Load & Validate Data", variant="primary")
        status_textbox = gr.Textbox(label="Status", interactive=False, lines=3)
        
    with gr.Tabs() as main_tabs:
        with gr.TabItem("EBM Analysis Workbench", interactive=False) as workbench_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    # (The rest of the UI layout is the same)
                    feature_dropdown = gr.Dropdown(label="Feature to Analyze", interactive=False)
                    window_slider = gr.Slider(minimum=3, maximum=51, step=2, value=5, label="Smoothing Window")
                    bins_textbox = gr.Textbox(label="Apply to Bins (e.g., 1,2,3)")
                    view_checkboxes = gr.CheckboxGroup(choices=["Show Original", "Show Smoothed", "Show Manually Adjusted", "Show Confidence Interval"], value=["Show Original", "Show Smoothed", "Show Manually Adjusted"], label="Display Elements")
                    update_button = gr.Button("Update Plot & Metrics", variant="primary")
                    with gr.Group():
                        gr.Markdown("#### Manual Adjustment")
                        selected_bin_text = gr.Textbox(label="Bin Index")
                        new_score_input = gr.Number(label="Set New Score")
                        apply_adj_btn = gr.Button("Apply")
                        reset_adj_btn = gr.Button("Reset Feature")
                    with gr.Group():
                        gr.Markdown("### Export Model")
                        export_btn = gr.Button("Export Adjusted Model")
                        download_link = gr.File(label="Download Model", interactive=False)
                
                with gr.Column(scale=3):
                    interactive_plot = gr.Plot()
                    light_metrics_df = gr.DataFrame(label="Key Metrics Summary")
                    with gr.Accordion("Full Performance Evaluation", open=False):
                        metrics_df = gr.DataFrame(label="All Metrics")
                        metrics_plot = gr.Plot(label="Confusion Matrices")

    # --- Event Wiring ---
    
    # NEW Step 1: Staging button
    upload_button.click(
        fn=handle_uploads,
        inputs=[model_upload, data_upload],
        outputs=[model_filepath_state, data_filepath_state, status_textbox]
    )

    # NEW Step 2: Validation button
    load_button.click(
        fn=load_and_validate_files,
        inputs=[model_filepath_state, data_filepath_state, target_col_textbox],
        outputs=[model_state, data_state, feature_cols_state, target_col_state, status_textbox, feature_dropdown, workbench_tab]
    )
    
    # The rest of the wiring uses the new master function
    master_inputs = [model_state, data_state, feature_cols_state, target_col_state, feature_dropdown, window_slider, bins_textbox, view_checkboxes, manual_adjustments_state]
    master_outputs = [interactive_plot, metrics_df, metrics_plot, light_metrics_df]

    update_button.click(fn=update_plot_and_metrics, inputs=master_inputs, outputs=master_outputs)
    apply_adj_btn.click(fn=apply_manual_adjustment, inputs=[manual_adjustments_state, feature_dropdown, selected_bin_text, new_score_input], outputs=[manual_adjustments_state]).then(fn=update_plot_and_metrics, inputs=master_inputs, outputs=master_outputs)
    reset_adj_btn.click(fn=reset_feature_adjustments, inputs=[manual_adjustments_state, feature_dropdown], outputs=[manual_adjustments_state]).then(fn=update_plot_and_metrics, inputs=master_inputs, outputs=master_outputs)
    
    export_inputs = [model_state, feature_dropdown, window_slider, bins_textbox, manual_adjustments_state]
    export_btn.click(fn=export_model, inputs=export_inputs, outputs=[download_link])

demo.launch()