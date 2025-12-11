import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path=None):
    """Load UCS data: Hardcoded fallback or CSV."""
    if csv_path:
        df = pd.read_csv(csv_path)
        df['compaction_num'] = (df['compaction'] == 'WAS').astype(int)
        df['cement_glass'] = df['cement'] * df['glass']
        df['cement_sq'] = df['cement'] ** 2
        df['glass_sq'] = df['glass'] ** 2
        print(f"Loaded {len(df)} rows from CSV.")
    else:
        # Hardcoded UCS data
        bslUCS = {
            7: {0: [76.73, 62.67, 70, 204], 2.5: [125, 404, 170, 502], 5: [140, 539, 420, 499], 7.5: [387, 520, 687.33, 534]},
            14: {0: [141.33, 70, 134.67, 228], 2.5: [298, 446.67, 539.5, 515], 5: [416.67, 577.33, 696, 715], 7.5: [480, 693.33, 720, 603.33]},
            28: {0: [184, 161.33, 686, 305], 2.5: [480, 530, 576.67, 548], 5: [540, 589.33, 753, 724.67], 7.5: [560, 582, 886, 870]}
        }
        wasUCS = {
            7: {0: [118, 121, 123, 116], 2.5: [232, 161.33, 270, 502], 5: [378, 294, 640, 633], 7.5: [490, 622, 840, 694]},
            14: {0: [141.33, 163, 201, 242], 2.5: [372, 550, 603, 570], 5: [498, 613.33, 828, 735], 7.5: [651, 793.33, 879, 792]},
            28: {0: [201, 225, 305, 352], 2.5: [474, 530, 612, 599.33], 5: [705, 738.67, 846.67, 740], 7.5: [870, 913.33, 1106, 970]}
        }
        data = []
        glass_levels = [0, 2.5, 5, 7.5]
        cement_levels = [0, 2.5, 5, 7.5]
        for curing in [7, 14, 28]:
            for cement in cement_levels:
                for glass in glass_levels:
                    gi = glass_levels.index(glass)
                    base = {'cement': cement, 'glass': glass, 'curing': curing}
                    data.append({**base, 'compaction': 'BSL', 'compaction_num': 0, 'UCS': bslUCS[curing][cement][gi]})
                    data.append({**base, 'compaction': 'WAS', 'compaction_num': 1, 'UCS': wasUCS[curing][cement][gi]})
        df = pd.DataFrame(data)
        df['cement_glass'] = df['cement'] * df['glass']
        df['cement_sq'] = df['cement'] ** 2
        df['glass_sq'] = df['glass'] ** 2
        print("Loaded 96 hardcoded UCS samples.")
    return df

def evaluate_models_generic(df, target_col='UCS', target_name='UCS'):
    df_valid = df.dropna(subset=[target_col])
    X_base = df_valid[['cement', 'glass', 'curing', 'compaction_num']]
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_base)
    X_tree = X_base.values
    y = df_valid[target_col].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'linear': (LinearRegression(), X_base.values),
        'polynomial': (LinearRegression(), X_poly),
        'random_forest': (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), X_tree),
        'gradient_boosting': (GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42), X_tree)
    }
    results = {}
    oof_preds = {}
    for name, (model, X) in models.items():
        cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
        test_r2, test_rmse = cv_r2.mean(), cv_rmse.mean()

        oof_pred = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            fold_model = type(model)()
            fold_model.fit(X[train_idx], y[train_idx])
            oof_pred[test_idx] = fold_model.predict(X[test_idx])
        oof_preds[name] = oof_pred

        full_model = type(model)()
        if name == 'polynomial':
            full_model.fit(X_poly, y)
        else:
            full_model.fit(X_base.values if name == 'linear' else X_tree, y)

        results[name] = {
            'model': full_model,
            'test_r2': test_r2, 'test_rmse': test_rmse,
            'test_preds': oof_pred, 'test_actual': y
        }
        print(f"{target_name} - {name}: 5-Fold CV R²={test_r2:.4f}, RMSE={test_rmse:.2f}")
    return results

def predict_generic(results, inputs, model_name=None):
    if model_name is None:
        model_name = max(results, key=lambda k: results[k]['test_r2'])
    model = results[model_name]['model']
    inputs['compaction_num'] = 1 if inputs['compaction'] == 'WAS' else 0
    df_in = pd.DataFrame([inputs])
    X_in = df_in[['cement', 'glass', 'curing', 'compaction_num']]
    if model_name == 'polynomial':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_in = poly.fit_transform(X_in)
    pred = max(0, model.predict(X_in)[0])
    rmse = results[model_name]['test_rmse']
    ci_lower = max(0, pred - 1.96 * rmse)
    ci_upper = pred + 1.96 * rmse
    return {'value': pred, 'lower': ci_lower, 'upper': ci_upper, 'model': model_name}

def optimize_generic(results, compaction='WAS'):
    best_name = max(results, key=lambda k: results[k]['test_r2'])
    model = results[best_name]['model']
    cement_range = np.arange(0, 7.75, 0.25)
    glass_range = np.arange(0, 7.75, 0.25)
    best_val = 0
    best_mix = None
    for cement in cement_range:
        for glass in glass_range:
            inputs = {'cement': cement, 'glass': glass, 'curing': 28, 'compaction': compaction}
            pred = predict_generic(results, inputs, best_name)['value']
            if pred > best_val:
                best_val = pred
                best_mix = {'cement': cement, 'glass': glass, 'value': best_val}
    return best_mix, best_name

def plot_results(ucs_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    

    # Bar chart
    models = list(ucs_results.keys())
    r2_scores = [ucs_results[m]['test_r2'] for m in models]
    rmse_scores = [ucs_results[m]['test_rmse'] for m in models]
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='skyblue', edgecolor='black', hatch='---')
    ax1.set_ylabel('R²', fontsize=16)
    ax1.set_ylim(0, 1.05)

    ax1_2 = ax1.twinx()
    bars2 = ax1_2.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral', edgecolor='black', hatch='\\\\')
    ax1_2.set_ylabel('RMSE (kN)', fontsize=16)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=20, ha='right', fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1_2.tick_params(axis='y', labelsize=14)
    ax1.set_title('')
    ax1.legend([bars1, bars2], ['R²', 'RMSE'], loc='upper center', fontsize=12)

    # Scatter plot
    best_name = max(ucs_results, key=lambda k: ucs_results[k]['test_r2'])
    best = ucs_results[best_name]
    actual = best['test_actual']
    preds = best['test_preds']
    r2 = best['test_r2']
    rmse = best['test_rmse']

    ax2.scatter(actual, preds, color='navy', alpha=0.7, edgecolor='black', s=80)
    min_val = min(actual.min(), preds.min())
    max_val = max(actual.max(), preds.max())
    buffer = (max_val - min_val) * 0.05
    ax2.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--', lw=2, label='Perfect fit')
    
    slope, intercept = np.polyfit(actual, preds, 1)
    ax2.plot(actual, slope * actual + intercept, 'g-', lw=2, label=f'y = {slope:.2f}x + {intercept:.1f}')
    
    ax2.set_xlabel('Actual UCS (kN)', fontsize=16)
    ax2.set_ylabel('Predicted UCS (kN)', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_title('')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.1f}', transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig('ucs_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(ucs_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    

    model = ucs_results['gradient_boosting']['model']
    importances = model.feature_importances_
    features = ['cement', 'glass', 'curing', 'compaction_num']
    feature_labels = ['Cement (%)', 'Glass (%)', 'Curing Time (days)', 'Compaction']

    sorted_idx = np.argsort(importances)[::-1]
    sorted_labels = [feature_labels[i] for i in sorted_idx]
    sorted_imp = importances[sorted_idx]

    bars = ax.barh(range(len(sorted_imp)), sorted_imp, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_labels, fontsize=14)
    ax.set_xlabel('Relative Importance', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                va='center', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('ucs_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_pdf(ucs_results, ucs_pred, ucs_opt, df, filename='ucs_report.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ML Soil Stabilization Report (UCS Prediction)", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", styles['Normal']),
        Paragraph(f"Dataset: 96 UCS samples (32 mixes × 3 curing ages)", styles['Normal']),
        Spacer(1, 12),
        Paragraph("UCS Models", styles['Heading2']),
    ]

    table_data = [['Model', 'R²', 'RMSE']]
    for m in ucs_results:
        table_data.append([m.replace('_', ' ').title(), f"{ucs_results[m]['test_r2']:.4f}", f"{ucs_results[m]['test_rmse']:.2f}"])
    story.append(Table(table_data, style=TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey), ('GRID',(0,0),(-1,-1),1,colors.black)])))
    story.append(Spacer(1, 12))

    best_ucs = max(ucs_results, key=lambda k: ucs_results[k]['test_r2'])
    story.append(Paragraph(f"Best UCS Model: {best_ucs.replace('_', ' ').title()} (R²={ucs_results[best_ucs]['test_r2']:.4f})", styles['Normal']))
    story.append(Paragraph(f"Optimal UCS Mix (28d WAS): Cement {ucs_opt['cement']:.2f}%, Glass {ucs_opt['glass']:.2f}% → {ucs_opt['value']:.1f} kN", styles['Normal']))
    story.append(Paragraph(f"Sample Prediction (5% cement + 2.5% glass, 28d WAS): UCS {ucs_pred['value']:.1f} kN "
                            f"(95% CI: [{ucs_pred['lower']:.1f}, {ucs_pred['upper']:.1f}])", styles['Normal']))

    doc.build(story)
    print(f"PDF report saved: {filename}")

if __name__ == "__main__":
    print("Starting UCS Prediction Analysis...")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(csv_path)

    ucs_results = evaluate_models_generic(df, 'UCS', 'UCS')

    sample_inputs = {'cement': 5.0, 'glass': 2.5, 'curing': 28, 'compaction': 'WAS'}
    ucs_pred = predict_generic(ucs_results, sample_inputs)

    ucs_opt, _ = optimize_generic(ucs_results, compaction='WAS')

    best_model_name = max(ucs_results, key=lambda k: ucs_results[k]['test_r2'])
    joblib.dump(ucs_results[best_model_name]['model'], 'best_ucs_model.pkl')
    print(f"Saved best UCS model: {best_model_name}")

    plot_results(ucs_results)
    plot_feature_importance(ucs_results)
    export_pdf(ucs_results, ucs_pred, ucs_opt, df)

    print("\nOptimal UCS Mix:", ucs_opt)
    print("Sample Prediction:", ucs_pred)
    print("Analysis complete!")
