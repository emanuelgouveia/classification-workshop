import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from app_utils import parse_node_value, evaluate_condition

st.set_page_config(page_title="Professor - Árvores de Decisão", layout="wide")

st.title("Avaliar árvores de decisão")

# ---------------------------
# Load test data
# ---------------------------
@st.cache_data
def load_test_data():
    df = pd.read_csv("titanic/test_augmented.csv")
    return df

@st.cache_data
def load_train_data():
    return pd.read_csv("titanic/train.csv")

df_test = load_test_data()
df_train = load_train_data()

# Minimal preprocessing function used by random forest
def preprocess_for_model(df, fit_encoders=None):
    # select a subset of columns consistent with student app: use all except Name, Ticket, Cabin
    cols = [c for c in df.columns if c not in ("Name", "Ticket", "Cabin")]
    X = df[cols].copy()

    # drop PassengerId from features
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    # Separate target if present
    y = None
    if "Survived" in X.columns:
        y = X["Survived"].astype(int)
        X = X.drop(columns=["Survived"])

    # Impute numeric
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object]).columns.tolist()

    imputer_num = SimpleImputer(strategy="median")
    imputer_cat = SimpleImputer(strategy="most_frequent")

    if fit_encoders is None:
        X[num_cols] = imputer_num.fit_transform(X[num_cols]) if num_cols else X[num_cols]
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols]) if cat_cols else X[cat_cols]
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols]) if cat_cols else X[cat_cols]
        return X, y, {"num_imp": imputer_num, "cat_imp": imputer_cat, "enc": enc, "num_cols": num_cols, "cat_cols": cat_cols}
    else:
        imputer_num = fit_encoders["num_imp"]
        imputer_cat = fit_encoders["cat_imp"]
        enc = fit_encoders["enc"]
        X[num_cols] = imputer_num.transform(X[num_cols]) if num_cols else X[num_cols]
        X[cat_cols] = imputer_cat.transform(X[cat_cols]) if cat_cols else X[cat_cols]
        if cat_cols:
            X[cat_cols] = enc.transform(X[cat_cols])
        return X, y, fit_encoders

# ---------------------------
# Upload student JSONs
# ---------------------------
st.sidebar.header("Upload de árvores (ficheiros JSON)")
uploaded_files = st.sidebar.file_uploader("Carrega os ficheiros .json gerados pelos grupos", accept_multiple_files=True, type=["json"]) 

# helper to evaluate a single tree on a dataframe
def predict_with_tree(tree_nodes, df):
    # start with all True mask
    n = len(df)
    preds = pd.Series([None] * n, index=df.index)

    # we'll collect leaf nodes (nodes with prediction set or no children)
    # traverse nodes and apply conditions to reach leaves
    def node_mask(node_id):
        mask = pd.Series([True] * n, index=df.index)
        # traverse path from root to node
        current_id = node_id
        path = []
        # build reversed path
        while current_id != "root":
            node = tree_nodes.get(current_id)
            if node is None:
                break
            parent = node.get("parent")
            if parent is None:
                break
            parent_node = tree_nodes.get(parent, {})
            # determine branch
            branch = None
            if parent_node.get("children", {}).get("true") == current_id:
                branch = "true"
            elif parent_node.get("children", {}).get("false") == current_id:
                branch = "false"
            path.append((parent_node, branch))
            current_id = parent
        # apply conditions in reverse (from root down)
        for parent_node, branch in reversed(path):
            feature = parent_node.get("feature")
            operator = parent_node.get("operator")
            value = parent_node.get("value")
            if feature is None or operator is None or value is None:
                continue
            # skip the target (students shouldn't split on it, but guard just in case)
            # all other features — including Name, PassengerId, Ticket, Cabin — are
            # present in the raw df_test and will evaluate normally
            if feature == "Survived" or feature not in df.columns:
                continue
            cond = evaluate_condition(df[feature], operator, value)
            mask = mask & (cond if branch == "true" else ~cond)
        return mask

    # find leaves (nodes with prediction not None and no children rules)
    for node_id, node in tree_nodes.items():
        if node.get("prediction") is not None:
            # only consider nodes that are actually reachable (path defined)
            m = node_mask(node_id)
            preds[m] = int(node.get("prediction"))

    # For any remaining None predictions, use default: majority of assigned default or 0
    preds = preds.fillna(0).astype(int)
    return preds

# Parse uploaded JSONs
trees = {}
if uploaded_files:
    for up in uploaded_files:
        try:
            content = up.read().decode('utf-8')
            data = json.loads(content)
            # basic validation
            if 'nodes' in data:
                name = up.name
                trees[name] = data['nodes']
        except Exception as e:
            st.sidebar.error(f"Erro a ler {up.name}: {e}")

# evaluate all trees on test data (test is expected to contain Survived)
results = {}
for name, nodes in trees.items():
    try:
        preds = predict_with_tree(nodes, df_test)
        true = df_test['Survived'].astype(int)
        acc = (preds == true).mean()
        results[name] = { 'preds': preds, 'accuracy': acc }
    except Exception as e:
        results[name] = { 'error': str(e), 'preds': None, 'accuracy': None }

# Sidebar select single tree
st.sidebar.header("Visualizar árvore")
selected_tree_name = st.sidebar.selectbox("Seleciona uma árvore para ver", options=list(trees.keys()) if trees else [])

plot_col, meta_col = st.columns([2, 3], gap='large')

with plot_col:
    st.subheader("Classificação no conjunto de teste")
    if selected_tree_name:
        res = results.get(selected_tree_name, {})
        if res.get('error'):
            st.error(res['error'])
        else:
            preds = res['preds']
            # plot: x = prediction, color = true label
            true = df_test['Survived'].astype(int)
            df_plot = pd.DataFrame({'pred': preds, 'true': true})
            ct = pd.crosstab(df_plot['pred'], df_plot['true']).reindex(index=[0,1], columns=[0,1], fill_value=0)

            fig, ax = plt.subplots(figsize=(4,3))
            x = np.array([0,1])
            width = 0.35
            colors = {0: '#d62728', 1: '#1f77b4'}
            # bars for true=0 and true=1 side-by-side
            ax.bar(x - width/2, ct[0].values, width=width, color=colors[0], label='Não sobreviveu')
            ax.bar(x + width/2, ct[1].values, width=width, color=colors[1], label='Sobreviveu')
            ax.set_xticks(x)
            ax.set_xticklabels(['Previsão:\nNão Sobrevive','Previsão:\nSobrevive'])
            ax.set_ylabel('Número de passageiros')
            ax.legend()
            st.pyplot(fig)

            st.metric('Taxa de acerto', f"{res['accuracy']:.3f}")
    else:
        st.info('Seleciona uma árvore na sidebar para ver a classificação.')

with meta_col:
    st.subheader('Leaderboard das árvores')
    if results:
        rows = []
        for name, r in results.items():
            rows.append({'file': name, 'Taxa de acerto': r.get('accuracy')})
        df_summary = pd.DataFrame(rows).sort_values(by=['Taxa de acerto'], ascending=False, na_position='last')
        st.dataframe(df_summary.reset_index(drop=True))
    else:
        st.info('Nenhuma árvore carregada ainda.')

# Ensemble + Random Forest side by side
st.header('Ensemble e Random Forest')

true = df_test['Survived'].astype(int).values
bins = np.linspace(0, 1, 21)

# Train RF upfront so both plots render together
X_train, y_train, encoders = preprocess_for_model(df_train)
X_test, y_test, _ = preprocess_for_model(df_test, fit_encoders=encoders)
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
if y_train is not None:
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    acc_rf = (rf_preds == true).mean()
else:
    rf_probs = None

col_ens, col_rf = st.columns(2, gap='large')

with col_ens:
    st.subheader('Ensemble das árvores')
    if results:
        pred_matrix = np.vstack([r['preds'].values for r in results.values() if r.get('preds') is not None])
        ensemble_score = pred_matrix.mean(axis=0)

        fig_e, ax_e = plt.subplots(figsize=(4, 3))
        ax_e.hist(ensemble_score[true == 0], bins=bins, alpha=0.6, color='#d62728', edgecolor='#d62728', linewidth=2, label='Não sobreviveu')
        ax_e.hist(ensemble_score[true == 1], bins=bins, alpha=0.6, color='#1f77b4', edgecolor='#1f77b4', linewidth=2, label='Sobreviveu')
        ax_e.set_xlabel('Score médio (0..1)')
        ax_e.set_ylabel('Número de passageiros')
        ax_e.legend()
        st.pyplot(fig_e)

        # best accuracy by thresholding
        thresholds = np.linspace(0, 1, 101)
        best_acc = max(((ensemble_score >= t).astype(int) == true).mean() for t in thresholds)
        st.metric('Melhor taxa de acerto', f'{best_acc:.3f}')
    else:
        st.info('Carrega pelo menos uma árvore para activar o ensemble.')

with col_rf:
    st.subheader('Random Forest')
    if rf_probs is not None:
        fig_rf, ax_rf = plt.subplots(figsize=(4, 3))
        ax_rf.hist(rf_probs[true == 0], bins=bins, alpha=0.6, color='#d62728', edgecolor='#d62728', linewidth=2, label='Não sobreviveu')
        ax_rf.hist(rf_probs[true == 1], bins=bins, alpha=0.6, color='#1f77b4', edgecolor='#1f77b4', linewidth=2, label='Sobreviveu')
        ax_rf.set_xlabel('Probabilidade predita de sobreviver')
        ax_rf.set_ylabel('Número de passageiros')
        ax_rf.legend()
        st.pyplot(fig_rf)
        st.metric('Melhor taxa de acerto', f'{acc_rf:.3f}')
    else:
        st.info('Ficheiro de treino sem rótulos. Não é possível treinar RF.')
