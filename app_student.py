import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
import matplotlib.pyplot as plt
from app_utils import (
    get_all_node_options,
    describe_selected_node,
    filter_sample_for_node,
    build_tree_lines,
    collect_conditions
)

# ---------------------------
# General page configuration
# ---------------------------
st.set_page_config(
    page_title="Árvore de Decisão – Titanic",
    layout="wide"
)

st.title("🛳️ Árvores de Decisão – Titanic")
st.write(
    "Nesta atividade vais construir **manualmente** uma árvore de decisão "
    "para prever se um passageiro sobreviveu ao Titanic."
)

# ---------------------------
# Group identification
# ---------------------------
group_id = st.number_input(
    "Número do grupo",
    value=0,
)

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    # Load the Titanic dataset from a local CSV file
    df = pd.read_csv("titanic/train.csv")
    return df

df = load_data()

# Select a simplified set of features and target
target = "Survived"
features = df.columns.drop(target)  # Exclude less relevant features

# Only drop rows where the target itself is missing (preserves the real ~38% survival rate).
# Rows with missing feature values are kept; NaNs are handled at plot/split time per feature.
df = df.dropna(subset=[target])

# Store tree structure persistently during the session
if "tree_nodes" not in st.session_state:
    st.session_state.tree_nodes = {
        "root": {
            "id": "root",
            "feature": None,
            "operator": None,
            "value": None,
            "prediction": None,
            "depth": 0,
            "children": {"true": None, "false": None}
        }
    }
    st.session_state.next_node_id = 1

# Each group works with a random subset of the data
df_group_sample = df.sample(frac=0.3, random_state=group_id)

st.subheader("📋 Dados atribuídos ao grupo")
st.dataframe(df_group_sample, height=400)

node_options = get_all_node_options(include_all_nodes=True)
node_option_labels = [label for _, label in node_options]
node_option_ids = [node_id for node_id, _ in node_options]

st.subheader("Seleciona o nó a explorar")
selected_node_label = st.selectbox(
    "Começa pela raiz. Volta aqui depois de fazeres uma divisão para explorares os novos nós criados.",
    node_option_labels,
    key="selected_node"
)
selected_node_id = node_option_ids[node_option_labels.index(selected_node_label)]

selected_node_conditions = describe_selected_node(selected_node_id)
if selected_node_conditions:
    st.caption("Amostra filtrada por: " + " e ".join(f"`{condition}`" for condition in selected_node_conditions))
else:
    st.caption("A explorar a amostra completa do grupo (raiz).")

df_sample = filter_sample_for_node(df_group_sample, selected_node_id)

# ---------------------------
# Data exploration
# ---------------------------
st.subheader("Vê o efeito de aplicar diferentes divisões")

if df_sample.empty:
    st.warning("Não há passageiros neste nó para explorar.")
    selected_feature = features[0]
else:
    selected_feature = st.selectbox("Seleciona a variável segundo a qual queres fazer a divisão.", features)

controls_col, plot_col, branch_col = st.columns([1, 2, 2], gap="large")

# Drop NaN for the selected feature only — keeps full sample size for other features
df_sample_feat = df_sample.dropna(subset=[selected_feature]) if not df_sample.empty else df_sample

is_numeric_feature = (not df_sample_feat.empty) and pd.api.types.is_numeric_dtype(df_sample_feat[selected_feature])
is_low_cardinality_integer = (
    (not df_sample_feat.empty)
    and pd.api.types.is_integer_dtype(df_sample_feat[selected_feature])
    and df_sample_feat[selected_feature].nunique() < 10
)

highlighted_categories = []

if df_sample.empty:
    with controls_col:
        st.info("Sem dados para definir uma nova condição neste nó.")
    with plot_col:
        st.info("Sem gráfico disponível.")
elif is_numeric_feature and not is_low_cardinality_integer:
    with controls_col:
        feature_min = df_sample_feat[selected_feature].min()
        feature_max = df_sample_feat[selected_feature].max()

        if pd.api.types.is_integer_dtype(df_sample_feat[selected_feature]):
            marker_value = st.slider(
                f"Limiar da seleção em {selected_feature}",
                min_value=int(feature_min),
                max_value=int(feature_max),
                value=int((feature_min + feature_max) / 2)
            )
        else:
            slider_step = float((feature_max - feature_min) / 100) if feature_max > feature_min else 0.1
            marker_value = st.slider(
                f"Limiar da seleção em {selected_feature}",
                min_value=float(feature_min),
                max_value=float(feature_max),
                value=float((feature_min + feature_max) / 2),
                step=slider_step
            )

    with plot_col:
        st.write(f"Histograma de {selected_feature}")
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(df_sample_feat[selected_feature].min(), df_sample_feat[selected_feature].max() + 1, 10)
        df_sample_feat[df_sample_feat[target] == 1][selected_feature].hist(
            alpha=0.6, label="Sobreviveu", ax=ax, color="#1f77b4", bins=bins
        )
        df_sample_feat[df_sample_feat[target] == 0][selected_feature].hist(
            alpha=0.6, label="Não sobreviveu", ax=ax, color="#d62728", bins=bins
        )
        ax.axvline(marker_value, color="black", linestyle="--", linewidth=2, label="Limiar")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Número de passageiros")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
else:
    counts = df_sample_feat.groupby([selected_feature, target]).size().unstack(fill_value=0)

    with controls_col:
        st.write(f"Categorias a selecionar em {selected_feature}")
        highlighted_categories = []
        with st.container(height=320):
            for category in counts.index:
                if st.checkbox(str(category), key=f"highlight_{selected_feature}_{category}"):
                    highlighted_categories.append(category)

    with plot_col:
        st.write(f"Distribuição de {selected_feature}")
        fig, ax = plt.subplots(figsize=(6, 4))
        x_positions = list(range(len(counts.index)))
        bar_width = 0.35

        for i, category in enumerate(counts.index):
            alpha = 1.0 if category in highlighted_categories else 0.35
            ax.bar(
                x_positions[i] - bar_width / 2,
                counts.loc[category].get(0, 0),
                width=bar_width,
                color="#d62728",
                alpha=alpha,
                label="Não sobreviveu" if i == 0 else ""
            )
            ax.bar(
                x_positions[i] + bar_width / 2,
                counts.loc[category].get(1, 0),
                width=bar_width,
                color="#1f77b4",
                alpha=alpha,
                label="Sobreviveu" if i == 0 else ""
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Número de passageiros")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

# ---------------------------
# Branch preview 
# ---------------------------

# Build the current condition from the data exploration section
if df_sample.empty:
    current_feature = selected_feature
    current_operator = None
    current_value = None
    mask = pd.Series([False] * len(df_sample), index=df_sample.index)
    branch_true_label = "Sem dados"
    branch_false_label = "Sem dados"
elif is_numeric_feature and not is_low_cardinality_integer:
    current_feature = selected_feature
    current_operator = "<="
    current_value = marker_value
    mask = df_sample[current_feature] <= current_value
    branch_true_label = f"{current_feature} ≤ {current_value}"
    branch_false_label = f"{current_feature} > {current_value}"
else:
    current_feature = selected_feature
    current_operator = "==" if len(highlighted_categories) == 1 else "in"
    if highlighted_categories:
        current_value = highlighted_categories[0] if len(highlighted_categories) == 1 else highlighted_categories
        mask = df_sample[current_feature].isin(highlighted_categories)
        branch_true_label = f"{current_feature} ∈ {highlighted_categories}"
        branch_false_label = f"{current_feature} ∉ {highlighted_categories}"
    else:
        # No category highlighted: use all vs none (trivial split)
        mask = pd.Series([False] * len(df_sample), index=df_sample.index)
        current_value = None
        branch_true_label = f"{current_feature} (nenhuma categoria selecionada)"
        branch_false_label = f"{current_feature} (todas)"

branch_true = df_sample[mask]
branch_false = df_sample[~mask]

with branch_col:
    st.write(branch_true_label+"?")
    fig_branch, ax_branch = plt.subplots(figsize=(6, 4))
    branch_labels = ["Verdadeiro","Falso"]
    survived_counts = [
        branch_true[target].sum(),
        branch_false[target].sum()
    ]
    not_survived_counts = [
        (branch_true[target] == 0).sum(),
        (branch_false[target] == 0).sum()
    ]
    x_pos = [0, 1]
    bar_w = 0.35
    ax_branch.bar([x - bar_w/2 for x in x_pos], not_survived_counts, width=bar_w, color="#d62728", label="Não sobreviveu")
    ax_branch.bar([x + bar_w/2 for x in x_pos], survived_counts, width=bar_w, color="#1f77b4", label="Sobreviveu")
    ax_branch.set_xticks(x_pos)
    ax_branch.set_xticklabels(branch_labels, wrap=True)
    ax_branch.set_ylabel("Número de passageiros")
    ax_branch.legend()
    st.pyplot(fig_branch, use_container_width=True)

add_rule_col, current_tree_col = st.columns([2, 3], gap="large")

with add_rule_col:
    # ---------------------------
    # Manual decision tree construction
    # ---------------------------
    st.subheader("Adiciona a divisão à árvore")

    st.write("Se a divisão encontrada separa bem os sobreviventes dos não sobreviventes, guarda-a na árvore de decisão.")

    selected_node = st.session_state.tree_nodes[selected_node_id]
    if selected_node["depth"] < 2:
        st.write(f"**No nó `{selected_node_id}`, aplicar condição {branch_true_label}.**")

        # Add rule from current exploration condition
        rule_pred_col, rule_btn_col = st.columns([1, 1])
        with rule_pred_col:
            prediction = st.selectbox(
                "Previsão do ramo verdadeiro",
                [1, 0],
                format_func=lambda x: "Sobrevive" if x == 1 else "Não sobrevive",
                key="rule_prediction"
            )
        with rule_btn_col:
            st.write("")
            st.write("")
            if st.button("➕ Adicionar divisão"):
                if current_operator is not None and current_value is not None:
                    new_node_id_true = f"node_{st.session_state.next_node_id}"
                    st.session_state.next_node_id += 1
                    new_node_id_false = f"node_{st.session_state.next_node_id}"
                    st.session_state.next_node_id += 1

                    selected_node["feature"] = current_feature
                    selected_node["operator"] = current_operator
                    selected_node["value"] = current_value
                    selected_node["children"]["true"] = new_node_id_true
                    selected_node["children"]["false"] = new_node_id_false

                    st.session_state.tree_nodes[new_node_id_true] = {
                        "id": new_node_id_true,
                        "feature": None,
                        "operator": None,
                        "value": None,
                        "prediction": prediction,
                        "depth": selected_node["depth"] + 1,
                        "children": {"true": None, "false": None},
                        "parent": selected_node_id
                    }
                    st.session_state.tree_nodes[new_node_id_false] = {
                        "id": new_node_id_false,
                        "feature": None,
                        "operator": None,
                        "value": None,
                        "prediction": 1 - prediction,
                        "depth": selected_node["depth"] + 1,
                        "children": {"true": None, "false": None},
                        "parent": selected_node_id
                    }
                    st.rerun()
                    st.success("Divisão adicionada! Explora os novos nós criados para continuar a construir a árvore.")
                else:
                    st.warning("Define uma condição válida para criar uma regra.")
    else:
        st.warning("O nó selecionado já está na profundidade máxima.")

with current_tree_col:

    # Display the current tree structure
    st.subheader("Estrutura atual da árvore")

    roots = [
        node_id
        for node_id, node in st.session_state.tree_nodes.items()
        if node.get("parent") is None
    ]

    if roots:
        roots = sorted(roots)

        schematic = []
        for root_id in roots:
            schematic.extend(build_tree_lines(root_id))

        st.markdown("```text\n" + "\n".join(schematic) + "\n```")

        conditions_by_node = {}

        for root_id in roots:
            collect_conditions(root_id, [], conditions_by_node)

        st.markdown("**Condições por nó:**")
        for node_id in sorted(conditions_by_node):
            node_conds = conditions_by_node[node_id]
            if node_conds:
                st.markdown(f"- **{node_id}**: " + " e ".join(f"`{cond}`" for cond in node_conds))
            else:
                st.markdown(f"- **{node_id}**: *(sem condições - raiz)*")

# ---------------------------
# Export decision tree
# ---------------------------
st.subheader("💾 Exportar a árvore")

# Tree structure exported as JSON
tree = {
    "group": group_id,
    "nodes": st.session_state.tree_nodes,
    "default_prediction": "Sobrevive"
}

rule_count = sum(
    1
    for node in st.session_state.tree_nodes.values()
    if node.get("feature") is not None
)

st.download_button(
    label="⬇️ Descarregar árvore (JSON)",
    data=json.dumps(tree, indent=2),
    file_name=f"arvore_{group_id}.json",
    mime="application/json",
    disabled=rule_count < 2,
    help="É preciso criar pelo menos 2 regras na árvore antes de descarregar."
)

if rule_count < 2:
    st.warning("Cria pelo menos 2 regras na árvore antes de a descarregar.")