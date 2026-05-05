import ast
import pandas as pd
import streamlit as st

def get_all_node_options(include_all_nodes=True):
    options = []
    for node_id, node in st.session_state.tree_nodes.items():
        if include_all_nodes or node["depth"] < 2:
            depth_label = "Raiz" if node["depth"] == 0 else f"Profundidade {node['depth']}"
            options.append((node_id, f"{depth_label} - {node_id}" if node_id != "root" else "Raiz"))
    return sorted(options, key=lambda item: (0 if item[0] == "root" else 1, item[0]))

def parse_node_value(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value

def evaluate_condition(series, operator, value):
    parsed_value = parse_node_value(value)

    if operator in ["in", "not in"] or isinstance(parsed_value, (list, tuple, set)):
        values = parsed_value if isinstance(parsed_value, (list, tuple, set)) else [parsed_value]
        result = series.isin(values)
        return ~result if operator == "not in" else result
    if operator == "==":
        return series == parsed_value
    if operator == "!=":
        return series != parsed_value
    if operator == "<=":
        return series <= parsed_value
    if operator == "<":
        return series < parsed_value
    if operator == ">=":
        return series >= parsed_value
    if operator == ">":
        return series > parsed_value
    return pd.Series([True] * len(series), index=series.index)

def get_node_path(node_id):
    path = []
    current_id = node_id

    while current_id != "root":
        current_node = st.session_state.tree_nodes.get(current_id)
        if current_node is None:
            break

        parent_id = current_node.get("parent")
        if parent_id is None:
            break

        parent_node = st.session_state.tree_nodes.get(parent_id)
        if parent_node is None:
            break

        branch = None
        if parent_node.get("children", {}).get("true") == current_id:
            branch = "true"
        elif parent_node.get("children", {}).get("false") == current_id:
            branch = "false"

        path.append((parent_id, branch))
        current_id = parent_id

    return list(reversed(path))

def filter_sample_for_node(dataframe, node_id):
    filtered_df = dataframe.copy()

    for parent_id, branch in get_node_path(node_id):
        parent_node = st.session_state.tree_nodes.get(parent_id, {})
        feature = parent_node.get("feature")
        operator = parent_node.get("operator")
        value = parent_node.get("value")

        if feature is None or operator is None or value is None:
            continue

        condition_mask = evaluate_condition(filtered_df[feature], operator, value)
        filtered_df = filtered_df[condition_mask] if branch == "true" else filtered_df[~condition_mask]

    return filtered_df

def format_condition(feature, operator, value, is_true_branch=True):
    value_repr = repr(parse_node_value(value))
    if is_true_branch:
        return f"{feature} {operator} {value_repr}"

    negated_ops = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        ">": "<=",
        "<=": ">",
        ">=": "<",
        "in": "not in",
        "not in": "in",
    }
    if operator in negated_ops:
        return f"{feature} {negated_ops[operator]} {value_repr}"
    return f"NOT ({feature} {operator} {value_repr})"

def describe_selected_node(node_id):
    path_conditions = []
    for parent_id, branch in get_node_path(node_id):
        parent_node = st.session_state.tree_nodes.get(parent_id, {})
        feature = parent_node.get("feature")
        operator = parent_node.get("operator")
        value = parent_node.get("value")
        if feature is not None and operator is not None:
            path_conditions.append(format_condition(feature, operator, value, branch == "true"))
    return path_conditions

def build_tree_lines(node_id, prefix="", is_last=True):
    line = f"{prefix}{'└── ' if is_last else '├── '}{node_id}"
    lines = [line]

    node = st.session_state.tree_nodes.get(node_id, {})
    children = []
    true_child = node.get("children", {}).get("true")
    false_child = node.get("children", {}).get("false")
    if true_child is not None:
        children.append(true_child)
    if false_child is not None:
        children.append(false_child)

    for idx, child_id in enumerate(children):
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        lines.extend(build_tree_lines(child_id, child_prefix, idx == len(children) - 1))
    return lines

def collect_conditions(node_id, path_conditions, conditions_dict):
    conditions_dict[node_id] = path_conditions
    node = st.session_state.tree_nodes.get(node_id, {})

    feature = node.get("feature")
    operator = node.get("operator")
    value = node.get("value")
    if feature is None or operator is None:
        return

    true_child = node.get("children", {}).get("true")
    false_child = node.get("children", {}).get("false")

    if true_child is not None:
        collect_conditions(
            true_child,
            path_conditions + [format_condition(feature, operator, value, True)],
            conditions_dict
        )
    if false_child is not None:
        collect_conditions(
            false_child,
            path_conditions + [format_condition(feature, operator, value, False)],
            conditions_dict
        )