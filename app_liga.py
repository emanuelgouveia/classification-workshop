import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold

st.set_page_config(page_title="1ª Liga – Previsão de Resultados", layout="wide")
st.title("⚽ 1ª Liga Portuguesa – Previsão de Resultados")

# ---------------------------
# Load & concatenate seasons
# ---------------------------
@st.cache_data
def load_raw():
    df24 = pd.read_csv("1aliga/primeira-liga-2024-GMTStandardTime.csv")
    df25 = pd.read_csv("1aliga/primeira-liga-2025-UTC.csv")
    df = pd.concat([df24, df25], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    # Drop rows without a result (unplayed matches)
    df = df[df["Result"].notna() & (df["Result"].str.strip() != "")].copy()
    # Parse goals
    df[["home_goals", "away_goals"]] = (
        df["Result"].str.split(" - ", expand=True).astype(int)
    )
    # Outcome from home team perspective
    df["outcome"] = np.where(
        df["home_goals"] > df["away_goals"], "Home Win",
        np.where(df["home_goals"] < df["away_goals"], "Away Win", "Draw")
    )
    return df.reset_index(drop=True)

raw = load_raw()

# ---------------------------
# Feature engineering helpers
# ---------------------------
def team_stats_before(df, match_idx, team, role, n_games):
    """
    Return form stats for `team` in the last `n_games` matches
    played before match at index `match_idx`.
    `role` is 'home' or 'away' — perspective used for wins/draws/losses.
    """
    past = df.loc[:match_idx - 1]
    team_matches = past[(past["Home Team"] == team) | (past["Away Team"] == team)].tail(n_games)

    if team_matches.empty:
        return dict(gf=0, ga=0, w=0, d=0, l=0)

    gf_list, ga_list, w_list, d_list, l_list = [], [], [], [], []
    for _, row in team_matches.iterrows():
        if row["Home Team"] == team:
            gf, ga = row["home_goals"], row["away_goals"]
        else:
            gf, ga = row["away_goals"], row["home_goals"]
        gf_list.append(gf)
        ga_list.append(ga)
        if gf > ga:
            w_list.append(1); d_list.append(0); l_list.append(0)
        elif gf == ga:
            w_list.append(0); d_list.append(1); l_list.append(0)
        else:
            w_list.append(0); d_list.append(0); l_list.append(1)

    return dict(
        gf=sum(gf_list),
        ga=sum(ga_list),
        w=sum(w_list),
        d=sum(d_list),
        l=sum(l_list),
    )

@st.cache_data
def build_features(df):
    rows = []
    for i, match in df.iterrows():
        home = match["Home Team"]
        away = match["Away Team"]
        past = df.loc[:i - 1]

        # Count how many games each team has played so far
        home_prior = ((past["Home Team"] == home) | (past["Away Team"] == home)).sum()
        away_prior = ((past["Home Team"] == away) | (past["Away Team"] == away)).sum()

        h1 = team_stats_before(df, i, home, "home", 1)
        a1 = team_stats_before(df, i, away, "away", 1)
        h5 = team_stats_before(df, i, home, "home", 5)
        a5 = team_stats_before(df, i, away, "away", 5)

        rows.append({
            # Last 1 game
            "home_gf_last1": h1["gf"],
            "home_ga_last1": h1["ga"],
            "home_w_last1":  h1["w"],
            "home_d_last1":  h1["d"],
            "home_l_last1":  h1["l"],

            "away_gf_last1": a1["gf"],
            "away_ga_last1": a1["ga"],
            "away_w_last1":  a1["w"],
            "away_d_last1":  a1["d"],
            "away_l_last1":  a1["l"],

            # Last 5 games
            "home_gf_last5": h5["gf"],
            "home_ga_last5": h5["ga"],
            "home_w_last5":  h5["w"],
            "home_d_last5":  h5["d"],
            "home_l_last5":  h5["l"],

            "away_gf_last5": a5["gf"],
            "away_ga_last5": a5["ga"],
            "away_w_last5":  a5["w"],
            "away_d_last5":  a5["d"],
            "away_l_last5":  a5["l"],

            "outcome": match["outcome"],
            # keep track so we can filter rows with insufficient history
            "_min_prior": min(home_prior, away_prior),
        })
    feat = pd.DataFrame(rows, index=df.index)  # preserve raw index
    # Keep only matches where both teams had at least 5 prior games
    mask = feat["_min_prior"] >= 5
    kept_idx = feat.index[mask]
    feat = feat.loc[mask].drop(columns=["_min_prior"]).reset_index(drop=True)
    return feat, kept_idx

with st.spinner("A calcular features de forma…"):
    feat_df, kept_idx = build_features(raw)
raw_kept = raw.loc[kept_idx].reset_index(drop=True)

feature_cols = [c for c in feat_df.columns if c != "outcome"]
X = feat_df[feature_cols].values
y = feat_df["outcome"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)   # Away Win=0, Draw=1, Home Win=2

# ---------------------------
# Train Random Forest + cross-validated predictions
# ---------------------------
@st.cache_resource
def train_rf(X, y_enc):
    rf = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_split=20)
    rf.fit(X, y_enc)
    return rf

@st.cache_data
def get_cv_probs(X, y_enc):
    rf_cv = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_split=20)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Returns OOF probabilities in the order of le.classes_
    oof_probs = cross_val_predict(rf_cv, X, y_enc, cv=cv, method="predict_proba")
    oof_preds = oof_probs.argmax(axis=1)
    cv_acc = (oof_preds == y_enc).mean()
    return oof_probs, cv_acc

rf = train_rf(X, y_enc)

with st.spinner("A calcular cross-validation (5-fold)…"):
    cv_probs, cv_acc = get_cv_probs(X, y_enc)

class_order = list(le.classes_)   # ['Away Win', 'Draw', 'Home Win']
palette = {
    "Home Win": "#1f77b4",
    "Draw":     "#ff7f0e",
    "Away Win": "#d62728",
}

# ---------------------------
# Layout: histogram + accuracy
# ---------------------------
st.header("Distribuição das probabilidades previstas (cross-validation)")

hist_cols = st.columns(3, gap="large")
for col_widget, cls in zip(hist_cols, class_order):
    cls_idx = class_order.index(cls)
    cls_probs = cv_probs[:, cls_idx]
    true_mask = (y == cls)
    color = palette[cls]

    with col_widget:
        st.subheader(cls)
        fig, ax = plt.subplots(figsize=(4, 3))
        bins = np.linspace(0, 1, 21)
        ax.hist(cls_probs[~true_mask], bins=bins, alpha=0.6, color="#999999",
                edgecolor="white", linewidth=0.5, label="Outro resultado")
        ax.hist(cls_probs[true_mask], bins=bins, alpha=0.7, color=color,
                edgecolor="white", linewidth=0.5, label=cls)
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Nº de jogos")
        ax.legend(fontsize=8)
        st.pyplot(fig)

st.metric("Taxa de acerto (5-fold CV)", f"{cv_acc:.3f}")

# ---------------------------
# Summary table
# ---------------------------
with st.expander("📋 Ver dados de forma usados no treino"):
    st.dataframe(
        feat_df.assign(
            home=raw_kept["Home Team"].values,
            away=raw_kept["Away Team"].values,
            date=raw_kept["Date"].dt.strftime("%Y-%m-%d").values,
        )[["date", "home", "away"] + feature_cols + ["outcome"]],
        height=350,
    )

# ---------------------------
# Manual prediction
# ---------------------------
st.header("🔮 Previsão manual")
st.write("Introduz as estatísticas de forma de cada equipa para obter uma previsão.")

with st.form("manual_pred"):
    st.subheader("Último jogo")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Casa**")
        h_gf1 = st.number_input("Golos marcados", min_value=0, max_value=20, value=1, key="h_gf1")
        h_ga1 = st.number_input("Golos sofridos",  min_value=0, max_value=20, value=1, key="h_ga1")
        h_w1  = st.selectbox("Vitória",  [0, 1], format_func=lambda x: "Sim" if x else "Não", key="h_w1")
        h_d1  = st.selectbox("Empate",   [0, 1], format_func=lambda x: "Sim" if x else "Não", key="h_d1")
        h_l1  = st.selectbox("Derrota",  [0, 1], format_func=lambda x: "Sim" if x else "Não", key="h_l1")
    with c2:
        st.markdown("**Fora**")
        a_gf1 = st.number_input("Golos marcados", min_value=0, max_value=20, value=1, key="a_gf1")
        a_ga1 = st.number_input("Golos sofridos",  min_value=0, max_value=20, value=1, key="a_ga1")
        a_w1  = st.selectbox("Vitória",  [0, 1], format_func=lambda x: "Sim" if x else "Não", key="a_w1")
        a_d1  = st.selectbox("Empate",   [0, 1], format_func=lambda x: "Sim" if x else "Não", key="a_d1")
        a_l1  = st.selectbox("Derrota",  [0, 1], format_func=lambda x: "Sim" if x else "Não", key="a_l1")

    st.subheader("Últimos 5 jogos")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Casa**")
        h_gf5 = st.number_input("Golos marcados (soma)", min_value=0, max_value=50, value=5, key="h_gf5")
        h_ga5 = st.number_input("Golos sofridos (soma)",  min_value=0, max_value=50, value=5, key="h_ga5")
        h_w5  = st.number_input("Vitórias",  min_value=0, max_value=5, value=2, key="h_w5")
        h_d5  = st.number_input("Empates",   min_value=0, max_value=5, value=1, key="h_d5")
        h_l5  = st.number_input("Derrotas",  min_value=0, max_value=5, value=2, key="h_l5")
    with c4:
        st.markdown("**Fora**")
        a_gf5 = st.number_input("Golos marcados (soma)", min_value=0, max_value=50, value=5, key="a_gf5")
        a_ga5 = st.number_input("Golos sofridos (soma)",  min_value=0, max_value=50, value=5, key="a_ga5")
        a_w5  = st.number_input("Vitórias",  min_value=0, max_value=5, value=2, key="a_w5")
        a_d5  = st.number_input("Empates",   min_value=0, max_value=5, value=1, key="a_d5")
        a_l5  = st.number_input("Derrotas",  min_value=0, max_value=5, value=2, key="a_l5")

    submitted = st.form_submit_button("Prever resultado")

if submitted:
    input_vec = np.array([[
        h_gf1, h_ga1, h_w1, h_d1, h_l1,
        a_gf1, a_ga1, a_w1, a_d1, a_l1,
        h_gf5, h_ga5, h_w5, h_d5, h_l5,
        a_gf5, a_ga5, a_w5, a_d5, a_l5,
    ]])
    pred_probs = rf.predict_proba(input_vec)[0]
    pred_class = le.inverse_transform([pred_probs.argmax()])[0]

    st.subheader("Resultado previsto")
    result_cols = st.columns(3, gap="large")
    for col_widget, cls in zip(result_cols, class_order):
        cls_idx = class_order.index(cls)
        p = pred_probs[cls_idx]
        is_pred = cls == pred_class
        with col_widget:
            st.metric(
                label=("✅ " if is_pred else "") + cls,
                value=f"{p:.1%}",
            )
