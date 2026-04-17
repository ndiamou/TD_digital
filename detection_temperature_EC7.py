"""
Détection et localisation de défauts du capteur Température — EC7
Méthode : Estimation récursive sur fenêtre glissante (Aitouche, 1990, chap. IV.1)
          + classification du type de défaut (biais, dérive, bruit, blocage)

Auteur  : Alternant Mountakha Ndiaye
Structure calquée sur le code Vitesse+Vibration existant,
adapté à une seule variable : Température.
"""

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ═══════════════════════════════════════════════════════════════════
#  CHRONOMÉTRAGE
# ═══════════════════════════════════════════════════════════════════
profiling = {}

def start_timer(label):
    profiling[label] = time.time()

def stop_timer(label):
    profiling[label] = time.time() - profiling[label]


# ═══════════════════════════════════════════════════════════════════
#  PARAMÈTRES GLOBAUX
# ═══════════════════════════════════════════════════════════════════

# Tu peux mettre :
# 1) soit le chemin d'un dossier contenant un ou plusieurs fichiers Excel
# 2) soit le chemin complet d'un fichier Excel précis

source_donnees = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\stats recherche\Defaut d'un capteur\EC7"

# Fenêtre glissante
T_segment = 7200       # durée d'une fenêtre en secondes (2 h)
pas_glissement = 1800  # pas de glissement en secondes (30 min)

# Années
annee_ref = [2018, 2019]
annees_compare = (2020, 2021, 2022, 2023, 2024, 2025)

# Seuil Aitouche — loi normale centrée réduite, α = 5 %
SEUIL_TCN = 1.96

# Critères de stabilité de la fenêtre glissante (pente + σ résidus)
SEUILS_PENTE = {
    "Temperature": {"pmin": -0.005, "pmax": 0.005, "r2max": 2.0},
}

# Classes de pente pour l'histogramme croisé (en °C/s)
plage_pente_temperature = np.array(
    [-np.inf, -0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01, np.inf]
)

# Palette couleurs par année
PALETTE = {
    2020: "red",
    2021: "blue",
    2022: "orange",
    2023: "pink",
    2024: "brown",
    2025: "purple",
}

# Couleurs par type de défaut
COULEURS_DEFAUT = {
    "Biais positif (offset)": "red",
    "Biais négatif (offset)": "darkred",
    "Dérive croissante (drift)": "orange",
    "Dérive décroissante (drift)": "darkorange",
    "Bruit excessif (fidélité)": "purple",
    "Blocage (capteur figé)": "black",
    "Transitoire / bruit passager": "lightgray",
    "Anomalie non classifiée": "gray",
}


# ═══════════════════════════════════════════════════════════════════
#  LECTURE EXCEL
# ═══════════════════════════════════════════════════════════════════

def trouver_fichier_excel(source):
    """
    Retourne le chemin complet d'un fichier Excel valide.

    Cas possibles :
    - source est un fichier .xlsx
    - source est un dossier contenant un ou plusieurs .xlsx
    """

    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Le chemin spécifié est introuvable :\n{source}\n\n"
            "Vérifie la variable 'source_donnees'."
        )

    # Cas 1 : source = fichier Excel direct
    if os.path.isfile(source):
        if source.lower().endswith(".xlsx") and not os.path.basename(source).startswith("~$"):
            return source
        else:
            raise ValueError(
                f"Le fichier indiqué n'est pas un fichier Excel .xlsx valide :\n{source}"
            )

    # Cas 2 : source = dossier
    if os.path.isdir(source):
        fichiers = [
            f for f in os.listdir(source)
            if f.lower().endswith(".xlsx") and not f.startswith("~$")
        ]

        if not fichiers:
            raise FileNotFoundError(
                f"Aucun fichier Excel valide (.xlsx) trouvé dans le dossier :\n{source}"
            )

        # Tri alphabétique pour toujours prendre le même premier fichier
        fichiers = sorted(fichiers)
        return os.path.join(source, fichiers[0])

    raise ValueError("La source fournie n'est ni un fichier valide ni un dossier.")


def lire_premier_excel(source):
    """
    Lit le fichier Excel trouvé.
    Renomme automatiquement la colonne température quelle que soit sa casse.
    Retourne : (chemin_excel, DataFrame nettoyé).
    """
    chemin = trouver_fichier_excel(source)

    print(f"[INFO] Fichier Excel utilisé : {chemin}")

    start_timer("Lecture Excel")

    df = pd.read_excel(chemin, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Renommage flexible
    mapping = {
        "date": "Date",
        "temperature": "Temperature",
        "temp": "Temperature",
        "température": "Temperature",
    }
    df.columns = [mapping.get(c.lower(), c) for c in df.columns]

    if "Date" not in df.columns:
        raise KeyError(
            "Colonne 'Date' introuvable après renommage. "
            f"Colonnes détectées : {list(df.columns)}"
        )

    if "Temperature" not in df.columns:
        raise KeyError(
            "Colonne 'Temperature' introuvable. "
            "Noms acceptés : 'temperature', 'temp', 'température'. "
            f"Colonnes détectées : {list(df.columns)}"
        )

    # Conversion des types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

    # Suppression des lignes incomplètes
    df.dropna(subset=["Date", "Temperature"], inplace=True)

    if df.empty:
        raise ValueError("Le fichier Excel ne contient aucune ligne exploitable après nettoyage.")

    # Tri chronologique
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Variables dérivées
    df["Secondes"] = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds()
    df["Année"] = df["Date"].dt.year
    df["Jour_annee"] = df["Date"].dt.dayofyear

    stop_timer("Lecture Excel")
    print(f"[Excel] {len(df)} lignes chargées — années : {sorted(df['Année'].unique())}")

    return chemin, df


def filtrer_annees(df, years):
    """Retourne une copie du DataFrame filtrée sur les années données."""
    return df[df["Année"].isin(years)].copy()


# ═══════════════════════════════════════════════════════════════════
#  STABILITÉ : pente + résidus par fenêtre glissante
# ═══════════════════════════════════════════════════════════════════

def analyser_variable(df, value_col, pmin, pmax, r2max, bins_pente):
    """
    Parcourt le signal par fenêtres glissantes.
    Pour chaque fenêtre : régression linéaire → pente + σ résidus.
    Classifie chaque fenêtre : Stable si pente ∈ [pmin, pmax] ET σ ≤ r2max.

    Retourne : (DataFrame des fenêtres, pct stabilité, agrégations).
    """
    secs = df["Secondes"].values
    y_all = df[value_col].values

    if len(secs) == 0:
        return pd.DataFrame(), None, None

    T_tot = secs[-1]
    out = []

    for t in range(0, int(T_tot - T_segment + 1), pas_glissement):
        m = (secs >= t) & (secs < t + T_segment)
        x = secs[m]
        y = y_all[m]

        if len(x) < 12:
            continue

        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (pente * x + intercept)
        R2 = np.std(resid)

        etat = "Stable" if (pmin <= pente <= pmax) and (R2 <= r2max) else "Instable"

        out.append({
            "centre": t + T_segment / 2.0,
            "pente": pente,
            "R2": R2,
            "etat": etat,
        })

    res = pd.DataFrame(out)

    if res.empty:
        return res, None, None

    res["pente_bin"] = pd.cut(res["pente"], bins=bins_pente, include_lowest=True)

    pct = (
        res["etat"]
        .value_counts(normalize=True) * 100
    ).reindex(["Stable", "Instable"]).fillna(0).reset_index()
    pct.columns = ["etat", "pourcentage"]

    agg = (
        res.groupby("etat")
        .agg(
            pente_min=("pente", "min"),
            pente_max=("pente", "max"),
            R2_min=("R2", "min"),
            R2_max=("R2", "max"),
        )
        .reset_index()
    )

    glob = pd.DataFrame([{
        "etat": "Global",
        "pente_min": res["pente"].min(),
        "pente_max": res["pente"].max(),
        "R2_min": res["R2"].min(),
        "R2_max": res["R2"].max(),
    }])

    agg = pd.concat([glob, agg], ignore_index=True)

    return res, pct, agg


def calcul_stabilite_toutes_annees(df, var_name):
    """
    Applique les critères de stabilité à toutes les années.
    Retourne un masque booléen (longueur = len(df)) :
      True  → le point appartient à au moins une fenêtre Stable.
    """
    s = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = s["pmin"], s["pmax"], s["r2max"]

    res_mask = np.zeros(len(df), dtype=bool)

    for an in sorted(df["Année"].unique()):
        dfa = df[df["Année"] == an].copy()
        if dfa.empty:
            continue

        secs = dfa["Secondes"].values
        sig = dfa[var_name].values

        if len(secs) < 20:
            continue

        T_loc = secs[-1]

        for t in range(0, int(T_loc - T_segment + 1), pas_glissement):
            idx = (secs >= t) & (secs < t + T_segment)
            x = secs[idx]
            y = sig[idx]

            if len(x) < 12:
                continue

            A = np.vstack([x, np.ones_like(x)]).T
            pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            R2 = np.std(y - (pente * x + intercept))

            if (pmin <= pente <= pmax) and (R2 <= r2max):
                res_mask[dfa.index[idx]] = True

    return res_mask


# ═══════════════════════════════════════════════════════════════════
#  PLOTS CRITÈRES DE STABILITÉ
# ═══════════════════════════════════════════════════════════════════

def plot_scatter_pente_R2(res, var_name, show_r2_threshold=True):
    """Diagramme Pente vs σ résidus — points colorés Stable / Instable."""
    s = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = s["pmin"], s["pmax"], s["r2max"]

    fig = go.Figure()
    st = res[res["etat"] == "Stable"]
    inst = res[res["etat"] == "Instable"]

    if not st.empty:
        fig.add_trace(go.Scatter(
            x=st["R2"], y=st["pente"], mode="markers",
            name="Stable",
            marker=dict(color="green", size=6, opacity=0.7),
        ))

    if not inst.empty:
        fig.add_trace(go.Scatter(
            x=inst["R2"], y=inst["pente"], mode="markers",
            name="Instable",
            marker=dict(color="red", size=6, opacity=0.7),
        ))

    fig.add_hline(y=pmin, line_dash="dash", line_color="gray")
    fig.add_hline(y=pmax, line_dash="dash", line_color="gray")

    if show_r2_threshold:
        fig.add_vline(x=r2max, line_dash="dash", line_color="gray")

    top = (
        f"(pente : {pmin} ≤ p ≤ {pmax}"
        + (f", σ ≤ {r2max} °C" if show_r2_threshold else "")
        + ")"
    )

    fig.update_layout(
        title=f"Diagramme Pente vs σ résidus — {var_name} {top}",
        xaxis_title="σ résidus (°C)",
        yaxis_title="Pente (°C/s)",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def plot_histogram_croise(res, var_name):
    """
    Histogramme croisé :
      % segments  = nb segments dans une classe / nb total segments
      % résidus   = somme(σ) dans une classe / somme(σ) totale
    """
    pct_segments = res["pente_bin"].value_counts(normalize=True).sort_index() * 100
    r2_sum = res.groupby("pente_bin")["R2"].sum().reindex(pct_segments.index)
    total = r2_sum.sum()
    pct_res = (r2_sum / total * 100) if total > 0 else r2_sum * 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_segments.index],
        y=pct_segments.values,
        name="% segments",
        text=[f"{v:.1f}%" for v in pct_segments.values],
        textposition="outside",
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_res.index],
        y=pct_res.values,
        name="% résidus",
        text=[f"{v:.1f}%" for v in pct_res.values],
        textposition="outside",
        marker_color="orange",
    ))
    fig.update_layout(
        title=f"Histogramme croisé — {var_name}",
        xaxis_title="Classes de pente (°C/s)",
        yaxis_title="Pourcentage (%)",
        template="plotly_white",
        barmode="group",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  AITOUCHE IV.1 — σ DE RÉFÉRENCE
# ═══════════════════════════════════════════════════════════════════

def calcul_sigma_temperature(df, stable_ref_mask, years_ref=annee_ref):
    """
    Aitouche p. 78 : la valeur vraie estimée = moyenne des mesures
    sur la référence stable 2018-2019.

    σ_ref = std(T − µ_ref) sur ces points.
    """
    ref = df.loc[
        stable_ref_mask & df["Année"].isin(years_ref),
        "Temperature"
    ].dropna()

    if ref.empty:
        raise ValueError(
            "Référence stable vide — aucun point stable sur les années de référence."
        )

    if len(ref) < 30:
        raise ValueError(
            f"Pas assez de points référence pour estimer σ (n={len(ref)}, min=30)."
        )

    mu_ref = float(ref.mean())
    sigma_ref = float(ref.std(ddof=0))
    return mu_ref, sigma_ref


# ═══════════════════════════════════════════════════════════════════
#  AITOUCHE IV.1 — CALCUL TCN PAR FENÊTRE GLISSANTE
# ═══════════════════════════════════════════════════════════════════

def calcul_tcn_fenetres(df, stable_global_mask, mu_ref, sigma_ref, years_order=None):
    """
    Pour chaque fenêtre glissante :
        TCN_k = (µ_fenêtre − µ_ref) / σ_ref
    """
    if years_order is None:
        years_order = sorted(df["Année"].unique())

    if sigma_ref == 0:
        raise ValueError("sigma_ref = 0 : signal de référence constant, calcul impossible.")

    rows = []

    for an in years_order:
        dfa = df[df["Année"] == an].copy()
        if dfa.empty:
            continue

        dfa_stable = dfa[stable_global_mask[dfa.index]]
        if dfa_stable.empty:
            continue

        secs = dfa_stable["Secondes"].values
        temp = dfa_stable["Temperature"].values
        doy = dfa_stable["Jour_annee"].values
        T_loc = secs[-1]

        for t in range(0, int(T_loc - T_segment + 1), pas_glissement):
            idx = (secs >= t) & (secs < t + T_segment)
            y = temp[idx]

            if len(y) < 12:
                continue

            mu_fen = float(np.mean(y))
            sigma_fen = float(np.std(y, ddof=0))
            tcn = (mu_fen - mu_ref) / sigma_ref

            if abs(tcn) <= SEUIL_TCN:
                etat = "Normal"
            elif abs(tcn) <= 3.0:
                etat = "Alerte"
            else:
                etat = "Défaut"

            rows.append({
                "Année": an,
                "centre_sec": t + T_segment / 2.0,
                "Jour_annee": float(np.mean(doy[idx])),
                "mu_fen": mu_fen,
                "sigma_fen": sigma_fen,
                "TCN": tcn,
                "etat": etat,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  LOCALISATION DU TYPE DE DÉFAUT
# ═══════════════════════════════════════════════════════════════════

def _creer_ligne(grp, annee, type_defaut):
    return {
        "Année": annee,
        "Jour_debut": float(grp["Jour_annee"].min()),
        "Jour_fin": float(grp["Jour_annee"].max()),
        "Durée_fenêtres": len(grp),
        "TCN_moyen": float(grp["TCN"].mean()),
        "TCN_max_abs": float(grp["TCN"].abs().max()),
        "mu_fen_moy": float(grp["mu_fen"].mean()),
        "Type_defaut": type_defaut,
        "Sévérité": grp["etat"].max(),
    }


def localiser_defaut(tcn_df, sigma_ref):
    """
    Identifie le type de défaut sur les séquences anormales.
    """
    if tcn_df.empty:
        return pd.DataFrame()

    resultats = []

    for an in sorted(tcn_df["Année"].unique()):
        df_an = tcn_df[tcn_df["Année"] == an].sort_values("centre_sec").copy()
        if df_an.empty:
            continue

        df_an["anomalie"] = df_an["etat"].isin(["Alerte", "Défaut"])
        df_an["groupe"] = (df_an["anomalie"] != df_an["anomalie"].shift()).cumsum()

        for _, grp in df_an[df_an["anomalie"]].groupby("groupe"):
            tcn_vals = grp["TCN"].values
            mu_fen_vals = grp["mu_fen"].values
            sigma_fen_vals = grp["sigma_fen"].values

            if len(grp) < 3:
                resultats.append(_creer_ligne(grp, an, "Transitoire / bruit passager"))
                continue

            sigma_mu = float(np.std(mu_fen_vals, ddof=0))
            sigma_fen_mean = float(np.mean(sigma_fen_vals))

            if sigma_mu < 0.05 * sigma_ref and sigma_fen_mean < 0.05 * sigma_ref:
                resultats.append(_creer_ligne(grp, an, "Blocage (capteur figé)"))
                continue

            changements_signe = int(np.sum(np.diff(np.sign(tcn_vals)) != 0))
            sigma_tcn = float(np.std(tcn_vals, ddof=0))

            if changements_signe >= len(tcn_vals) // 2 and sigma_tcn > 2.0:
                resultats.append(_creer_ligne(grp, an, "Bruit excessif (fidélité)"))
                continue

            x_seq = np.arange(len(tcn_vals), dtype=float)
            A = np.vstack([x_seq, np.ones_like(x_seq)]).T
            coeffs = np.linalg.lstsq(A, tcn_vals, rcond=None)[0]
            pente_tcn = float(coeffs[0])
            resid_tendance = float(
                np.std(tcn_vals - (pente_tcn * x_seq + coeffs[1]), ddof=0)
            )

            if abs(pente_tcn) > 0.1 and resid_tendance < sigma_tcn * 0.6:
                sens = "croissante" if pente_tcn > 0 else "décroissante"
                resultats.append(_creer_ligne(grp, an, f"Dérive {sens} (drift)"))
                continue

            pct_positif = float(np.mean(tcn_vals > 0))

            if pct_positif > 0.80:
                resultats.append(_creer_ligne(grp, an, "Biais positif (offset)"))
            elif pct_positif < 0.20:
                resultats.append(_creer_ligne(grp, an, "Biais négatif (offset)"))
            else:
                resultats.append(_creer_ligne(grp, an, "Anomalie non classifiée"))

    return pd.DataFrame(resultats)


# ═══════════════════════════════════════════════════════════════════
#  UTILITAIRE : courbe continue par jour
# ═══════════════════════════════════════════════════════════════════

def courbe_continue_par_jour(df_part, value_col, method="linear"):
    s = df_part.groupby("Jour_annee")[value_col].median()
    s = s.reindex(range(1, 366))
    s = s.interpolate(method=method, limit_direction="both")
    out = s.reset_index()
    out.columns = ["Jour_annee", value_col]
    return out


# ═══════════════════════════════════════════════════════════════════
#  PLOTS TCN
# ═══════════════════════════════════════════════════════════════════

def plot_tcn_annees_superposees(tcn_df, years_ref=annee_ref, years_compare=annees_compare):
    fig = go.Figure()

    ref = tcn_df[tcn_df["Année"].isin(years_ref)]
    if not ref.empty:
        ref_mean = (
            ref.groupby("Jour_annee")["TCN"]
            .mean()
            .reindex(range(1, 366))
            .interpolate(method="linear", limit_direction="both")
        )
        fig.add_trace(go.Scatter(
            x=np.arange(1, 366), y=ref_mean.values,
            mode="lines", name="2018-2019 (réf stable)",
            line=dict(color="black", width=3),
        ))

    for an in sorted(years_compare):
        part = tcn_df[tcn_df["Année"] == an]
        if part.empty:
            continue

        s = (
            part.set_index("Jour_annee")["TCN"]
            .reindex(range(1, 366))
            .interpolate(method="linear", limit_direction="both")
        )

        fig.add_trace(go.Scatter(
            x=np.arange(1, 366), y=s.values,
            mode="lines", name=str(an),
            line=dict(color=PALETTE.get(an, "gray"), width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=+SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="+1.96 (alerte)", annotation_position="top left")
    fig.add_hline(y=-SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="-1.96", annotation_position="bottom left")
    fig.add_hline(y=+3.0, line_dash="dot", line_color="red",
                  annotation_text="+3σ (défaut franc)", annotation_position="top left")
    fig.add_hline(y=-3.0, line_dash="dot", line_color="red",
                  annotation_text="-3σ", annotation_position="bottom left")

    fig.update_traces(line_shape="spline")
    fig.update_layout(
        title="TCN Température — années superposées (Aitouche IV.1)",
        xaxis_title="Jour de l'année",
        yaxis_title="TCN = (µ_fen − µ_ref) / σ_ref",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_tcn_continu(tcn_df,
                     years_order=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
                     out_dir=None, export_html=True):
    pieces = []

    for an in years_order:
        part = tcn_df[tcn_df["Année"] == an][["Jour_annee", "TCN"]].copy()
        if part.empty:
            continue

        s = (
            part.set_index("Jour_annee")["TCN"]
            .reindex(range(1, 366))
            .interpolate(method="linear", limit_direction="both")
            .reset_index()
        )

        s.columns = ["Jour_annee", "TCN"]
        s["X_global"] = an * 366 + s["Jour_annee"].astype(int)
        s["Année"] = an
        pieces.append(s)

    if not pieces:
        raise ValueError("Aucune année exploitable pour la courbe TCN continue.")

    allc = pd.concat(pieces, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=allc["X_global"], y=allc["TCN"],
        mode="lines", name="TCN continu",
        line=dict(width=2, color="black"),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=+SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="+1.96")
    fig.add_hline(y=-SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="-1.96")
    fig.add_hline(y=+3.0, line_dash="dot", line_color="red",
                  annotation_text="+3σ")
    fig.add_hline(y=-3.0, line_dash="dot", line_color="red",
                  annotation_text="-3σ")

    for an in years_order:
        fig.add_vline(x=an * 366 + 1, line_dash="dot", line_color="lightgray")

    tickvals = [an * 366 + 183 for an in years_order]
    ticktext = [str(an) for an in years_order]

    fig.update_layout(
        title="TCN Température — axe continu 2018-2025 (Aitouche IV.1)",
        xaxis_title="Année",
        yaxis_title="TCN = (µ_fen − µ_ref) / σ_ref",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
    )

    if out_dir is not None and export_html:
        path = os.path.join(out_dir, "tcn_temperature_continu.html")
        fig.write_html(path)
        print(f"[TCN continu] HTML sauvegardé : {path}")

    return fig


def plot_localisation_defauts(localisation_df, tcn_df):
    fig = go.Figure()

    for an in sorted(tcn_df["Année"].unique()):
        part = tcn_df[tcn_df["Année"] == an].sort_values("Jour_annee")
        if part.empty:
            continue

        fig.add_trace(go.Scatter(
            x=part["Jour_annee"], y=part["TCN"],
            mode="lines", name=str(an),
            line=dict(color=PALETTE.get(an, "gray"), width=1.5),
            opacity=0.6,
        ))

    for _, row in localisation_df.iterrows():
        couleur = COULEURS_DEFAUT.get(row["Type_defaut"], "gray")

        fig.add_vrect(
            x0=row["Jour_debut"],
            x1=row["Jour_fin"],
            fillcolor=couleur,
            opacity=0.15,
            line_width=0,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=+SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="+1.96 (alerte)")
    fig.add_hline(y=-SEUIL_TCN, line_dash="dash", line_color="orange",
                  annotation_text="-1.96")
    fig.add_hline(y=+3.0, line_dash="dot", line_color="red",
                  annotation_text="+3σ (défaut)")
    fig.add_hline(y=-3.0, line_dash="dot", line_color="red",
                  annotation_text="-3σ")

    fig.update_layout(
        title="Localisation des types de défauts — Température EC7",
        xaxis_title="Jour de l'année",
        yaxis_title="TCN = (µ_fen − µ_ref) / σ_ref",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_temperature_domaine_temporel(df, stable_temp_ref, seuil_min=None):
    fig = go.Figure()

    ref_all = df.loc[
        stable_temp_ref & df["Année"].isin(annee_ref),
        ["Année", "Jour_annee", "Temperature"]
    ].copy()

    ref_all["Temperature"] = pd.to_numeric(ref_all["Temperature"], errors="coerce")
    ref_all.dropna(subset=["Temperature"], inplace=True)

    if seuil_min is not None:
        ref_all = ref_all[ref_all["Temperature"] > seuil_min]

    curves = []
    for y in annee_ref:
        part = ref_all[ref_all["Année"] == y][["Jour_annee", "Temperature"]]
        if part.empty:
            continue
        curves.append(courbe_continue_par_jour(part, "Temperature"))

    if curves:
        ref_curve_df = pd.concat(
            [c.set_index("Jour_annee")["Temperature"] for c in curves],
            axis=1
        )
        ref_mean = (
            ref_curve_df.mean(axis=1)
            .reindex(range(1, 366))
            .astype(float)
            .interpolate(method="linear", limit_direction="both")
        )

        fig.add_trace(go.Scatter(
            x=np.arange(1, 366), y=ref_mean.values,
            mode="lines", name="2018-2019 (réf stable)",
            line=dict(color="black", width=3),
        ))

    for an in sorted(df["Année"].unique()):
        if an in annee_ref:
            continue

        part = df[df["Année"] == an][["Jour_annee", "Temperature"]].dropna().copy()

        if seuil_min is not None:
            part = part[part["Temperature"] > seuil_min]

        if part.empty:
            continue

        cont = courbe_continue_par_jour(part, "Temperature")

        fig.add_trace(go.Scatter(
            x=cont["Jour_annee"], y=cont["Temperature"],
            mode="lines", name=str(an),
            line=dict(color=PALETTE.get(an, "gray"), width=2),
        ))

    fig.update_traces(line_shape="spline")
    fig.update_layout(
        title="Température — domaine temporel (jour de l'année)",
        xaxis_title="Jour de l'année",
        yaxis_title="Température (°C)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # 1. Lecture des données
    chemin, df = lire_premier_excel(source_donnees)
    out_dir = os.path.dirname(chemin)

    # 2. Stabilité sur toutes les années
    start_timer("Stabilité toutes années")
    stable_temp_all = calcul_stabilite_toutes_annees(df, "Temperature")
    stable_global = stable_temp_all
    stop_timer("Stabilité toutes années")

    nb_stable = int(stable_global.sum())
    print(f"[Stabilité] {nb_stable} points stables / {len(df)} total ({100 * nb_stable / len(df):.1f} %)")

    stable_ref = stable_global & df["Année"].isin(annee_ref)

    # 3. Critères + histogramme croisé
    df_ref = filtrer_annees(df, annee_ref)
    bins = plage_pente_temperature
    s = SEUILS_PENTE["Temperature"]

    res, pct, agg = analyser_variable(
        df_ref, "Temperature", s["pmin"], s["pmax"], s["r2max"], bins
    )

    if res is not None and not res.empty:
        print(f"[Critères] {len(res)} fenêtres analysées sur la référence.")

        if pct is not None:
            print(pct.to_string(index=False))

        plot_scatter_pente_R2(res, "Temperature", show_r2_threshold=True).write_html(
            os.path.join(out_dir, "criteres_stabilite_Temperature.html")
        )

        plot_histogram_croise(res, "Temperature").write_html(
            os.path.join(out_dir, "histogramme_croise_Temperature.html")
        )
    else:
        print("[Critères] Aucun segment exploitable sur la référence.")

    # 4. Domaine temporel brut
    stable_temp_ref = stable_temp_all & df["Année"].isin(annee_ref)
    fig_temp = plot_temperature_domaine_temporel(df, stable_temp_ref)
    fig_temp.write_html(os.path.join(out_dir, "temperature_temporelle.html"))

    # 5. σ de référence
    try:
        mu_ref, sigma_ref = calcul_sigma_temperature(df, stable_ref)
        print(
            f"[Sigma] µ_ref = {mu_ref:.3f} °C | "
            f"σ_ref = {sigma_ref:.3f} °C | "
            f"seuil alerte = ±{SEUIL_TCN * sigma_ref:.3f} °C"
        )
    except Exception as e:
        print(f"[Sigma] ERREUR : {e}")
        return

    # 6. Calcul du TCN
    start_timer("Calcul TCN")
    tcn_df = calcul_tcn_fenetres(
        df,
        stable_global,
        mu_ref,
        sigma_ref,
        years_order=list(annee_ref) + list(annees_compare),
    )
    stop_timer("Calcul TCN")

    print(f"[TCN] {len(tcn_df)} fenêtres calculées")

    if not tcn_df.empty:
        print(
            tcn_df.groupby(["Année", "etat"])
            .size()
            .unstack(fill_value=0)
            .to_string()
        )

    # 7. Plots TCN
    if not tcn_df.empty:
        plot_tcn_annees_superposees(tcn_df).write_html(
            os.path.join(out_dir, "tcn_temperature_superpose.html")
        )

        plot_tcn_continu(
            tcn_df,
            years_order=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
            out_dir=out_dir,
            export_html=True,
        )

    # 8. Localisation du type de défaut
    if not tcn_df.empty:
        start_timer("Localisation")
        localisation_df = localiser_defaut(tcn_df, sigma_ref)
        stop_timer("Localisation")

        if not localisation_df.empty:
            print("\n--- Défauts localisés ---")
            cols_affich = [
                "Année", "Jour_debut", "Jour_fin",
                "Type_defaut", "Sévérité", "TCN_max_abs", "Durée_fenêtres",
            ]
            print(localisation_df[cols_affich].to_string(index=False))

            path_excel = os.path.join(out_dir, "localisation_defauts.xlsx")
            localisation_df.to_excel(path_excel, index=False)
            print(f"[Localisation] Tableau sauvegardé : {path_excel}")

            fig_loc = plot_localisation_defauts(localisation_df, tcn_df)
            fig_loc.write_html(os.path.join(out_dir, "localisation_defauts.html"))
        else:
            print("[Localisation] Aucun défaut détecté sur la période analysée.")

    # 9. Résumé des temps d'exécution
    print("\n--- Temps d'exécution ---")
    for k, v in profiling.items():
        print(f"  {k:<35s}: {v:.2f} s")


if __name__ == "__main__":
    main()