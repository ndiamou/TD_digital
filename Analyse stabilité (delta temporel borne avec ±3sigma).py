# Stabilité toutes les années 
# + Corrélation 
# + Critères 
# + Histogramme croisé 
# + domaine temporel 
# + résidus e = Vib - f(Vit) avec bandes ±3σ


import os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Chronométrage
profiling = {}
def start_timer(label): profiling[label] = time.time()
def stop_timer(label): profiling[label] = time.time() - profiling[label]

# Chemin du fichiert
dossier = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\rou_test_withou_zero\EC7"

# Fenêtres glissantes pour les pente de stabilité
T_segment = 7200
pas_glissement = 1800

# Années à analyser
annee_ref = [2018, 2019]
annees_compare = (2020, 2021, 2022, 2023, 2024, 2025)

# Zone vitesse pour la corrélation (limité la droite de regression polynomiale)
V_MIN = 6000
V_MAX = 12000

# Critére des seuils stabilité (pente + résidus)
SEUILS_PENTE = {
    "Vitesse": {"pmin": -0.25, "pmax": 0.25, "r2max": 300.0},
    "Vibration": {"pmin": -0.0005, "pmax": 0.0005, "r2max": 3.0},
}

# Classes de pente (histogramme croisé)
plage_pente_vitesse = np.array([-np.inf, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, np.inf])
plage_pente_vibration = np.array([-np.inf, -0.0015, -0.001, -0.0005, 0, 0.0005, 0.001, 0.0015, np.inf])

# Palette années
PALETTE = {2020:"red", 2021:"blue", 2022:"orange", 2023:"pink", 2024:"brown", 2025:"purple"}


# Lecture Excel
def lire_premier_excel(dossier):

    # Liste tous les fichiers du dossier qui: finissent par ".xlsx"...
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".xlsx") and not f.startswith("~$")]
    # Si aucun fichier Excel "valide" n'est trouvé , s'arrêter avec une erreur explicite
    if not fichiers:
        raise FileNotFoundError("Aucun fichier Excel valide trouvé dans le dossier.")
    chemin = os.path.join(dossier, fichiers[0])

    # Démarre un chronométrage
    start_timer("Lecture Excel")
    df = pd.read_excel(chemin, engine="openpyxl")

    # Nettoie les noms de colonnes: supprime espaces en (début  fin), et harmonise les noms
    df.columns = [c.strip() for c in df.columns]
    mapping = {"date":"Date", "vitesse":"Vitesse", "vibration":"Vibration"}
    df.columns = [mapping.get(c.lower(), c) for c in df.columns]

    # Vérifie que la colonne "Date" existe après de le renommer
    if "Date" not in df.columns:
        raise KeyError("Colonne 'Date' introuvable.")

    # T Convertit "Date" en datetime, et les variables en numerique
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Vitesse" in df.columns: df["Vitesse"] = pd.to_numeric(df["Vitesse"], errors="coerce")
    if "Vibration" in df.columns: df["Vibration"] = pd.to_numeric(df["Vibration"], errors="coerce")

    # Nettoyage
    # Détermine les colonnes "requises" minimales
    req = ["Date"]
    if "Vitesse" in df.columns: req.append("Vitesse")
    if "Vibration" in df.columns: req.append("Vibration")
    # Supprime les lignes où ces colonnes requises contiennent des NaN
    df.dropna(subset=req, inplace=True)

    # Variables temps
    df["Secondes"] = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds()
    # "Année" (numérique) et "Jour_annee"
    df["Année"] = df["Date"].dt.year
    df["Jour_annee"] = df["Date"].dt.dayofyear

    # Arrêter le chronométrage apres le traitement
    stop_timer("Lecture Excel")
    return chemin, df


def filtrer_annees(df, years):
    """Retourne une copie du df filtrée sur les années."""
    return df[df["Année"].isin(years)].copy()


# STABILITÉ : pente + résidus par fenêtre
def analyser_variable(df, value_col, pmin, pmax, r2max, bins_pente):
    # Récupère le vecteur temps (en secondes depuis le début) et la série à analyser
    secs = df["Secondes"].values
    y_all = df[value_col].values
    # si y'a pas donnée, ça renvoie des structures vides
    if len(secs) == 0:
        return pd.DataFrame(), None, None

    # Durée totale (en secondes)
    T_tot = secs[-1]
    out = [] # initialiser le dict

    # Parcours du signal par fenêtres glissantes :
    for t in range(0, int(T_tot - T_segment + 1), pas_glissement):
        m = (secs >= t) & (secs < t + T_segment)
        # x = temps de la fenêtre ; y = valeurs de la variable dans la fenêtre
        x = secs[m]; y = y_all[m]
        if len(x) < 12:
            continue

        # Prépare la matrice de régression [x, 1] pour y = a*x + b
        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (pente * x + intercept)
        R2 = np.std(resid)

        # Classement stabilité selon seuils sur la pente et la dispersion des résidus
        etat = "Stable" if (pmin <= pente <= pmax) and (R2 <= r2max) else "Instable"
        out.append({"centre": t + T_segment/2.0, "pente": pente, "R2": R2, "etat": etat})

    # DataFrame des fenêtres
    res = pd.DataFrame(out)
    if res.empty:
        return res, None, None

    # Binning des pentes selon des classes définies
    res["pente_bin"] = pd.cut(res["pente"], bins=bins_pente, include_lowest=True)
    pct = (res["etat"].value_counts(normalize=True) * 100)\
        .reindex(["Stable", "Instable"]).fillna(0).reset_index() #  Pourcentages par état
    pct.columns = ["etat", "pourcentage"]

    # Agrégations par état : min/max pente et R2
    agg = (res.groupby("etat")
        .agg(pente_min=("pente","min"), pente_max=("pente","max"),
             R2_min=("R2","min"), R2_max=("R2","max"))
        .reset_index())

    glob = pd.DataFrame([{
        "etat":"Global",
        "pente_min":res["pente"].min(), "pente_max":res["pente"].max(),
        "R2_min":res["R2"].min(), "R2_max":res["R2"].max()
    }])
    agg = pd.concat([glob, agg], ignore_index=True)

    return res, pct, agg


# Appliquer la stabilité toutes les années
def calcul_stabilite_toutes_annees(df, var_name):
    """
    Applique les critères stabilité à toutes les années.
    Retourne un masque booléen (len(df)) :
      True si le point appartient à au moins une fenêtre "Stable".
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
            x = secs[idx]; y = sig[idx]
            if len(x) < 12:
                continue

            A = np.vstack([x, np.ones_like(x)]).T
            pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            R2 = np.std(y - (pente * x + intercept))

            if (pmin <= pente <= pmax) and (R2 <= r2max):
                global_idx = dfa.index[idx]
                res_mask[global_idx] = True

    return res_mask

# Plot critères stabilité (pente + residus)
def plot_scatter_pente_R2(res, var_name, show_r2_threshold=True):
    s = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = s["pmin"], s["pmax"], s["r2max"]

    fig = go.Figure()
    st = res[res["etat"] == "Stable"]
    inst = res[res["etat"] == "Instable"]

    if not st.empty:
        fig.add_trace(go.Scatter(
            x=st["R2"], y=st["pente"], mode="markers",
            name="Stable", marker=dict(color="green", size=6, opacity=0.7)
        ))
    if not inst.empty:
        fig.add_trace(go.Scatter(
            x=inst["R2"], y=inst["pente"], mode="markers",
            name="Instable", marker=dict(color="red", size=6, opacity=0.7)
        ))

    fig.add_hline(y=pmin, line_dash="dash", line_color="gray")
    fig.add_hline(y=pmax, line_dash="dash", line_color="gray")
    if show_r2_threshold:
        fig.add_vline(x=r2max, line_dash="dash", line_color="gray")

    top = f"(pente: {pmin} ≤ pente ≤ {pmax}" + (f", R² ≤ {r2max}" if show_r2_threshold else "") + ")"
    fig.update_layout(
        title=f"Diagramme Pente vs Résidu — {var_name} {top}",
        xaxis_title="R² (std résidus)",
        yaxis_title="Pente",
        template="plotly_white",
        hovermode="closest"
    )
    return fig

# Plot critére stabilité (histogramme croisé)
def plot_histogram_croise(res, var_name):
    """
    % segments = nb segments dans une classe / nb total segments
    % résidus = somme(R2) dans une classe / somme(R2) totale
    """
    pct_segments = (res["pente_bin"].value_counts(normalize=True).sort_index() * 100)
    r2_sum_per_bin = res.groupby("pente_bin")["R2"].sum().reindex(pct_segments.index)
    total_r2 = r2_sum_per_bin.sum()
    pct_residus = (r2_sum_per_bin / total_r2 * 100) if total_r2 > 0 else r2_sum_per_bin * 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_segments.index],
        y=pct_segments.values,
        name="segments",
        text=[f"{v:.1f}%" for v in pct_segments.values],
        textposition="outside",
        marker_color="steelblue"
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_residus.index],
        y=pct_residus.values,
        name="résidus",
        text=[f"{v:.1f}%" for v in pct_residus.values],
        textposition="outside",
        marker_color="orange"
    ))
    fig.update_layout(
        title=f"Histogramme croisé — {var_name}",
        xaxis_title="Classes de pente",
        yaxis_title="Pourcentage (%)",
        template="plotly_white",
        barmode="group"
    )
    return fig

# Correlation + la droite polynômiale
def plot_correlation_by_year(df, stable_ref_mask, deg=4):
    fig = go.Figure()

    ref_pts = df[stable_ref_mask]
    fig.add_trace(go.Scatter(
        x=ref_pts["Vitesse"], y=ref_pts["Vibration"],
        mode="markers", name="2018-2019 (stable)",
        marker=dict(color="green", size=3), opacity=0.7
    ))

    for an in sorted(df["Année"].unique()):
        if an in annee_ref:
            continue
        subset = df[df["Année"] == an]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Vitesse"], y=subset["Vibration"],
            mode="markers", name=str(an),
            marker=dict(color=PALETTE.get(an, "gray"), size=3), opacity=0.7
        ))

    # Polynomiale sur zone vitesse
    zone = ref_pts[(ref_pts["Vitesse"] >= V_MIN) & (ref_pts["Vitesse"] <= V_MAX)]
    X = zone["Vitesse"].to_numpy(dtype=float)
    Y = zone["Vibration"].to_numpy(dtype=float)

    if len(X) > 20:
        ok = np.isfinite(X) & np.isfinite(Y)
        poly = np.poly1d(np.polyfit(X[ok], Y[ok], deg))
        X_fit = np.linspace(X[ok].min(), X[ok].max(), 500)
        Y_fit = poly(X_fit)

        fig.add_trace(go.Scatter(
            x=X_fit, y=Y_fit, mode="lines",
            line=dict(color="darkred", width=3),
            name=f"Régression polynomiale deg={deg} (réf)"
        ))

    fig.update_layout(
        title="Corrélation : Vitesse & Vibration (réf = zones stables 2018-2019)",
        xaxis_title="Vitesse",
        yaxis_title="Vibration (µm)",
        template="plotly_white"
    )
    return fig

# Domaine temporel vibration brute année par année
def courbe_continue_par_jour(df_year, value_col, method="linear"):
    s = df_year.groupby("Jour_annee")[value_col].median()
    s = s.reindex(range(1, 366))
    s = s.interpolate(method=method, limit_direction="both")
    out = s.reset_index()
    out.columns = ["Jour_annee", value_col]
    return out


# Supperpose les années
def supperposition_année(df, stable_mask_vib_ref, years_ref=annee_ref, seuil_min=8):
    fig = go.Figure()
    var_name = "Vibration"

    # Référence stable (2018-2019)
    ref_all = df.loc[stable_mask_vib_ref & df["Année"].isin(years_ref), ["Année","Jour_annee",var_name]].copy()
    ref_all[var_name] = pd.to_numeric(ref_all[var_name], errors="coerce")
    ref_all = ref_all.dropna(subset=[var_name])
    ref_all = ref_all[ref_all[var_name] > seuil_min]

    curves = []
    for y in years_ref:
        part = ref_all[ref_all["Année"] == y][["Jour_annee", var_name]]
        if part.empty:
            continue
        curves.append(courbe_continue_par_jour(part, var_name))

    if curves:
        ref_curve_df = pd.concat([c.set_index("Jour_annee")[var_name] for c in curves], axis=1)
        ref_mean = ref_curve_df.mean(axis=1).reindex(range(1, 366)).astype(float)
        ref_mean = ref_mean.interpolate(method="linear", limit_direction="both")

        fig.add_trace(go.Scatter(
            x=np.arange(1, 366),
            y=ref_mean.values,
            mode="lines",
            name="2018-2019 (réf stable)",
            line=dict(color="black", width=3)
        ))

    # Autres années
    for y in sorted(df["Année"].unique()):
        if y in years_ref:
            continue
        part = df[df["Année"] == y][["Jour_annee", var_name]].dropna().copy()
        if part.empty:
            continue
        part[var_name] = pd.to_numeric(part[var_name], errors="coerce")
        part = part.dropna(subset=[var_name])
        part = part[part[var_name] > seuil_min]
        if part.empty:
            continue

        cont = courbe_continue_par_jour(part, var_name)
        fig.add_trace(go.Scatter(
            x=cont["Jour_annee"], y=cont[var_name],
            mode="lines", name=str(y),
            line=dict(color=PALETTE.get(y, "gray"), width=2)
        ))

    fig.update_traces(line_shape="spline")
    fig.update_layout(
        title="Vibration — domaine temporel (jour de l’année)",
        xaxis_title="Temps (jour de l’année)",
        yaxis_title="Vibration (µm)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# Calcul sigma
def calcul_sigma(df,stable_ref_mask,years_ref=annee_ref,deg=4,vmin=V_MIN,vmax=V_MAX):
    """
    Vibration = f(Vitesse) sur la référence stable (2018-2019),
    puis calcule e = Vib - f(Vit) sur ces points.
    Retourne: poly, mu_e, sigma_e (µm).
    """
    # verifier que le dataframe contio bel et bien les collonnes vitesse et vibration
    if "Vitesse" not in df.columns or "Vibration" not in df.columns:
        raise KeyError("Il faut les colonnes 'Vitesse' et 'Vibration'.")
     # Sélectionne la sous-partie de df correspondant à la référence stable
    ref = df.loc[stable_ref_mask & df["Année"].isin(years_ref), ["Vitesse","Vibration"]].dropna().copy()
    # Si après filtrage la référence est vide, on ne peut pas estimer le modèle → erreur explicite
    if ref.empty:
        raise ValueError("Référence stable vide pour sigma (corrélation).")

    # Restreint la référence à une plage de vitesse [vmin, vmax], pour éviter
    ref = ref[(ref["Vitesse"] >= vmin) & (ref["Vitesse"] <= vmax)]
    
    # S'assurer que j'ai un volume minimal de points pour un ajustement fiable.
    # Ici, c'est fixé à 30

    if len(ref) < 30:
        raise ValueError(f"Pas assez de points ref dans [{vmin},{vmax}] (n={len(ref)}).")
     # Convertit les colonnes en tableaux numpy de type float pour l'ajustement
    x = ref["Vitesse"].to_numpy(dtype=float)
    y = ref["Vibration"].to_numpy(dtype=float)

    # recuperer le polynome et faire ma différence entre la vibration mesurée et le modèle prédit
    poly = np.poly1d(np.polyfit(x, y, deg))
    resid = y - poly(x)
    # Moyenne et ecartype des résidus (biais), Si le modèle capte bien la tendance
    mu_e = float(np.mean(resid))
    sigma_e = float(np.std(resid, ddof=0)) # population

    return poly, mu_e, sigma_e

# domaine temporel: résidus e = Vib - f(Vit)
def calcul_difference_supperposee(
    df,
    poly,
    sigma_e,
    stable_global_mask,
    years_ref=annee_ref,
    years_compare=annees_compare,
    vmin=V_MIN,
    vmax=V_MAX,
    out_dir=None,
    export_html=True
):
    """
    - résidus e = Vib - f(Vit) sur points STABLES
    - bandes ±3σ
    """
    # Ne conserve que les colonnes utiles, sur les lignes marquées "stables"
    d = df.loc[stable_global_mask, ["Année","Jour_annee","Vitesse","Vibration"]].dropna().copy()
    d = d[(d["Vitesse"] >= vmin) & (d["Vitesse"] <= vmax)]
    if d.empty:
        raise ValueError("Aucun point stable exploitable (résidus DOY).")
    # Calcule le résidu e = mesuré - prédit, en évaluant le polynôme "poly" sur la vitesse.
    d["Resid"] = d["Vibration"].to_numpy(dtype=float) - poly(d["Vitesse"].to_numpy(dtype=float))
    # initialisation
    fig = go.Figure()

    # Référence 2018-2019
    ref = d[d["Année"].isin(years_ref)][["Année","Jour_annee","Resid"]].copy()
    curves_ref = [] # stocker les courbes continue par année
    for y in years_ref:
        part = ref[ref["Année"] == y][["Jour_annee","Resid"]]
        if part.empty:  
            # Si une année de référence n'a pas de points stables, on la saute
            continue
        # Transforme les points en courbe "continue" par jour
        curves_ref.append(courbe_continue_par_jour(part, "Resid"))
    
    # Concatène les courbes par année en colonnes alignées sur Jour_annee
    # Jz calcule la moyenne par jour sur les années
    if curves_ref:
        ref_curve_df = pd.concat([c.set_index("Jour_annee")["Resid"] for c in curves_ref], axis=1)
        ref_mean = ref_curve_df.mean(axis=1).reindex(range(1, 366)).astype(float)
        # Interpolation linéaire pour combler les jours manquants aux extrémités 
        ref_mean = ref_mean.interpolate(method="linear", limit_direction="both")

        # Trace la courbe moyenne de référence (qui correspond à la droite de regression)
        fig.add_trace(go.Scatter(
            x=np.arange(1, 366),
            y=ref_mean.values,
            mode="lines",
            name="Réf 2018-2019 (résidus stables)",
            line=dict(width=3, color="black")
        ))

    # Courbes années de comparaison
    for y in years_compare:
        part = d[d["Année"] == y][["Jour_annee","Resid"]]
        if part.empty: # si pas points de stables pour une année on saute
            continue
        cont = courbe_continue_par_jour(part, "Resid")
        fig.add_trace(go.Scatter(
            x=cont["Jour_annee"], y=cont["Resid"],
            mode="lines",
            name=str(y),
            line=dict(width=2, color=PALETTE.get(y, "gray"))
        ))

    # Bandes ±3σ
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=+3.5*sigma_e, line_dash="dash", line_color="deepskyblue",
                  annotation_text=f"+3σ", annotation_position="top left")
    fig.add_hline(y=-3.5*sigma_e, line_dash="dash", line_color="deepskyblue",
                  annotation_text=f"-3σ", annotation_position="bottom left")

    # Rend les lignes (spline) pour un aspect plus lissé
    fig.update_traces(line_shape="spline")
    fig.update_layout(
        title="Résidus temporels",
        xaxis_title="Temps",
        yaxis_title="Résidu e (µm)",
        template="plotly_white",
        hovermode="x unified"
    )
    # export HTML
    if out_dir is not None and export_html:
        # le chemin dusorti
        path_html = os.path.join(out_dir, "temporel_an_by_an.html")
        fig.write_html(path_html)
        print(f"[Résidus DOY] HTML sauvegardé: {path_html}")

    return fig


# Calcul difference temporelle continue
def temporel_continue(
    df,
    poly,
    sigma_e,
    stable_global_mask,
    years_order=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
    vmin=V_MIN,
    vmax=V_MAX,
    out_dir=None,
    export_html=True
):
    """
    Même méthode que 'années superposées' (médiane par jour + interpolation),
    mais je colle les années successivement sur un axe continu (toutes les années).
    Résidus e = Vib - f(Vit) calculés sur points stables.
    """
    # Sélectionne dans df uniquement les lignes marquées stables et les colonnes nécessaires.
    d = df.loc[stable_global_mask, ["Année","Jour_annee","Vitesse","Vibration"]].dropna().copy()
    # Filtre les points sur la plage de vitesse [vmin, vmax], cohérente avec le domaine
    d = d[(d["Vitesse"] >= vmin) & (d["Vitesse"] <= vmax)]
    if d.empty:
        raise ValueError("Aucun point stable exploitable (courbe continue).")

    
    # Résidu de la vibration : e = y_mesuré - y_prédit
    # Conversion en float pour s'assurer de types numériques propres.
    d["Resid"] = d["Vibration"].to_numpy(float) - poly(d["Vitesse"].to_numpy(float))

    pieces = [] # Il contient les segments "continu interpolé" pour chaque année
    for y in years_order:
        part = d[d["Année"] == y][["Jour_annee", "Resid"]].copy()
        if part.empty:
            continue

        # Transforme les points en courbe continue par jour
        cont = courbe_continue_par_jour(part, "Resid") # renvoie Jour_annee + Resid (interpolé)

        # courbe_continue_par_jour' renvoie un DF avec colonnes ["Jour_annee","Resid"] triées.
        cont["X_global"] = (y * 366) + cont["Jour_annee"].astype(int)
        cont["Année"] = y
        pieces.append(cont)

    if not pieces:
        # Si aucune année n'a pu être utilisé, s'arrêter proprement.
        raise ValueError("Aucune année exploitable pour la courbe continue.")

    allc = pd.concat(pieces, ignore_index=True)

    # Création de la figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=allc["X_global"],
        y=allc["Resid"],
        mode="lines",
        name="temporel continue",
        line=dict(width=2, color="black")
    ))

    # Bandes ±3σ (sigma ref)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=+3.5*sigma_e, line_dash="dash", line_color="deepskyblue",
                  annotation_text=f"+3σ", annotation_position="top left")
    fig.add_hline(y=-3.5*sigma_e, line_dash="dash", line_color="deepskyblue",
                  annotation_text=f"-3σ", annotation_position="bottom left")

    # Marqueur debut de chaque 
    for y in years_order:
        x_sep = y * 366 + 1
        fig.add_vline(x=x_sep, line_dash="dot", line_color="lightgray")

    # # Définition des ticks de l'axe X : un tick centré par année pour une lecture simple
    tickvals = [y * 366 + 183 for y in years_order] # milieu de l'année
    ticktext = [str(y) for y in years_order]

    # Mise en forme de la figure
    fig.update_layout(
        title="temporel continue",
        xaxis_title="Année",
        yaxis_title="Résidu e (µm)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    )

    # Export HTML
    if out_dir is not None and export_html:
        path_html = os.path.join(out_dir, "temporel_continue.html")
        fig.write_html(path_html)
        print(f"[Résidus collés] HTML sauvegardé: {path_html}")

    return fig


# Main
def main():
    chemin, df = lire_premier_excel(dossier)
    out_dir = os.path.dirname(chemin)

    # Stabilité sur toutes années
    start_timer("Stabilité toutes années")
    stable_vit_all = calcul_stabilite_toutes_annees(df, "Vitesse") if "Vitesse" in df.columns else np.zeros(len(df), bool)
    stable_vib_all = calcul_stabilite_toutes_annees(df, "Vibration") if "Vibration" in df.columns else np.zeros(len(df), bool)
    stable_global = stable_vit_all & stable_vib_all
    stop_timer("Stabilité toutes années")   

    # Référence stable 2018-2019 (pour apprendre poly + sigma)
    stable_ref = stable_global & df["Année"].isin(annee_ref)

    # Corrélation
    if "Vitesse" in df.columns and "Vibration" in df.columns:
        fig_corr = plot_correlation_by_year(df, stable_ref, deg=4)
        fig_corr.write_html(os.path.join(out_dir, "correlation_vitesse_vibration.html"))

    # Critères + histogrammes
    df_ref = filtrer_annees(df, annee_ref)
    for var in [c for c in ["Vitesse","Vibration"] if c in df.columns]:
        bins = plage_pente_vitesse if var == "Vitesse" else plage_pente_vibration
        s = SEUILS_PENTE[var]

        res, pct, agg = analyser_variable(df_ref, var, s["pmin"], s["pmax"], s["r2max"], bins)
        if res is None or res.empty:
            print(f"[Critères] Aucun segment exploitable pour {var}.")
            continue

        fig_scatter = plot_scatter_pente_R2(res, var, show_r2_threshold=True)
        fig_scatter.write_html(os.path.join(out_dir, f"criteres_stabilite_{var}.html"))

        fig_hist = plot_histogram_croise(res,var)
        fig_hist.write_html(os.path.join(out_dir, f"histogramme_croise_{var}.html"))

    # Domaine temporel vibration brute
    if "Vibration" in df.columns:
        stable_vib_ref = stable_vib_all & df["Année"].isin(annee_ref)
        fig_vib = supperposition_année(df, stable_vib_ref, seuil_min=8)
        #fig_vib.write_html(os.path.join(out_dir, "vibration_temporelle.html"))

    # Poly + sigma et domaines temporels résidus
    poly = None
    mu_e = None
    sigma_e = None

    if "Vitesse" in df.columns and "Vibration" in df.columns:
        try:
            poly, mu_e, sigma_e = calcul_sigma(
                df=df,
                stable_ref_mask=stable_ref, # sigma calculé sur référence stable
                years_ref=annee_ref,
                deg=4,
                vmin=V_MIN,
                vmax=V_MAX
            )
            print(f"[Sigma] mu_e={mu_e:.4f} µm, sigma_e={sigma_e:.4f} µm, 3σ={3.5*sigma_e:.4f} µm")
        except Exception as e:
            print(f"[Sigma] ERREUR: {e}")

    if poly is not None and sigma_e is not None:
        calcul_difference_supperposee(
            df=df,
            poly=poly,
            sigma_e=sigma_e,
            stable_global_mask=stable_global, # stabilité sur toutes années
            years_ref=annee_ref,
            years_compare=annees_compare,
            vmin=V_MIN,
            vmax=V_MAX,
            out_dir=out_dir,
            export_html=True
        )

        temporel_continue(
    df=df,
    poly=poly,
    sigma_e=sigma_e,
    stable_global_mask=stable_global,
    years_order=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
    vmin=V_MIN,
    vmax=V_MAX,
    out_dir=out_dir,
    export_html=True
    )

    # Temps d’exécution
    print("\n--- Temps d'exécution ---")
    for k, v in profiling.items():
        print(f"{k:35s}: {v:.2f} s")

if __name__ == "__main__":
    main()
