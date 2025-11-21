import os
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

# Chemin du dossier
dossier = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\Rousssine - 3"

# Fenêtres glissantes
T_segment = 7200
pas_glissement = 1800

# Seuils stabilité pente & R²
SEUILS_PENTE = {
    "Vitesse": {"pmin": -0.1, "pmax": 0.1, "r2max": 300.0, "color": "blue"},
    "Vibration": {"pmin": -0.0005, "pmax": 0.0005, "r2max": 4.0, "color": "red"},
    "Rpression": {"pmin": -0.00005, "pmax": 0.00005, "r2max": 0.05, "color": "purple"},
}

# Bins de pentes
plage_pente_vitesse = np.array([-np.inf, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, np.inf])
plage_pente_vibration = np.array([-np.inf, -0.0015,-0.001, -0.0005, 0, 0.0005,0.001,0.0015, np.inf])
plage_pente_rpression = np.array([-np.inf, -0.0004, -0.0003, -0.0002, -0.0001, 0, 0.0001, 0.0002, 0.0003, 0.0004, np.inf])

# Années de référence
annee_ref = [2018, 2019]

# Chronométrage
profiling = {}
def start_timer(label): profiling[label] = time.time()
def stop_timer(label): profiling[label] = time.time() - profiling[label]

# Lecture + préparation des données
def lire_premier_excel(dossier):
    fichiers = [f for f in os.listdir(dossier)
                if f.lower().endswith('.xlsx') and not f.startswith('~$')]
    if not fichiers:
        raise FileNotFoundError("Aucun fichier Excel valide trouvé dans le dossier.")
   
    chemin = os.path.join(dossier, fichiers[0])
    start_timer("Lecture Excel")
   
    df = pd.read_excel(chemin, engine='openpyxl')

    # Nettoyage colonnes
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        'date': 'Date',
        'vitesse': 'Vitesse',
        'vibration': 'Vibration',
        'rpression': 'Rpression'
    }
    df.columns = [mapping.get(c.lower(), c) for c in df.columns]

    # Conversion types
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Vitesse' in df.columns: df['Vitesse'] = pd.to_numeric(df['Vitesse'], errors='coerce')
    if 'Vibration' in df.columns: df['Vibration'] = pd.to_numeric(df['Vibration'], errors='coerce')
    if 'Rpression' in df.columns: df['Rpression'] = pd.to_numeric(df['Rpression'], errors='coerce')

    # Suppression NaN (on garde uniquement les lignes complètes)
    req = ['Date']
    if 'Vitesse' in df.columns: req.append('Vitesse')
    if 'Vibration' in df.columns: req.append('Vibration')
    if 'Rpression' in df.columns: req.append('Rpression')
    df.dropna(subset=req, inplace=True)

    # Colonnes temporelles
    df = df.sort_values('Date').reset_index(drop=True)
    df['Secondes'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
    df['Année'] = df['Date'].dt.year
    df['Jour_annee'] = df['Date'].dt.dayofyear

    stop_timer("Lecture Excel")

    return chemin, df

def filtrer_annees(df, years):
    return df[df['Année'].isin(years)].copy()

# Analyse par fenêtres glissantes (pente + R²)
def analyser_variable(df, value_col, pmin, pmax, r2max, bins_pente):
    secs = df['Secondes'].values
    y_all = df[value_col].values

    print(f"\nDébut analyse pour {value_col}")
    print("Nombre total de points :", len(secs))
    if len(secs) == 0:
        print("Aucun point dans la série temporelle.")
        return pd.DataFrame(), None, None

    T_tot = secs[-1]
    out = []
    for t in range(0, int(T_tot - T_segment + 1), pas_glissement):
        m = (secs >= t) & (secs < t + T_segment)
        x = secs[m]; y = y_all[m]
        if len(x) < 12:
            continue

        # Régression linéaire locale
        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        R = y - (pente*x + intercept)
        R2 = np.std(R)

        etat = "Stable" if (pmin <= pente <= pmax) and (R2 <= r2max) else "Instable"
        out.append({"centre": t + T_segment/2.0, "pente": pente, "R2": R2, "etat": etat})

    res = pd.DataFrame(out)
    if res.empty:
        print("Aucun segment analysé.")
        return res, None, None

    # Catégorisation des pentes
    res['pente_bin'] = pd.cut(res['pente'], bins=bins_pente, include_lowest=True)

    # Pourcentage par état
    pct = (res['etat'].value_counts(normalize=True)*100)\
            .reindex(['Stable','Instable']).fillna(0).reset_index()
    pct.columns = ['etat','pourcentage']

    # Agrégations
    agg = (res.groupby('etat')
            .agg(pente_min=('pente','min'), pente_max=('pente','max'),
                 R2_min=('R2','min'), R2_max=('R2','max'))
            .reset_index())
    glob = pd.DataFrame([{
        'etat':'Global',
        'pente_min':res['pente'].min(), 'pente_max':res['pente'].max(),
        'R2_min':res['R2'].min(), 'R2_max':res['R2'].max()
    }])
    agg = pd.concat([glob, agg], ignore_index=True)
    return res, pct, agg

# Masque de zones stables de référence (2018–2019)
def calcul_stabilite_ref(df, var_name, annees=annee_ref):
    seuils = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = seuils['pmin'], seuils['pmax'], seuils['r2max']
    mask_years = df['Année'].isin(annees)

    res_mask = np.zeros(len(df), dtype=bool)
    secs = df.loc[mask_years, 'Secondes'].values
    signal = df.loc[mask_years, var_name].values
    if len(secs) == 0:
        return res_mask

    T_loc = secs[-1]
    for t in range(0, int(T_loc - T_segment + 1), pas_glissement):
        idx = (secs >= t) & (secs < t + T_segment)
        x = secs[idx]; y = signal[idx]
        if len(x) < 12:
            continue

        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        R2 = np.std(y - (pente*x + intercept))
        if (pmin <= pente <= pmax) and (R2 <= r2max):
            global_idx = df.index[mask_years][idx]
            res_mask[global_idx] = True
    return res_mask

# Graphiques critères de stabilité (pente vs R²)
def plot_scatter_pente_R2(res, var_name, show_r2_threshold=True):
    s = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = s['pmin'], s['pmax'], s['r2max']
    fig = go.Figure()
    st = res[res['etat'] == "Stable"]
    inst = res[res['etat'] == "Instable"]

    print(f"\nCréation du scatter pour {var_name}")
    print("Segments stables :", len(st), "Segments instables :", len(inst))

    if not st.empty:
        fig.add_trace(go.Scatter(
            x=st['R2'], y=st['pente'], mode='markers',
            name='Stable', marker=dict(color='green', size=6, opacity=0.7)
        ))
    if not inst.empty:
        fig.add_trace(go.Scatter(
            x=inst['R2'], y=inst['pente'], mode='markers',
            name='Instable', marker=dict(color='red', size=6, opacity=0.7)
        ))

    fig.add_hline(y=pmin, line_dash="dash", line_color="gray")
    fig.add_hline(y=pmax, line_dash="dash", line_color="gray")
    if show_r2_threshold:
        fig.add_vline(x=r2max, line_dash="dash", line_color="gray")

    top = f"(critères pente: {pmin} ≤ pente ≤ {pmax}" + (f", R² ≤ {r2max}" if show_r2_threshold else "") + ")"
    fig.update_layout(
        title=f"Critère de stabilité Pente vs Résidu — {var_name} {top}",
        xaxis_title="R² (résidus)", yaxis_title="Pente",
        template="plotly_white", hovermode="closest"
    )
    return fig

# Histogramme croisé pentes / résidus
def plot_histogram_croise(res, var_name):
    print(f"\nCréation de l'histogramme pour {var_name}")
    # % segments par bin
    pct_segments = (res['pente_bin'].value_counts(normalize=True).sort_index() * 100)

    # % résidus par bin
    r2_sum_per_bin = res.groupby('pente_bin')['R2'].sum().reindex(pct_segments.index)
    total_r2 = r2_sum_per_bin.sum()
    pct_residus = (r2_sum_per_bin / total_r2 * 100) if total_r2 > 0 else r2_sum_per_bin * 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_segments.index], y=pct_segments.values,
        name="segments", text=[f"{v:.1f}%" for v in pct_segments.values],
        textposition="outside", marker_color="steelblue"
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_residus.index], y=pct_residus.values,
        name="résidus", text=[f"{v:.1f}%" for v in pct_residus.values],
        textposition="outside", marker_color="orange"
    ))
    fig.update_layout(
        title=f"Histogramme — {var_name}",
        xaxis_title="Classes de pente", yaxis_title="Pourcentage (%)",
        template="plotly_white", barmode="group"
    )
    return fig

# Corrélation 3D : Vitesse (X), Vibration (Y), Rpression (Z)
def plot_correlation(df, stable_ref_mask):
    """
    Corrélation 3D :
      - X = Vitesse
      - Y = Vibration
      - Z = Rpression
    On met en évidence les points de référence stables (2018–2019).
    Pas de polynôme 2D d’ordre 4 ici (impraticable), seulement le nuage 3D.
    """
    fig = go.Figure()

    # Points stables de référence (2018–2019)
    ref_pts = df[stable_ref_mask].copy()
    if not ref_pts.empty:
        fig.add_trace(go.Scatter3d(
            x=ref_pts['Vitesse'], y=ref_pts['Vibration'], z=ref_pts['Rpression'],
            mode='markers', name="2018-2019 (stable)",
            marker=dict(color='green', size=3), opacity=0.8
        ))

    # Autres années
    couleurs = {2020:'red', 2021:'blue', 2022:'orange', 2023:'yellow', 2024:'brown', 2025:'purple'}
    for annee in sorted(df['Année'].unique()):
        if annee in annee_ref:
            continue # dejà représentées via ref_pts
        subset = df[df['Année'] == annee]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=subset['Vitesse'], y=subset['Vibration'], z=subset['Rpression'],
            mode='markers', name=str(annee),
            marker=dict(color=couleurs.get(annee, 'gray'), size=3), opacity=0.6
        ))

    fig.update_layout(
        title="Corrélation : Vitesse, Vibration, Rpression",
        scene=dict(
            xaxis_title="Vitesse",
            yaxis_title="Vibration (µm)",
            zaxis_title="Rpression"
        ),
        template="plotly_white"
    )
    return fig

# Main
def main():
    # Lecture
    chemin, df = lire_premier_excel(dossier)
    out_dir = os.path.dirname(chemin)

    # Stabilité (réf 2018–2019) par variable
    start_timer("Détection zones stables 2018-2019")
    stable_vit = calcul_stabilite_ref(df, 'vitesse') if 'Vitesse' in df.columns else np.zeros(len(df), bool)
    stable_vib = calcul_stabilite_ref(df, 'vibration') if 'Vibration' in df.columns else np.zeros(len(df), bool)
    stable_rp = calcul_stabilite_ref(df, 'Rpression') if 'Rpression' in df.columns else np.zeros(len(df), bool)

    # Zone de référence "stable" = les 3 variables stables en même temps
    stable_ref = stable_vit & stable_vib & stable_rp
    stop_timer("Détection zones stables 2018-2019")

    # Corrélation de (Vib = f(Vitesse, Rpression))
    fig_corr3d = plot_correlation(df, stable_ref)
    fig_corr3d.write_html(os.path.join(out_dir, "correlation_vitesse_vibration_rpression.html"))

    # Critères & histogrammes (uniquement sur 2018–2019)
    df_ref = filtrer_annees(df, annee_ref)
    for var in [c for c in ['Vitesse','Vibration','Rpression'] if c in df.columns]:
        if var == 'Vitesse':
            bins = plage_pente_vitesse
        elif var == 'Vibration':
            bins = plage_pente_vibration
        else: # Rpression
            bins = plage_pente_rpression

        s = SEUILS_PENTE[var]
        res, pct, agg = analyser_variable(df_ref, var, s['pmin'], s['pmax'], s['r2max'], bins)
        if res is None or res.empty:
            print(f"Aucun segment exploitable pour {var}.")
            continue

        fig_scatter = plot_scatter_pente_R2(res, var, show_r2_threshold=True)
        fig_scatter.write_html(os.path.join(out_dir, f"criteres_stabilite_{var}.html"))

        fig_hist_cross = plot_histogram_croise(res, var)
        fig_hist_cross.write_html(os.path.join(out_dir, f"histogramme_croise_{var}.html"))

    # Temps d’exécution du script
    print("\n--- Temps d'exécution ---")
    for k, v in profiling.items():
        print(f"{k:40s}: {v:.2f} sec")

    if __name__== "__main__":
        main()