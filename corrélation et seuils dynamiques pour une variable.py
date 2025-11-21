# Analyse stabilité + Histogrammes + Corrélation + Temporel
# (réf 2018–2019 seulement pour les critères)

import os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Chronométrage 
profiling = {}
def start_timer(label): profiling[label] = time.time()
def stop_timer(label): profiling[label] = time.time() - profiling[label]

# Chemin
dossier = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\Rousssine - 0"

# Fenêtres glissantes
T_segment = 7200
pas_glissement = 1800

# Seuils stabilité pente & R²
SEUILS_PENTE = {
    "Vitesse": {"pmin": -0.4, "pmax": 0.4, "r2max": 600.0, "color": "blue"},
    "Vibration": {"pmin": -0.0005, "pmax": 0.0005, "r2max": 4.0, "color": "red"},
}

# Bins de pentes
plage_pente_vitesse = np.array([-np.inf, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, np.inf])
plage_pente_vibration = np.array([-np.inf, -0.0015, -0.001, -0.0005, 0, 0.0005, 0.001, 0.0015, np.inf])

# Années de référence
annee_ref = [2018, 2019]

# Lecture
def lire_premier_excel(dossier):
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not fichiers:
        raise FileNotFoundError("Aucun fichier Excel valide trouvé dans le dossier.")
    chemin = os.path.join(dossier, fichiers[0])

    start_timer("Lecture Excel")
    df = pd.read_excel(chemin, engine='openpyxl')

    # Colonnes propres
    df.columns = [c.strip() for c in df.columns]
    mapping = {'date':'Date','vitesse':'Vitesse','vibration':'Vibration'}
    df.columns = [mapping.get(c.lower(), c) for c in df.columns]

    # Types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Vitesse' in df.columns: df['Vitesse'] = pd.to_numeric(df['Vitesse'], errors='coerce')
    if 'Vibration' in df.columns: df['Vibration'] = pd.to_numeric(df['Vibration'], errors='coerce')

    # Nettoyage
    req = ['Date']
    if 'Vitesse' in df.columns: req.append('Vitesse')
    if 'Vibration' in df.columns: req.append('Vibration')
    df.dropna(subset=req, inplace=True)

    # Temps & année
    df['Secondes'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
    df['Année'] = df['Date'].dt.year
    df['Jour_annee'] = df['Date'].dt.dayofyear
    stop_timer("Lecture Excel")
    return chemin, df

def filtrer_annees(df, years):
    return df[df['Année'].isin(years)].copy()

# Analyse par fenêtres
def analyser_variable(df, value_col, pmin, pmax, r2max, bins_pente):
    secs = df['Secondes'].values
    y_all = df[value_col].values
    if len(secs) == 0:
        return pd.DataFrame(), None, None

    T_tot = secs[-1]
    out = []
    for t in range(0, int(T_tot - T_segment + 1), pas_glissement):
        m = (secs >= t) & (secs < t + T_segment)
        x = secs[m]; y = y_all[m]
        if len(x) < 12: continue

        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        R = y - (pente*x + intercept)
        R2 = np.std(R)
        etat = "Stable" if (pmin <= pente <= pmax) and (R2 <= r2max) else "Instable"
        out.append({"centre": t + T_segment/2.0, "pente": pente, "R2": R2, "etat": etat})

    res = pd.DataFrame(out)
    if res.empty: return res, None, None

    res['pente_bin'] = pd.cut(res['pente'], bins=bins_pente, include_lowest=True)

    pct = (res['etat'].value_counts(normalize=True)*100)\
            .reindex(['Stable','Instable']).fillna(0).reset_index()
    pct.columns = ['etat','pourcentage']

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

# Zones stables sur la référence
def calcul_stabilite_ref(df, var_name, annees=annee_ref):
    seuils = SEUILS_PENTE[var_name]
    pmin, pmax, r2max = seuils['pmin'], seuils['pmax'], seuils['r2max']
    mask_years = df['Année'].isin(annees)

    res_mask = np.zeros(len(df), dtype=bool)
    secs = df.loc[mask_years, 'Secondes'].values
    signal = df.loc[mask_years, var_name].values
    if len(secs) == 0: return res_mask

    T_loc = secs[-1]
    for t in range(0, int(T_loc - T_segment + 1), pas_glissement):
        idx = (secs >= t) & (secs < t + T_segment)
        x = secs[idx]; y = signal[idx]
        if len(x) < 12: continue

        A = np.vstack([x, np.ones_like(x)]).T
        pente, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        R2 = np.std(y - (pente*x + intercept))
        if (pmin <= pente <= pmax) and (R2 <= r2max):
            global_idx = df.index[mask_years][idx]
            res_mask[global_idx] = True
    return res_mask

# Graphiques critères de stabilité
def plot_scatter_pente_R2(res, var_name, show_r2_threshold=True):
    s = SEUILS_PENTE[var_name]; pmin, pmax, r2max = s['pmin'], s['pmax'], s['r2max']
    fig = go.Figure()
    st = res[res['etat']=="Stable"]; inst = res[res['etat']=="Instable"]
    if not st.empty:
        fig.add_trace(go.Scatter(x=st['R2'], y=st['pente'], mode='markers',
                                 name='Stable', marker=dict(color='green', size=6, opacity=0.7)))
    if not inst.empty:
        fig.add_trace(go.Scatter(x=inst['R2'], y=inst['pente'], mode='markers',
                                 name='Instable', marker=dict(color='red', size=6, opacity=0.7)))
    fig.add_hline(y=pmin, line_dash="dash", line_color="gray")
    fig.add_hline(y=pmax, line_dash="dash", line_color="gray")
    if show_r2_threshold: fig.add_vline(x=r2max, line_dash="dash", line_color="gray")

    top = f"(critères pente: {pmin} ≤ pente ≤ {pmax}" + (f", R² ≤ {r2max}" if show_r2_threshold else "") + ")"
    fig.update_layout(title=f"Diagramme Pente vs Résidu — {var_name} {top}",
                      xaxis_title="R² (résidus)", yaxis_title="Pente",
                      template="plotly_white", hovermode="closest")
    return fig

def plot_histogram_croise(res, var_name):
    """
    Histogramme croisé par classes de pente :
    - barres bleues = % de segments
    - barres violettes = % de 'masse de résidus' ΣR² par classe
    """
    # % segments par bin
    pct_segments = (res['pente_bin'].value_counts(normalize=True).sort_index() * 100)

    # % résidus par bin (normalisé à 100%)
    r2_sum_per_bin = res.groupby('pente_bin')['R2'].sum().reindex(pct_segments.index)
    total_r2 = r2_sum_per_bin.sum()
    pct_residus = (r2_sum_per_bin / total_r2 * 100) if total_r2 > 0 else r2_sum_per_bin*0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_segments.index], y=pct_segments.values,
        name="segments ", text=[f"{v:.1f}%" for v in pct_segments.values],
        textposition="outside", marker_color="steelblue"
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in pct_residus.index], y=pct_residus.values,
        name="résidus ", text=[f"{v:.1f}%" for v in pct_residus.values],
        textposition="outside", marker_color="orange"
    ))
    fig.update_layout(
        title=f"Histogramme — {var_name} ",
        xaxis_title="Classes de pente", yaxis_title="Pourcentage (%)",
        template="plotly_white", barmode="group"
    )
    return fig

# Correlation Vitesse–Vibration + polynomiale (réf stable)
def plot_correlation_by_year(df, stable_ref_mask):
    """Nuage Vitesse vs Vibration par année + polynomiale (ordre 4) sur la référence (2018–2019)."""
    fig = go.Figure()

    # Points stables 2018–2019
    ref_pts = df[stable_ref_mask]
    fig.add_trace(go.Scatter(
        x=ref_pts['Vitesse'], y=ref_pts['Vibration'],
        mode='markers', name="2018-2019 (stable)",
        marker=dict(color='green', size=3), opacity=0.7
    ))

    # Autres années
    couleurs = {2020:'red', 2021:'blue', 2022:'orange', 2023:'pink', 2024:'brown', 2025:'purple'}
    for annee in sorted(df['Année'].unique()):
        if annee in annee_ref: # on saute 2018-2019 vu qu'elle est déja representée
            continue
        subset = df[df['Année']==annee]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset['Vitesse'], y=subset['Vibration'],
            mode='markers', name=str(annee),
            marker=dict(color=couleurs.get(annee,'gray'), size=3), opacity=0.7
        ))

    # Régression polynomiale
    vmin, vmax = 6000, 12000
    zone = ref_pts[(ref_pts['Vitesse']>=vmin) & (ref_pts['Vitesse']<=vmax)]
    X = zone['Vitesse'].to_numpy(); Y = zone['Vibration'].to_numpy()
    if len(X) > 20:
        ok = (~np.isnan(X)) & (~np.isnan(Y))
        coeffs = np.polyfit(X[ok], Y[ok], 4)
        X_fit = np.linspace(X[ok].min(), X[ok].max(), 500)
        Y_fit = np.polyval(coeffs, X_fit)
        fig.add_trace(go.Scatter(
            x=X_fit, y=Y_fit, mode='lines',
            line=dict(color='darkred', width=3),
            name="Régression polynomiale (réf)"
        ))

    fig.update_layout(
        title="Corrélation : Vitesse & Vibration (réf = zones stables 2018-2019)",
        xaxis_title="Vitesse",
        yaxis_title="Vibration (µm)",
        template="plotly_white"
    )
    return fig

# Lissage
def continuous_curve(df_year, value_col, method='linear'):
    
    s = df_year.groupby('Jour_annee')[value_col].median()

    # Supprimer les valeurs <= 0
    s = s.where(s > 4000, np.nan)

    # Reindex sur 1..365
    s = s.reindex(range(1, 366))

    # Interpolation pour relier les sommets
    s_interp = s.interpolate(method=method, limit_direction='both')

    return s_interp.reset_index().rename(columns={'index': 'Jour_annee', 0: value_col})

def plot_timeseries_variable_superpose(df, var_name, stable_mask_for_var, years_ref=annee_ref):
    """
    Courbes continues par année (1..365) :
    - Réf 2018–2019 (uniquement zones stables) en vert (moyenne des deux ans)
    - Autres années superposées
    (0 et négatifs éliminés automatiquement)
    """
    fig = go.Figure()
    dfx = df[['Année','Jour_annee','Date',var_name]].copy()

    # référence (stables)
    ref_all = df[stable_mask_for_var & df['Année'].isin(years_ref)][['Année','Jour_annee',var_name]]
    curves = []
    for an in years_ref:
        part = ref_all[ref_all['Année']==an][['Jour_annee',var_name]]
        if part.empty: continue
        curves.append(continuous_curve(part, var_name))
    if curves:
        ref_curve = pd.concat([c.set_index('Jour_annee')[var_name] for c in curves], axis=1)
        ref_mean = ref_curve.mean(axis=1).reset_index().rename(columns={0:var_name})
        fig.add_trace(go.Scatter(
            x=ref_mean['Jour_annee'], y=ref_mean[var_name],
            mode='lines', name='2018-2019 (réf)',
            line=dict(color='green', width=3)
        ))

    # autres années
    palette = {2020:'red', 2021:'blue', 2022:'orange', 2023:'pink', 2024:'brown', 2025:'purple'}
    for an in sorted(dfx['Année'].unique()):
        if an in years_ref: continue
        part = dfx[dfx['Année']==an][['Jour_annee',var_name]].dropna()
        if part.empty: continue
        cont = continuous_curve(part, var_name)
        fig.add_trace(go.Scatter(
            x=cont['Jour_annee'], y=cont[var_name],
            mode='lines', name=str(an),
            line=dict(color=palette.get(an,'gray'), width=2)
        ))

    fig.update_traces(line_shape='spline')
    fig.update_layout(title=f"{var_name} — domaine temporel ",
                      xaxis_title="temps\an", yaxis_title=var_name,
                      template="plotly_white", hovermode="x unified")
    return fig

# Modèle Vib = f(Vitesse) (appris sur réf stable)
def fit_vitesse_of_vibration_on_ref(df, stable_ref_mask, deg=4, vib_min=None, vib_max=None):
    ref = df.loc[stable_ref_mask & df['Année'].isin(annee_ref), ['Vibration','Vitesse']].dropna()
    if vib_min is not None: ref = ref[ref['Vibration'] >= vib_min]
    if vib_max is not None: ref = ref[ref['Vibration'] <= vib_max]
    if len(ref) < 20: raise ValueError("Référence insuffisante pour Vitesse=f(Vibration).")

    X = ref['Vibration'].to_numpy(); Y = ref['Vitesse'].to_numpy()
    coeffs = np.polyfit(X, Y, deg)
    return np.poly1d(coeffs)

def vib_from_vitesse(df, poly_f):
    df = df.copy()
    df['Vib_from_vitesse'] = np.where(df['Vibration'].notna(),poly_f(df['Vibration']), np.nan)
    return df

# Main
def main():
    # 1) Lecture
    chemin, df = lire_premier_excel(dossier)
    out_dir = os.path.dirname(chemin)

    # Stabilité (réf 2018–2019) par variable
    start_timer("Détection zones stables 2018-2019")
    stable_vit = calcul_stabilite_ref(df, 'Vitesse') if 'Vitesse' in df.columns else np.zeros(len(df), bool)
    stable_vib = calcul_stabilite_ref(df, 'Vibration') if 'Vibration' in df.columns else np.zeros(len(df), bool)
    stable_ref = stable_vit & stable_vib
    stop_timer("Détection zones stables 2018-2019")

    # Corrélation + régression polynomiale (réf stable)
    fig_corr = plot_correlation_by_year(df, stable_ref)
    fig_corr.write_html(os.path.join(out_dir, "correlation_vitesse_vibration.html"))

    # Courbes temporelles continues (réf vs autres) : Vibration & Vitesse
    if 'Vibration' in df.columns:
        fig_vib = plot_timeseries_variable_superpose(df, 'Vibration', stable_vib)
        #fig_vib.write_html(os.path.join(out_dir, "vibration temporelle.html"))
    if 'Vitesse' in df.columns:
        fig_vit = plot_timeseries_variable_superpose(df, 'Vitesse', stable_vit)
        #fig_vit.write_html(os.path.join(out_dir, "vitesse temporelle.html"))

    #  Vitesse prédite depuis Vibration (V = f(Vib) appris sur réf stable)
    poly_v = fit_vitesse_of_vibration_on_ref(df, stable_ref, deg=4)
    df_model = vib_from_vitesse(df, poly_v)
    fig_pred = plot_timeseries_variable_superpose(df_model, 'Vib_from_vitesse', stable_ref)
    fig_pred.write_html(os.path.join(out_dir, "vibration en fonction de la vitesse.html"))

    # Critères & histogrammes (uniquement sur 2018–2019)
    df_ref = filtrer_annees(df, annee_ref)
    for var in [c for c in ['Vitesse','Vibration'] if c in df.columns]:
        bins = plage_pente_vitesse if var=='Vitesse' else plage_pente_vibration
        s = SEUILS_PENTE[var]
        res, pct, agg = analyser_variable(df_ref, var, s['pmin'], s['pmax'], s['r2max'], bins)
        if res is None or res.empty:
            print(f"Aucun segment exploitable pour {var}."); continue

        fig_scatter = plot_scatter_pente_R2(res, var, show_r2_threshold=True)
        fig_scatter.write_html(os.path.join(out_dir, f"criteres_stabilite_{var}.html"))

        fig_hist_cross = plot_histogram_croise(res, var)
        fig_hist_cross.write_html(os.path.join(out_dir, f"histogramme_croise_{var}.html"))


    # Temps d’exécution
    print("\n--- Temps d'exécution ---")
    for k, v in profiling.items(): print(f"{k:40s}: {v:.2f} sec")

if __name__ == "__main__":
    main()