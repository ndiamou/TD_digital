import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser

# Paramètres d'analyse
T_segment = 7200 # Durée d’un segment (en secondes)
pas_glissement = 600 # Décalage entre les segments (en secondes)

# Lecture du fichier Excel
def lire_fichier(chemin):
    df = pd.read_excel(chemin)
    df.columns = [col.strip() for col in df.columns]
    correspondance = {'date': 'Date', 'vitesse': 'Vitesse'}
    df.columns = [correspondance.get(col.lower(), col) for col in df.columns]

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Vitesse'] = pd.to_numeric(df['Vitesse'], errors='coerce')
    df.dropna(subset=['Date', 'Vitesse'], inplace=True)
    df['Secondes'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
    return df

# Analyse d'un segment
def analyser_segment(df, debut, fin):
    """Analyse un segment : calcule la pente et les statistiques des résidus."""

    # Extraction des données du segment
    segment = df[(df['Secondes'] >= debut) & (df['Secondes'] < fin)]
    if len(segment) < 12:
        return None, None

    # Régression linéaire (calcul de la pente)
    x = segment['Secondes'].values
    y = segment['Vitesse'].values
    pente, intercept = np.polyfit(x, y, 1)
    y_pente = pente * x + intercept

    # Résidus entre le signal réel et la droite de régression
    R = y - y_pente
    moyenne = sum(R) / len(R)
    variance = sum((r - moyenne)**2 for r in R) / len(R)
    ecart_type = variance**0.5 if variance > 0 else 1e-8
    asym = sum((r - moyenne)**3 for r in R) / (len(R) * ecart_type**3)
    plat = sum((r - moyenne)**4 for r in R) / (len(R) * ecart_type**4)

    # Détection de la stabilité
    if (variance/7500 <= 75) and (-4 <= asym <= 4) and (plat <= 20):
        etat = 'Stable'
    else:
        etat = 'Instable'

    # Debug print facultatif :
    print(f"{debut/3600:.1f}h à {fin/3600:.1f}h | Var={variance:.2f} | Asym={asym:.2f} | Plat={plat:.2f} -> {etat}")

    stats = {
        'centre': (debut + fin) / 2,
        'moyenne': moyenne,
        'variance': variance/7500,
        'asymetrie': asym,
        'applatissement': plat,
        'etat': etat
    }

    return (x, y_pente), stats
# Traitement du fichier complet
def traiter_fichier(chemin):
    print(f"\n--- Traitement de : {os.path.basename(chemin)} ---")
    df = lire_fichier(chemin)
    if df is None:
        return

    T_total = df['Secondes'].iloc[-1]
    toutes_stats = []
    pentes = []

    t = 0
    while t + T_segment <= T_total:
        regression, stats = analyser_segment(df, t, t + T_segment)
        if regression:
            pentes.append(regression)
        if stats:
            toutes_stats.append(stats)
        t += pas_glissement

    stats_df = pd.DataFrame(toutes_stats)
    if stats_df.empty:
        print("Aucune statistique calculée.")
        return

    # Création du graphe Plotly
    fig = go.Figure()

    # Signal brut
    fig.add_trace(go.Scatter(x=df['Secondes'], y=df['Vitesse'],
                             name='Signal Brut', line=dict(color='black')))

    # Statistiques
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['moyenne'], name='Moyenne', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['variance'], name='Variance', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['asymetrie'], name='Asymétrie', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['applatissement'], name='Aplatissement', line=dict(color='orange')))

    # Stabilité : n'affiche que les points instables ( le plus simple )
    instables = stats_df[stats_df['etat'] == 'Instable']

    if not instables.empty:
        fig.add_trace(go.Scatter(
        x=instables['centre'],
        y=[1.05 * df['Vitesse'].max()] * len(instables),
        mode='markers',
        marker=dict(color='red', size=10),
        text=instables['etat'],
        textposition='top center',
        name="Instabilités"
    ))
    

    # Mise en page
    fig.update_layout(
        title=f"Analyse interactive du signal - {os.path.basename(chemin)}",
        xaxis_title="Temps (s)",
        yaxis_title="Valeurs / Résidus",
        template="plotly_white",
        hovermode="x unified"
    )

    # Enregistrement du graphe
    output_path = os.path.join(os.path.dirname(chemin), "graphique_interactif.html")
    fig.write_html(output_path)
    webbrowser.open(output_path)

# Lancer le traitement
dossier_parent = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\Rousssine - 0"
fichiers = [f for f in os.listdir(dossier_parent) if f.endswith('.xlsx') and not f.startswith('~$')]
if fichiers:
    chemin_fichier = os.path.join(dossier_parent, fichiers[0])
    traiter_fichier(chemin_fichier)
else:
    print("Aucun fichier Excel trouvé.")