import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser

# Paramètres d'analyse
T_segment = 7200 # Durée d’un segment de pente (120 min)
pas_glissement = 600 # Décalage entre les pentes (10 min)

# Lecture du fichier
def lire_fichier(chemin):
    df = pd.read_excel(chemin)

    # Nettoyage des noms de colonnes
    df.columns = [col.strip() for col in df.columns]
    correspondance = {'date': 'Date', 'vitesse': 'Vitesse'}
    df.columns = [correspondance.get(col.lower(), col) for col in df.columns]

    # Conversion des colonnes
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Vitesse'] = pd.to_numeric(df['Vitesse'], errors='coerce')
    df.dropna(subset=['Date', 'Vitesse'], inplace=True)

    df['Secondes'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
    return df

# Analyse par segment + calcul stats sur les résidus
def analyser_segment(df, debut, fin):
    """Applique une régression linéaire sur un segment et calcule les stats des écarts"""
    segment = df[(df['Secondes'] >= debut) & (df['Secondes'] < fin)]
    if len(segment) < 12: #pour qu'il ait minimum 12 point
        return None, None

    x = segment['Secondes'].values
    y = segment['Vitesse'].values

    # Régression linéaire (pente)
    pente, intercept = np.polyfit(x, y, 1)
    y_pente = pente * x + intercept

    # Calcul des résidus
    segment = segment.copy()
    segment['ecart'] = y - y_pente
    R = segment['ecart'].values

    moyenne = sum(R) / len(R)
    variance = sum((x - moyenne) ** 2 for x in R) / len(R)
    ecart_type = variance ** 0.5 if variance > 0 else 1e-8 # éviter division par zéro
    asymetrie = sum((x - moyenne) ** 3 for x in R) / (len(R) * ecart_type ** 3)
    aplatissement = sum((x - moyenne) ** 4 for x in R) / (len(R) * ecart_type ** 4)

    stats = {
        'centre': (debut + fin) / 2,
        'moyenne': moyenne,
        'variance': variance/100,
        'asymetrie': asymetrie,
        'applatissement': aplatissement
    }

    return (x, y_pente), stats

# Traitement complet du fichier
def traiter_fichier(chemin):
    print(f"\n--- Traitement de : {os.path.basename(chemin)} ---")
    df = lire_fichier(chemin)
    if df is None:
        return

    T_total = df['Secondes'].iloc[-1]
    toutes_stats = []
    pentes = []

    # Pentes glissantes + stats
    t = 0
    while t + T_segment <= T_total:
        regression, stats = analyser_segment(df, t, t + T_segment)
        if regression:
            pentes.append(regression)        
        if stats:
            toutes_stats.append(stats)
        t += pas_glissement

    # Affichage des stats
    stats_df = pd.DataFrame(toutes_stats)
    if stats_df.empty:
        print("Aucune statistique calculée.")
        return

  
    # Création graphique Plotly
    fig = go.Figure()

    # Signal brut
    fig.add_trace(go.Scatter(x=df['Secondes'], y=df['Vitesse'],name='Signal Brut', line=dict(color='black')))

    # Tracé des pentes successives
    for x_reg, y_reg in pentes:
        """fig.add_trace(go.Scatter(x=x_reg, y=y_reg,mode='lines', line=dict(color='red'),showlegend=False))"""

    # Statistiques (ordre 1 à 4)
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['moyenne'], name='Moyenne', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['variance'], name='Variance', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['asymetrie'], name='Asymétrie', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=stats_df['centre'], y=stats_df['applatissement'], name='Aplatissement', line=dict(color='orange')))

    # Mise en page
    fig.update_layout(
        title=f"Analyse interactive du signal - {os.path.basename(chemin)}",
        xaxis_title="Temps (s)",
        yaxis_title="Valeurs / Résidus",
        template="plotly_white",
        hovermode="x unified"
    )

    # Enregistrement + ouverture automatique
    output_path = os.path.join(os.path.dirname(chemin), "graphique_interactif.html")
    fig.write_html(output_path)
    webbrowser.open(output_path)

# chemin vers le dossier contenant le fichier Excel
dossier_parent = r"G:\_NPI\00-Digital\Alternant Mountakha Ndiaye\Stats documents\Rousssine - 1"


fichiers = [f for f in os.listdir(dossier_parent) if f.endswith('.xlsx') and not f.startswith('~$')]
if fichiers:
    chemin_fichier = os.path.join(dossier_parent, fichiers[0])
    traiter_fichier(chemin_fichier)
else:
    print("Aucun fichier Excel trouvé.")