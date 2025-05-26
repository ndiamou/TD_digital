import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser 

# Paramètres d’analyse
T_segment = 7200 # Durée d’un segment (en secondes)
pas_glissement = 600 # Décalage glissant (en secondes)

# Lecture d'un fichier Excel
def lire_fichier(chemin):
    df = pd.read_excel(chemin)
    df.columns = [col.strip() for col in df.columns]
    correspondance = {'date': 'Date', 'vitesse': 'Vitesse'}
    df.columns = [correspondance.get(col.lower(), col) for col in df.columns]
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Vitesse'] = pd.to_numeric(df['Vitesse'], errors='coerce')
    df.dropna(subset=['Date', 'Vitesse'], inplace=True)

    # Supprimer les lignes avec vitesse = 0 au début
    idx = df[df['Vitesse'] != 0].index.min()
    if pd.isna(idx):
        print("Toutes les vitesses sont nulles.")
        return None
    df = df.loc[idx:]
    df['Secondes'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
    return df

# Analyse d’un segment
def analyser_segment(df, debut, fin):
    segment = df[(df['Secondes'] >= debut) & (df['Secondes'] < fin)]
    if len(segment) < 12:
        return None, None

    x = segment['Secondes'].values
    y = segment['Vitesse'].values

    pente, intercept = np.polyfit(x, y, 1)
    y_pente = pente * x + intercept
    R = y - y_pente

    moyenne = sum(R) / len(R)
    variance = sum((x - moyenne)**2 for x in R) / len(R)
    ecart_type = variance ** 0.5 if variance > 0 else 1e-8
    asym = sum((x - moyenne) ** 3 for x in R) / (len(R)*ecart_type ** 3 if ecart_type != 0 else 0)
    plat = sum((x - moyenne) ** 4 for x in R) / (len(R)*ecart_type ** 4 if ecart_type != 0 else 0)

    stats = {
        'centre': (debut + fin) / 2,
        'moyenne': moyenne,
        'variance': variance,
        'asymetrie': asym,
        'applatissement': plat
    }

    return (x, y_pente), stats

# Normaliser les donnees
def normaliser(serie):
    min_val, max_val  = np.min(serie), np.max(serie)
    return (serie - min_val) / (max_val - min_val)  if max_val > min_val else serie * 0

# Traitement d’un seul fichier
def traiter_fichier(chemin):
    print(f"\nTraitement de : {os.path.basename(chemin)}")
    df = lire_fichier(chemin)
    if df is None:
        return

    T_total = df['Secondes'].iloc[-1]
    toutes_stats = []
    courbes_pentes = []

    t = 0 
    while t +T_segment <= T_total:
        regression, stats = analyser_segment(df, t, t + T_segment)
        if regression:
            courbes_pentes.append(regression)
        if stats:
            toutes_stats.append(stats)
        t += pas_glissement

    if not toutes_stats:
        print("aucune statisitiques calculée")
        return
    
    stats_df = pd.DataFrame(toutes_stats)

    # Normalisation
    df['Vitesse_norm'] = normaliser(df['Vitesse'])
    for col in ['moyenne', 'variance', 'asymetrie', 'applatissement']:
        stats_df[col] = normaliser(stats_df[col])
    
    # Création graphique Plotly
    fig = go.Figure()


    # Trace le graphe
    fig.add_trace(go.Scatter(x=df['Secondes'], y=df['Vitesse_norm'],name='Signal Brut', line=dict(color='black')))

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

# Traitement de tous les fichiers dans le dossie
fichiers = [f for f in os.listdir(dossier_parent) if f.endswith('.xlsx') and not f.startswith('~$')]
if fichiers:
    chemin_fichier = os.path.join(dossier_parent, fichiers[0])
    traiter_fichier(chemin_fichier)
else:
    print("Aucun fichier Excel trouvé.")