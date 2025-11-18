import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Configuration de la page
st.set_page_config(
    page_title="Analyse Loi de Zipf - Gisements d'Or",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    .stApp {
        background: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f4ff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f0ff 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #667eea;
        margin: 10px 0;
    }
    .existing-badge {
        background-color: #48bb78;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .predicted-badge {
        background-color: #ed8936;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# Titre et en-tête
st.title("📊 Analyse Loi de Zipf - Gisements d'Or")
st.markdown("""
    ### Analyse statistique avancée avec test de Kolmogorov-Smirnov et prédictions
    **Auteurs:** Didier Ouedraogo, PGeo | Koulou Danshoko, Geo
""")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Séparateur
    separator = st.selectbox(
        "Séparateur de colonnes",
        [",", ";", "\t", "|"],
        format_func=lambda x: {",": "Virgule (,)", ";": "Point-virgule (;)", "\t": "Tabulation", "|": "Pipe (|)"}[x]
    )
    
    st.divider()
    
    # Options d'affichage
    st.subheader("📊 Affichage")
    show_log_scale = st.checkbox("Échelle logarithmique", value=True)
    show_grid = st.checkbox("Afficher la grille", value=True)
    
    st.divider()
    
    # Paramètres de prédiction
    st.subheader("🔮 Prédictions")
    predict_count = st.number_input(
        "Nombre de gisements à prédire",
        min_value=1,
        max_value=50,
        value=3,
        step=1
    )
    
    cutoff_value = st.number_input(
        "Cutoff minimum (Oz)",
        min_value=0,
        value=10000000,
        step=100000,
        format="%d"
    )

# Fonction pour charger les données exemple
def load_example_data():
    example_data = """Mine Muruntau,15000000
Mine Grasberg,12000000
Mine Olimpiada,10000000
Mine Pueblo Viejo,9000000
Mine Lihir,8500000
Mine Goldstrike,8000000
Mine Cortez,7500000
Mine Yanacocha,7000000
Mine Veladero,6500000
Mine Boddington,6000000
Mine Cadia,5500000
Mine Detour Lake,5000000
Mine Mponeng,4500000
Mine Kibali,4000000
Mine Loulo-Gounkoto,3800000
Mine Tropicana,3500000
Mine Fekola,3200000
Mine Bulyanhulu,3000000
Mine Kumtor,2800000
Mine Geita,2500000"""
    return example_data

# Zone de saisie des données
st.header("📥 Données des Gisements")

col1, col2 = st.columns([3, 1])

with col1:
    data_input = st.text_area(
        "Entrez les données (format: Nom, Taille en Oz)",
        value=load_example_data(),
        height=300,
        help="Une ligne par gisement, séparée par le séparateur choisi"
    )

with col2:
    st.markdown("### Actions")
    if st.button("🗑️ Effacer tout", use_container_width=True):
        data_input = ""
        st.rerun()
    
    if st.button("📂 Charger exemple", use_container_width=True):
        data_input = load_example_data()
        st.rerun()

# Fonctions d'analyse
def parse_data(data_str, sep):
    """Parse les données textuelles en DataFrame"""
    try:
        lines = [line.strip() for line in data_str.strip().split('\n') if line.strip()]
        deposits = []
        
        for line in lines:
            parts = line.split(sep)
            if len(parts) >= 2:
                name = parts[0].strip()
                size = float(parts[1].strip().replace(',', ''))
                if size > 0:
                    deposits.append({'name': name, 'size': size})
        
        if len(deposits) < 3:
            return None
        
        df = pd.DataFrame(deposits)
        df = df.sort_values('size', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du parsing: {e}")
        return None

def perform_zipf_analysis(df):
    """Effectue l'analyse de Zipf"""
    ranks = df['rank'].values
    sizes = df['size'].values
    
    # Transformation logarithmique
    log_ranks = np.log(ranks)
    log_sizes = np.log(sizes)
    
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_sizes)
    
    alpha = -slope
    C = np.exp(intercept)
    r_squared = r_value ** 2
    
    # Test de Kolmogorov-Smirnov
    n = len(sizes)
    xmin = sizes.min()
    
    max_diff = 0
    for i, size in enumerate(sorted(sizes, reverse=True)):
        empirical_cdf = (i + 1) / n
        theoretical_cdf = 1 - (size / xmin) ** (-alpha)
        diff = abs(empirical_cdf - theoretical_cdf)
        max_diff = max(max_diff, diff)
    
    critical_value = 1.36 / np.sqrt(n)
    p_value_ks = np.exp(-2 * n * max_diff ** 2)
    accept_ks = max_diff < critical_value
    
    return {
        'alpha': alpha,
        'C': C,
        'r_squared': r_squared,
        'ks_statistic': max_diff,
        'ks_critical': critical_value,
        'ks_pvalue': p_value_ks,
        'ks_accept': accept_ks,
        'slope': slope,
        'intercept': intercept
    }

def generate_predictions(df, analysis, count, cutoff):
    """Génère les prédictions de gisements"""
    alpha = analysis['alpha']
    C = analysis['C']
    
    predicted_deposits = []
    current_rank = 1
    generated_count = 0
    
    while generated_count < count and current_rank < 10000:
        size = C / (current_rank ** alpha)
        
        if size >= cutoff:
            predicted_deposits.append({
                'name': f'Prédit {generated_count + 1}',
                'size': size,
                'type': 'predicted',
                'original_rank': current_rank
            })
            generated_count += 1
        
        current_rank += 1
    
    if generated_count < count:
        return None, "Impossible de trouver assez de gisements au-dessus du cutoff"
    
    # Combiner existants et prédits
    existing = df.copy()
    existing['type'] = 'existing'
    existing['original_rank'] = existing['rank']
    
    predicted_df = pd.DataFrame(predicted_deposits)
    
    combined = pd.concat([
        existing[['name', 'size', 'type', 'original_rank']],
        predicted_df
    ], ignore_index=True)
    
    combined = combined.sort_values('size', ascending=False).reset_index(drop=True)
    combined['new_rank'] = combined.index + 1
    
    return combined, None

# Fonction pour créer un graphique Zipf avec Plotly
def create_zipf_plot(df, analysis, show_log=True):
    """Crée le graphique de Zipf"""
    ranks = df['rank'].values
    sizes = df['size'].values
    alpha = analysis['alpha']
    C = analysis['C']
    
    predicted_sizes = C / (ranks ** alpha)
    
    fig = go.Figure()
    
    # Données observées
    fig.add_trace(go.Scatter(
        x=ranks,
        y=sizes,
        mode='markers',
        name='Données Observées',
        marker=dict(size=10, color='#667eea', line=dict(width=2, color='white')),
        hovertemplate='<b>%{text}</b><br>Rang: %{x}<br>Taille: %{y:,.0f} Oz<extra></extra>',
        text=df['name']
    ))
    
    # Modèle de Zipf
    fig.add_trace(go.Scatter(
        x=ranks,
        y=predicted_sizes,
        mode='lines',
        name='Modèle de Zipf',
        line=dict(color='#ed64a6', width=3, dash='dash'),
        hovertemplate='Rang: %{x}<br>Taille prédite: %{y:,.0f} Oz<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Loi de Zipf (α = {alpha:.4f}, R² = {analysis["r_squared"]:.4f})',
        xaxis_title='Rang',
        yaxis_title='Taille (Oz)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    
    if show_log:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    return fig

def create_distribution_plot(df):
    """Crée l'histogramme de distribution"""
    fig = px.bar(
        df,
        x='name',
        y='size',
        title='Distribution des Gisements par Taille',
        labels={'size': 'Taille (M Oz)', 'name': 'Gisement'},
        color='size',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Taille: %{y:,.0f} Oz<extra></extra>'
    )
    
    # Convertir en millions d'onces
    fig.update_yaxes(title='Taille (M Oz)')
    fig.data[0].y = df['size'] / 1e6
    
    return fig

def create_combined_plot(combined_df, analysis, cutoff, show_log=True):
    """Crée le graphique combiné avec prédictions"""
    fig = go.Figure()
    
    # Gisements existants
    existing = combined_df[combined_df['type'] == 'existing']
    fig.add_trace(go.Scatter(
        x=existing['new_rank'],
        y=existing['size'],
        mode='markers',
        name='Gisements Existants',
        marker=dict(size=10, color='#48bb78', symbol='circle'),
        hovertemplate='<b>%{text}</b><br>Nouveau rang: %{x}<br>Taille: %{y:,.0f} Oz<extra></extra>',
        text=existing['name']
    ))
    
    # Gisements prédits
    predicted = combined_df[combined_df['type'] == 'predicted']
    fig.add_trace(go.Scatter(
        x=predicted['new_rank'],
        y=predicted['size'],
        mode='markers',
        name='Gisements Prédits',
        marker=dict(size=12, color='#ed8936', symbol='triangle-up'),
        hovertemplate='<b>%{text}</b><br>Nouveau rang: %{x}<br>Taille: %{y:,.0f} Oz<extra></extra>',
        text=predicted['name']
    ))
    
    # Modèle de Zipf
    max_rank = combined_df['new_rank'].max()
    ranks_line = np.arange(1, max_rank + 1)
    zipf_line = analysis['C'] / (ranks_line ** analysis['alpha'])
    
    fig.add_trace(go.Scatter(
        x=ranks_line,
        y=zipf_line,
        mode='lines',
        name='Modèle de Zipf',
        line=dict(color='#ed64a6', width=2, dash='dash')
    ))
    
    # Ligne de cutoff
    fig.add_trace(go.Scatter(
        x=[1, max_rank],
        y=[cutoff, cutoff],
        mode='lines',
        name=f'Cutoff ({cutoff/1e6:.2f} M Oz)',
        line=dict(color='#f56565', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='Distribution Reclassée - Échelle Log-Log',
        xaxis_title='Nouveau Rang',
        yaxis_title='Taille (Oz)',
        hovermode='closest',
        template='plotly_white',
        height=600
    )
    
    if show_log:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    return fig

def create_reranked_histogram(combined_df):
    """Crée l'histogramme de la distribution reclassée"""
    fig = go.Figure()
    
    # Créer deux barres séparées pour existants et prédits
    existing = combined_df[combined_df['type'] == 'existing']
    predicted = combined_df[combined_df['type'] == 'predicted']
    
    fig.add_trace(go.Bar(
        x=existing['name'],
        y=existing['size'] / 1e6,
        name='Gisements Existants',
        marker_color='#48bb78',
        hovertemplate='<b>%{x}</b><br>Taille: %{y:.2f} M Oz<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=predicted['name'],
        y=predicted['size'] / 1e6,
        name='Gisements Prédits',
        marker_color='#ed8936',
        hovertemplate='<b>%{x}</b><br>Taille: %{y:.2f} M Oz<extra></extra>'
    ))
    
    fig.update_layout(
        title='Distribution Reclassée - Histogramme',
        xaxis_title='Gisement',
        yaxis_title='Taille (M Oz)',
        xaxis_tickangle=-45,
        height=500,
        barmode='group',
        template='plotly_white'
    )
    
    return fig

# Fonctions d'export
def df_to_excel(dataframes_dict):
    """Convertit plusieurs DataFrames en fichier Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

# Analyse des données
if data_input:
    df = parse_data(data_input, separator)
    
    if df is not None and len(df) >= 3:
        # Effectuer l'analyse
        analysis = perform_zipf_analysis(df)
        
        # Affichage des résultats
        st.header("📈 Résultats de l'Analyse")
        
        # Onglets
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistiques", "📉 Graphiques", "🔮 Prédictions", "📋 Tableau"])
        
        with tab1:
            st.subheader("Statistiques Descriptives")
            
            # Métriques en colonnes
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Exposant α</div>
                        <div class="metric-value">{analysis['alpha']:.4f}</div>
                        <div class="metric-label">Loi de Zipf</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Coefficient R²</div>
                        <div class="metric-value">{analysis['r_squared']:.4f}</div>
                        <div class="metric-label">Qualité d'ajustement</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                ks_status = "✓ Accepté" if analysis['ks_accept'] else "✗ Rejeté"
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Test KS</div>
                        <div class="metric-value">{ks_status}</div>
                        <div class="metric-label">p = {analysis['ks_pvalue']:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Gisements</div>
                        <div class="metric-value">{len(df)}</div>
                        <div class="metric-label">Échantillons</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Taille Maximale", f"{df['size'].max()/1e6:.2f} M Oz")
            
            with col2:
                st.metric("Taille Moyenne", f"{df['size'].mean()/1e6:.2f} M Oz")
            
            with col3:
                st.metric("Ressources Totales", f"{df['size'].sum()/1e6:.2f} M Oz")
            
            with col4:
                st.metric("Constante C", f"{analysis['C']/1e6:.2f} M Oz")
        
        with tab2:
            st.subheader("Graphiques d'Analyse")
            
            # Graphique de Zipf
            zipf_fig = create_zipf_plot(df, analysis, show_log_scale)
            st.plotly_chart(zipf_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                zipf_html = zipf_fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📥 Télécharger Zipf (HTML)",
                    data=zipf_html,
                    file_name="graphique_zipf.html",
                    mime="text/html"
                )
            
            st.divider()
            
            # Graphique de distribution
            dist_fig = create_distribution_plot(df)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                dist_html = dist_fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📥 Télécharger Distribution (HTML)",
                    data=dist_html,
                    file_name="distribution_gisements.html",
                    mime="text/html"
                )
        
        with tab3:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("🔮 Prédiction des Gisements Non Découverts")
            
            st.info(f"""
                **Mode de prédiction :** Utilisation du modèle de Zipf pour prédire {predict_count} gisements 
                au-dessus d'un cutoff de {cutoff_value/1e6:.2f} M Oz. Les rangs seront automatiquement 
                recalculés en intégrant les gisements prédits dans la distribution existante.
            """)
            
            if st.button("🎯 Générer Prédictions avec Reclassement", use_container_width=True):
                combined_df, error = generate_predictions(df, analysis, predict_count, cutoff_value)
                
                if error:
                    st.error(error)
                else:
                    st.session_state['combined_df'] = combined_df
                    st.success(f"{predict_count} gisements prédits. Distribution reclassée avec nouveaux rangs.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Afficher les résultats de prédiction si disponibles
            if 'combined_df' in st.session_state:
                combined_df = st.session_state['combined_df']
                
                # Statistiques de prédiction
                st.divider()
                st.subheader("📊 Statistiques des Prédictions")
                
                existing_above = combined_df[(combined_df['type'] == 'existing') & (combined_df['size'] >= cutoff_value)]
                predicted = combined_df[combined_df['type'] == 'predicted']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Gisements Existants ≥ Cutoff", len(existing_above))
                    st.metric("Ressources Existantes", f"{existing_above['size'].sum()/1e6:.2f} M Oz")
                
                with col2:
                    st.metric("Gisements Prédits", len(predicted))
                    st.metric("Ressources Prédites", f"{predicted['size'].sum()/1e6:.2f} M Oz")
                
                with col3:
                    st.metric("Total Gisements ≥ Cutoff", len(existing_above) + len(predicted))
                    st.metric("Ressources Totales ≥ Cutoff", 
                             f"{(existing_above['size'].sum() + predicted['size'].sum())/1e6:.2f} M Oz")
                
                # Graphiques combinés
                st.divider()
                st.subheader("📉 Visualisations de la Distribution Reclassée")
                
                # Graphique log-log
                combined_fig = create_combined_plot(combined_df, analysis, cutoff_value, show_log_scale)
                st.plotly_chart(combined_fig, use_container_width=True)
                
                combined_html = combined_fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📥 Télécharger Distribution Reclassée Log-Log (HTML)",
                    data=combined_html,
                    file_name="distribution_reclassee_loglog.html",
                    mime="text/html"
                )
                
                # Histogramme
                hist_fig = create_reranked_histogram(combined_df)
                st.plotly_chart(hist_fig, use_container_width=True)
                
                hist_html = hist_fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📥 Télécharger Distribution Reclassée Histogramme (HTML)",
                    data=hist_html,
                    file_name="distribution_reclassee_histogramme.html",
                    mime="text/html"
                )
                
                # Tableau reclassé
                st.divider()
                st.subheader("📋 Tableau Complet avec Nouveaux Rangs")
                
                # Préparer le tableau d'affichage
                display_df = combined_df.copy()
                display_df['Nouveau Rang'] = display_df['new_rank']
                display_df['Type'] = display_df['type'].map({'existing': '🟢 Existant', 'predicted': '🟠 Prédit'})
                display_df['Nom'] = display_df['name']
                display_df['Taille (M Oz)'] = (display_df['size'] / 1e6).round(2)
                display_df['Rang Original'] = display_df['original_rank']
                
                # Calculer le changement de rang
                def calc_change(row):
                    if row['type'] == 'existing':
                        change = row['original_rank'] - row['new_rank']
                        if change == 0:
                            return '='
                        elif change > 0:
                            return f'▲ {change}'
                        else:
                            return f'▼ {abs(change)}'
                    else:
                        return '🆕 Nouveau'
                
                display_df['Changement'] = display_df.apply(calc_change, axis=1)
                
                st.dataframe(
                    display_df[['Nouveau Rang', 'Type', 'Nom', 'Taille (M Oz)', 'Rang Original', 'Changement']],
                    use_container_width=True,
                    height=400
                )
                
                # Export CSV et Excel
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name="distribution_reclassee.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_data = df_to_excel({
                        'Distribution Reclassée': display_df[['Nouveau Rang', 'Type', 'Nom', 'Taille (M Oz)', 'Rang Original', 'Changement']],
                        'Données Brutes': combined_df
                    })
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=excel_data,
                        file_name="distribution_reclassee.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with tab4:
            st.subheader("📋 Tableau des Données")
            
            # Préparer le tableau
            display_df = df.copy()
            display_df['Rang'] = display_df['rank']
            display_df['Nom'] = display_df['name']
            display_df['Taille (Oz)'] = display_df['size'].apply(lambda x: f"{x:,.0f}")
            display_df['Taille (M Oz)'] = (display_df['size'] / 1e6).round(2)
            
            st.dataframe(
                display_df[['Rang', 'Nom', 'Taille (Oz)', 'Taille (M Oz)']],
                use_container_width=True,
                height=500
            )
            
            # Export
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="donnees_gisements.csv",
                    mime="text/csv"
                )
            
            with col2:
                excel_data = df_to_excel({
                    'Données': display_df[['Rang', 'Nom', 'Taille (Oz)', 'Taille (M Oz)']]
                })
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=excel_data,
                    file_name="donnees_gisements.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.warning("⚠️ Veuillez entrer au moins 3 gisements valides pour effectuer l'analyse.")
else:
    st.info("👆 Entrez vos données ci-dessus pour commencer l'analyse.")

# Pied de page
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Analyse Loi de Zipf - Gisements d'Or</strong></p>
        <p>Développé avec ❤️ par Didier Ouedraogo & Koulou Danshoko</p>
    </div>
""", unsafe_allow_html=True)