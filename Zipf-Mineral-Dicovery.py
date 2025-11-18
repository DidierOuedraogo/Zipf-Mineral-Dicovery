import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Zipf Mineral Discovery Application",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le pied de page et centrage
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #262730;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e0e0e0;
        z-index: 999;
    }
    .main {
        margin-bottom: 60px;
    }
    h1, h2, h3 {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# En-tête principal
st.title("⛏️ Zipf Mineral Discovery Application")
st.markdown("### Test de Kolmogorov-Smirnov et prédictions des gisements non découverts dans une province aurifère")
st.markdown("<p style='text-align: center;'><strong>Auteurs: Didier Ouedraogo, PGeo | Koulou Danshoko, Geo</strong></p>", unsafe_allow_html=True)
st.divider()

# Sidebar pour les paramètres
with st.sidebar:
    st.header("📊 Paramètres de l'analyse")
    
    # Options de chargement des données
    data_source = st.radio(
        "Source des données",
        ["Données d'exemple", "Charger un fichier"]
    )
    
    uploaded_file = None
    if data_source == "Charger un fichier":
        uploaded_file = st.file_uploader(
            "Charger un fichier CSV ou Excel",
            type=["csv", "xlsx", "xls"]
        )
    
    st.divider()
    
    # Paramètres de l'analyse
    confidence_level = st.slider(
        "Niveau de confiance (%)",
        min_value=90,
        max_value=99,
        value=95,
        step=1
    )
    
    alpha = 1 - (confidence_level / 100)
    
    st.divider()
    st.info("💡 L'analyse de Zipf permet d'estimer le nombre de gisements non encore découverts dans une province minière.")

# Fonction pour générer des données d'exemple
def generate_example_data():
    np.random.seed(42)
    n_deposits = 25
    tonnages = np.sort(np.random.lognormal(mean=5, sigma=1.5, size=n_deposits))[::-1]
    
    data = pd.DataFrame({
        'Rang': range(1, n_deposits + 1),
        'Tonnage (Mt)': tonnages,
        'Nom_Gisement': [f'Gisement_{i}' for i in range(1, n_deposits + 1)]
    })
    return data

# Chargement des données
df = None
if data_source == "Données d'exemple":
    df = generate_example_data()
    st.success("✅ Données d'exemple chargées avec succès!")
else:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {e}")
            st.stop()
    else:
        st.warning("⚠️ Veuillez charger un fichier pour continuer.")
        st.stop()

# Vérification des colonnes requises
if df is not None:
    required_columns = ['Tonnage (Mt)']
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ Le fichier doit contenir au minimum la colonne: {required_columns}")
        st.info("Colonnes disponibles: " + ", ".join(df.columns.tolist()))
        st.stop()

# Validation et préparation des données
st.header("📋 Aperçu des données")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nombre de gisements", len(df))
with col2:
    st.metric("Tonnage total (Mt)", f"{df['Tonnage (Mt)'].sum():.2f}")
with col3:
    st.metric("Tonnage moyen (Mt)", f"{df['Tonnage (Mt)'].mean():.2f}")

st.dataframe(df, use_container_width=True)

# Tri des données par tonnage décroissant
df_sorted = df.sort_values('Tonnage (Mt)', ascending=False).reset_index(drop=True)
df_sorted['Rang'] = range(1, len(df_sorted) + 1)

# Analyse de Zipf
st.header("📈 Analyse de Zipf")

# Transformation logarithmique
df_sorted['log_Rang'] = np.log(df_sorted['Rang'])
df_sorted['log_Tonnage'] = np.log(df_sorted['Tonnage (Mt)'])

# Régression linéaire
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_sorted['log_Rang'], 
    df_sorted['log_Tonnage']
)

# Prédictions
df_sorted['log_Tonnage_pred'] = slope * df_sorted['log_Rang'] + intercept
df_sorted['Tonnage_pred'] = np.exp(df_sorted['log_Tonnage_pred'])

# Affichage des résultats de régression
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Résultats de la régression")
    st.write(f"**Pente (b):** {slope:.4f}")
    st.write(f"**Ordonnée à l'origine (a):** {intercept:.4f}")
    st.write(f"**Coefficient de corrélation (R):** {r_value:.4f}")
    st.write(f"**R² (coefficient de détermination):** {r_value**2:.4f}")
    st.write(f"**p-value:** {p_value:.4e}")
    st.write(f"**Erreur standard:** {std_err:.4f}")
    
    if r_value**2 > 0.9:
        st.success("✅ Excellente qualité d'ajustement (R² > 0.9)")
    elif r_value**2 > 0.7:
        st.warning("⚠️ Bonne qualité d'ajustement (R² > 0.7)")
    else:
        st.error("❌ Qualité d'ajustement faible (R² < 0.7)")

with col2:
    st.subheader("📐 Équation de la droite de Zipf")
    st.latex(r"\log(T) = a + b \times \log(R)")
    st.write(f"**Équation:** log(T) = {intercept:.4f} + {slope:.4f} × log(R)")
    st.latex(r"T = e^{a} \times R^{b}")
    st.write(f"**Forme exponentielle:** T = {np.exp(intercept):.4f} × R^{{{slope:.4f}}}")

# Graphique de Zipf (échelle log-log)
st.subheader("📉 Graphique de Zipf (échelle log-log)")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(df_sorted['Rang'], df_sorted['Tonnage (Mt)'], 
            alpha=0.6, s=100, label='Données observées', color='steelblue')
ax1.plot(df_sorted['Rang'], df_sorted['Tonnage_pred'], 
         'r--', linewidth=2, label='Droite de Zipf ajustée')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Rang (échelle log)', fontsize=12)
ax1.set_ylabel('Tonnage Mt (échelle log)', fontsize=12)
ax1.set_title('Loi de Zipf - Distribution des tonnages', fontsize=14, fontweight='bold', loc='center')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)
plt.close()

# Test de Kolmogorov-Smirnov
st.header("🔬 Test de Kolmogorov-Smirnov")

# Calcul des résidus
residuals = df_sorted['log_Tonnage'] - df_sorted['log_Tonnage_pred']
residuals_standardized = (residuals - residuals.mean()) / residuals.std()

# Test KS pour normalité
ks_statistic, ks_pvalue = stats.kstest(residuals_standardized, 'norm')

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Résultats du test KS")
    st.write(f"**Statistique KS:** {ks_statistic:.4f}")
    st.write(f"**p-value:** {ks_pvalue:.4f}")
    st.write(f"**Niveau de signification (α):** {alpha:.4f}")
    
    if ks_pvalue > alpha:
        st.success(f"✅ Les résidus suivent une distribution normale (p-value = {ks_pvalue:.4f} > α = {alpha:.4f})")
    else:
        st.warning(f"⚠️ Les résidus ne suivent pas une distribution normale (p-value = {ks_pvalue:.4f} ≤ α = {alpha:.4f})")

with col2:
    st.subheader("📈 Distribution des résidus")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(residuals_standardized, bins=15, density=True, 
             alpha=0.7, color='skyblue', edgecolor='black')
    
    x = np.linspace(residuals_standardized.min(), residuals_standardized.max(), 100)
    ax2.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Distribution normale')
    ax2.set_xlabel('Résidus standardisés', fontsize=12)
    ax2.set_ylabel('Densité', fontsize=12)
    ax2.set_title('Distribution des résidus vs. Normale', fontsize=14, fontweight='bold', loc='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

# Q-Q plot
st.subheader("📊 Q-Q Plot (Quantile-Quantile)")
fig3, ax3 = plt.subplots(figsize=(8, 6))
stats.probplot(residuals_standardized, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot - Vérification de la normalité', fontsize=14, fontweight='bold', loc='center')
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)
plt.close()

# Prédiction des gisements non découverts
st.header("🔮 Prédiction des gisements non découverts")

# Estimation du nombre total de gisements
n_observed = len(df_sorted)
extrapolation_factor = st.slider(
    "Facteur d'extrapolation (% au-delà du dernier rang observé)",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

n_total_estimated = int(n_observed * (1 + extrapolation_factor / 100))
n_undiscovered = n_total_estimated - n_observed

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Gisements observés", n_observed)
with col2:
    st.metric("Gisements estimés (total)", n_total_estimated)
with col3:
    st.metric("Gisements non découverts", n_undiscovered, delta=f"+{(n_undiscovered/n_observed)*100:.1f}%")

# Prédiction des tonnages futurs
future_ranks = np.arange(n_observed + 1, n_total_estimated + 1)
future_log_tonnages = slope * np.log(future_ranks) + intercept
future_tonnages = np.exp(future_log_tonnages)

df_future = pd.DataFrame({
    'Rang': future_ranks,
    'Tonnage_prédit (Mt)': future_tonnages
})

st.subheader("📋 Gisements prédits non découverts")
st.dataframe(df_future.head(20), use_container_width=True)

total_future_tonnage = future_tonnages.sum()
st.info(f"💎 **Tonnage total estimé des gisements non découverts:** {total_future_tonnage:.2f} Mt")

# Graphique comparatif
st.subheader("📊 Comparaison: Gisements observés vs. Prédits")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=df_sorted['Rang'],
    y=df_sorted['Tonnage (Mt)'],
    mode='markers',
    name='Gisements observés',
    marker=dict(size=10, color='steelblue')
))
fig4.add_trace(go.Scatter(
    x=df_future['Rang'],
    y=df_future['Tonnage_prédit (Mt)'],
    mode='markers',
    name='Gisements prédits',
    marker=dict(size=8, color='coral', symbol='diamond')
))
fig4.update_xaxes(type="log", title="Rang (échelle log)")
fig4.update_yaxes(type="log", title="Tonnage Mt (échelle log)")
fig4.update_layout(
    title={
        'text': "Distribution complète: Observés + Prédictions",
        'x': 0.5,
        'xanchor': 'center'
    },
    height=500,
    hovermode='closest'
)
st.plotly_chart(fig4, use_container_width=True)

# Analyse de sensibilité
st.header("🎯 Analyse de sensibilité")

st.write("Impact du facteur d'extrapolation sur les estimations:")

sensitivity_factors = [10, 25, 50, 75, 100, 150, 200]
sensitivity_results = []

for factor in sensitivity_factors:
    n_total = int(n_observed * (1 + factor / 100))
    n_undiscov = n_total - n_observed
    future_r = np.arange(n_observed + 1, n_total + 1)
    future_t = np.exp(slope * np.log(future_r) + intercept)
    total_t = future_t.sum()
    
    sensitivity_results.append({
        'Facteur (%)': factor,
        'Gisements totaux': n_total,
        'Gisements non découverts': n_undiscov,
        'Tonnage prédit (Mt)': total_t
    })

df_sensitivity = pd.DataFrame(sensitivity_results)
st.dataframe(df_sensitivity, use_container_width=True)

# Graphique de sensibilité
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

ax5a.plot(df_sensitivity['Facteur (%)'], df_sensitivity['Gisements non découverts'], 
          marker='o', linewidth=2, color='steelblue')
ax5a.set_xlabel('Facteur d\'extrapolation (%)', fontsize=12)
ax5a.set_ylabel('Nombre de gisements non découverts', fontsize=12)
ax5a.set_title('Impact sur le nombre de gisements', fontsize=14, fontweight='bold', loc='center')
ax5a.grid(True, alpha=0.3)

ax5b.plot(df_sensitivity['Facteur (%)'], df_sensitivity['Tonnage prédit (Mt)'], 
          marker='s', linewidth=2, color='coral')
ax5b.set_xlabel('Facteur d\'extrapolation (%)', fontsize=12)
ax5b.set_ylabel('Tonnage total prédit (Mt)', fontsize=12)
ax5b.set_title('Impact sur le tonnage total', fontsize=14, fontweight='bold', loc='center')
ax5b.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig5)
plt.close()

# Statistiques supplémentaires
st.header("📊 Statistiques supplémentaires")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 Tonnages observés")
    st.write(f"**Minimum:** {df_sorted['Tonnage (Mt)'].min():.2f} Mt")
    st.write(f"**Maximum:** {df_sorted['Tonnage (Mt)'].max():.2f} Mt")
    st.write(f"**Médiane:** {df_sorted['Tonnage (Mt)'].median():.2f} Mt")
    st.write(f"**Écart-type:** {df_sorted['Tonnage (Mt)'].std():.2f} Mt")

with col2:
    st.subheader("🔮 Tonnages prédits")
    st.write(f"**Minimum:** {df_future['Tonnage_prédit (Mt)'].min():.2f} Mt")
    st.write(f"**Maximum:** {df_future['Tonnage_prédit (Mt)'].max():.2f} Mt")
    st.write(f"**Médiane:** {df_future['Tonnage_prédit (Mt)'].median():.2f} Mt")
    st.write(f"**Écart-type:** {df_future['Tonnage_prédit (Mt)'].std():.2f} Mt")

with col3:
    st.subheader("💰 Totaux cumulés")
    st.write(f"**Tonnage observé:** {df_sorted['Tonnage (Mt)'].sum():.2f} Mt")
    st.write(f"**Tonnage prédit:** {total_future_tonnage:.2f} Mt")
    st.write(f"**Total combiné:** {df_sorted['Tonnage (Mt)'].sum() + total_future_tonnage:.2f} Mt")
    percentage_increase = (total_future_tonnage / df_sorted['Tonnage (Mt)'].sum()) * 100
    st.write(f"**Augmentation:** +{percentage_increase:.1f}%")

# Export des résultats
st.header("💾 Export des résultats")

col1, col2, col3 = st.columns(3)

with col1:
    # Export des données observées
    csv_observed = df_sorted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données observées (CSV)",
        data=csv_observed,
        file_name="donnees_observees.csv",
        mime="text/csv"
    )

with col2:
    # Export des prédictions
    csv_predictions = df_future.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les prédictions (CSV)",
        data=csv_predictions,
        file_name="predictions_gisements.csv",
        mime="text/csv"
    )

with col3:
    # Export de l'analyse de sensibilité
    csv_sensitivity = df_sensitivity.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger l'analyse de sensibilité (CSV)",
        data=csv_sensitivity,
        file_name="analyse_sensibilite.csv",
        mime="text/csv"
    )

# Documentation et méthodologie
with st.expander("📖 Documentation et Méthodologie"):
    st.markdown("""
    ## Méthodologie de l'Analyse de Zipf
    
    ### 1. Loi de Zipf
    La loi de Zipf, appliquée aux gisements minéraux, stipule que le tonnage d'un gisement est inversement proportionnel à son rang:
    
    **T = C × R^(-b)**
    
    Où:
    - T = Tonnage du gisement
    - R = Rang du gisement (du plus grand au plus petit)
    - C = Constante
    - b = Exposant de Zipf (généralement proche de 1)
    
    ### 2. Transformation logarithmique
    En appliquant le logarithme naturel, on obtient une relation linéaire:
    
    **log(T) = a + b × log(R)**
    
    ### 3. Régression linéaire
    La régression linéaire permet d'estimer les paramètres a et b, ainsi que la qualité de l'ajustement (R²).
    
    ### 4. Test de Kolmogorov-Smirnov
    Ce test vérifie si les résidus suivent une distribution normale, validant ainsi le modèle statistique.
    
    ### 5. Prédictions
    Le modèle calibré permet d'extrapoler et d'estimer le nombre et le tonnage des gisements non encore découverts.
    
    ### Limites et précautions
    - Les prédictions dépendent de la qualité des données d'entrée
    - Le facteur d'extrapolation doit être choisi judicieusement
    - Les résultats doivent être interprétés dans le contexte géologique de la province
    """)

# Pied de page
st.markdown("""
    <div class="footer">
        <p><strong>Développé par Didier Ouedraogo, P.Geo | Koulou Danshoko, Geo</strong></p>
    </div>
    """, unsafe_allow_html=True)