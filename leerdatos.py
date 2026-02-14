import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Cargar datos
file = 'datos_sinteticos.csv'
df = pd.read_csv(file)

print("=" * 80)
print("ANÁLISIS DE DATOS DE CAMPAÑAS DE MARKETING")
print("=" * 80)

# 1. Información general del dataset
print("\n1. INFORMACIÓN GENERAL DEL DATASET")
print(f"   Total de registros: {len(df)}")
print(f"   Total de columnas: {len(df.columns)}")
print(f"   Rango de fechas: {df['fecha_campana'].min()} a {df['fecha_campana'].max()}")

# 2. Resumen estadístico
print("\n2. RESUMEN ESTADÍSTICO DE MÉTRICAS NUMÉRICAS")
print(df.describe().round(2))

# 3. Análisis por Plataforma
print("\n3. ANÁLISIS POR PLATAFORMA")
plataformas = df.groupby('plataforma').agg({
    'campana_id': 'count',
    'presupuesto_diario': 'sum',
    'impresiones': 'sum',
    'clicks': 'sum',
    'conversiones': 'sum',
    'revenue_generado': 'sum',
    'costo_total': 'sum',
    'roas': 'mean',
    'engagement_rate': 'mean',
    'conversion_rate': 'mean'
}).round(2)
plataformas.columns = ['Campañas', 'Presupuesto Total', 'Impresiones', 'Clicks', 
                       'Conversiones', 'Revenue', 'Costo Total', 'ROAS promedio', 
                       'Engagement %', 'Conversion %']
print(plataformas)

# 4. Análisis por Tipo de Campaña
print("\n4. ANÁLISIS POR TIPO DE CAMPAÑA")
tipos = df.groupby('tipo_campana').agg({
    'campana_id': 'count',
    'revenue_generado': ['sum', 'mean'],
    'costo_total': 'sum',
    'roas': 'mean',
    'conversion_rate': 'mean'
}).round(2)
print(tipos)

# 5. Análisis por Audiencia Objetivo
print("\n5. ANÁLISIS POR AUDIENCIA OBJETIVO")
audiencia = df.groupby('audiencia_objetivo').agg({
    'campana_id': 'count',
    'conversiones': 'sum',
    'revenue_generado': 'sum',
    'costo_total': 'sum',
    'conversion_rate': 'mean',
    'roas': 'mean'
}).round(2)
audiencia.columns = ['Campañas', 'Conversiones Total', 'Revenue', 'Costo', 'Conversion %', 'ROAS promedio']
print(audiencia)

# 6. Campañas con mejor y peor desempeño
print("\n6. CAMPAÑAS CON MEJOR Y PEOR DESEMPEÑO (por ROAS)")
print("\n   TOP 5 CAMPAÑAS (Mayor ROAS):")
top_5 = df.nlargest(5, 'roas')[['campana_id', 'plataforma', 'tipo_campana', 'revenue_generado', 'costo_total', 'roas']]
print(top_5.to_string())

print("\n   PEORES 5 CAMPAÑAS (Menor ROAS):")
bottom_5 = df.nsmallest(5, 'roas')[['campana_id', 'plataforma', 'tipo_campana', 'revenue_generado', 'costo_total', 'roas']]
print(bottom_5.to_string())

# 7. Correlaciones importantes
print("\n7. CORRELACIONES ENTRE MÉTRICAS CLAVE")
columnas_numericas = ['presupuesto_diario', 'impresiones', 'clicks', 'conversiones', 
                      'costo_total', 'revenue_generado', 'engagement_rate', 'conversion_rate', 'roas']
correlaciones = df[columnas_numericas].corr()['roas'].sort_values(ascending=False)
print(correlaciones.round(3))

# 8. CTR y Conversion Rate promedio
print("\n8. MÉTRICAS DE RENDIMIENTO PROMEDIO")
print(f"   CTR promedio: {df['ctr'].mean():.2f}%")
print(f"   Conversion Rate promedio: {df['conversion_rate'].mean():.2f}%")
print(f"   Engagement Rate promedio: {df['engagement_rate'].mean():.2f}%")
print(f"   ROAS promedio: {df['roas'].mean():.2f}")
print(f"   CPA promedio: ${df['cpa'].mean():.2f}")
print(f"   CPC promedio: ${df['cpc'].mean():.2f}")

# 9. Rentabilidad general
print("\n9. ANÁLISIS DE RENTABILIDAD GENERAL")
revenue_total = df['revenue_generado'].sum()
costo_total = df['costo_total'].sum()
ganancia = revenue_total - costo_total
roi = ((revenue_total - costo_total) / costo_total * 100) if costo_total > 0 else 0

print(f"   Revenue total generado: ${revenue_total:,.2f}")
print(f"   Costo total: ${costo_total:,.2f}")
print(f"   Ganancia neta: ${ganancia:,.2f}")
print(f"   ROI: {roi:.2f}%")
print(f"   ROAS general: {revenue_total/costo_total:.2f}")

# 10. Valores perdidos
print("\n10. ANÁLISIS DE VALORES FALTANTES")
valores_nulos = df.isnull().sum()
if valores_nulos.sum() == 0:
    print("   No hay valores faltantes en el dataset")
else:
    print(valores_nulos[valores_nulos > 0])

print("\n" + "=" * 80)

# ============================================================================
# GENERACIÓN DE VISUALIZACIONES
# ============================================================================

print("\n✓ Generando visualizaciones...")

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(18, 14))

# 1. ROAS por Plataforma
ax1 = plt.subplot(3, 3, 1)
plataforma_roas = df.groupby('plataforma')['roas'].mean().sort_values(ascending=False)
colors = sns.color_palette("husl", len(plataforma_roas))
plataforma_roas.plot(kind='barh', ax=ax1, color=colors)
ax1.set_title('ROAS Promedio por Plataforma', fontsize=12, fontweight='bold')
ax1.set_xlabel('ROAS')
for i, v in enumerate(plataforma_roas):
    ax1.text(v + 0.2, i, f'{v:.2f}', va='center')

# 2. Conversiones por Tipo de Campaña
ax2 = plt.subplot(3, 3, 2)
tipo_conversiones = df.groupby('tipo_campana')['conversiones'].sum().sort_values(ascending=False)
ax2.bar(tipo_conversiones.index, tipo_conversiones.values, color=sns.color_palette("Set2", len(tipo_conversiones)))
ax2.set_title('Total Conversiones por Tipo', fontsize=12, fontweight='bold')
ax2.set_ylabel('Conversiones')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, v in enumerate(tipo_conversiones.values):
    ax2.text(i, v + 5, str(int(v)), ha='center', fontweight='bold')

# 3. Revenue por Plataforma
ax3 = plt.subplot(3, 3, 3)
plataforma_revenue = df.groupby('plataforma')['revenue_generado'].sum().sort_values(ascending=False)
colors = sns.color_palette("coolwarm", len(plataforma_revenue))
wedges, texts, autotexts = ax3.pie(plataforma_revenue.values, labels=plataforma_revenue.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('Distribución de Revenue por Plataforma', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 4. Conversión Rate por Audiencia
ax4 = plt.subplot(3, 3, 4)
audiencia_conversion = df.groupby('audiencia_objetivo')['conversion_rate'].mean().sort_values(ascending=False)
ax4.bar(audiencia_conversion.index, audiencia_conversion.values, color=sns.color_palette("viridis", len(audiencia_conversion)))
ax4.set_title('Conversion Rate Promedio por Audiencia', fontsize=12, fontweight='bold')
ax4.set_ylabel('Conversion Rate (%)')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, v in enumerate(audiencia_conversion.values):
    ax4.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=9)

# 5. Impresiones vs Conversiones (scatter)
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(df['impresiones'], df['conversiones'], 
                     s=df['revenue_generado']/2, 
                     c=df['roas'], cmap='viridis', alpha=0.6, edgecolors='black')
ax5.set_xlabel('Impresiones')
ax5.set_ylabel('Conversiones')
ax5.set_title('Impresiones vs Conversiones (tamaño=Revenue)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('ROAS')

# 6. Costo vs Revenue por Campaña
ax6 = plt.subplot(3, 3, 6)
x = np.arange(len(df['campana_id']))
width = 0.35
ax6.bar(x - width/2, df['costo_total'], width, label='Costo Total', alpha=0.8)
ax6.bar(x + width/2, df['revenue_generado'], width, label='Revenue', alpha=0.8)
ax6.set_xlabel('Campaña')
ax6.set_ylabel('Monto ($)')
ax6.set_title('Costo vs Revenue por Campaña', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([f'C{i+1}' for i in range(len(df))], fontsize=8)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Engagement Rate por Tipo de Campaña
ax7 = plt.subplot(3, 3, 7)
tipo_engagement = df.groupby('tipo_campana')['engagement_rate'].mean().sort_values(ascending=False)
ax7.barh(tipo_engagement.index, tipo_engagement.values, color=sns.color_palette("RdYlGn", len(tipo_engagement)))
ax7.set_title('Engagement Rate Promedio por Tipo', fontsize=12, fontweight='bold')
ax7.set_xlabel('Engagement Rate (%)')
for i, v in enumerate(tipo_engagement.values):
    ax7.text(v + 0.2, i, f'{v:.2f}%', va='center')

# 8. CTR por Plataforma
ax8 = plt.subplot(3, 3, 8)
plataforma_ctr = df.groupby('plataforma')['ctr'].mean().sort_values(ascending=False)
ax8.bar(plataforma_ctr.index, plataforma_ctr.values, color=sns.color_palette("muted", len(plataforma_ctr)))
ax8.set_title('CTR Promedio por Plataforma', fontsize=12, fontweight='bold')
ax8.set_ylabel('CTR (%)')
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, v in enumerate(plataforma_ctr.values):
    ax8.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=9)

# 9. Top 5 Campañas por ROAS
ax9 = plt.subplot(3, 3, 9)
top_5_roas = df.nlargest(5, 'roas')[['campana_id', 'roas']].sort_values('roas')
colors_roas = sns.color_palette("Spectral", len(top_5_roas))
ax9.barh(top_5_roas['campana_id'], top_5_roas['roas'], color=colors_roas)
ax9.set_title('Top 5 Campañas por ROAS', fontsize=12, fontweight='bold')
ax9.set_xlabel('ROAS')
for i, v in enumerate(top_5_roas['roas'].values):
    ax9.text(v + 0.3, i, f'{v:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_campanas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: analisis_campanas.png")

# ============================================================================
# Gráfico 2: Análisis detallado de correlaciones
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Matriz de correlación
ax1 = axes[0, 0]
columnas_cor = ['presupuesto_diario', 'impresiones', 'clicks', 'conversiones', 
                'costo_total', 'revenue_generado', 'engagement_rate', 'conversion_rate', 'roas']
corr_matrix = df[columnas_cor].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1, 
            cbar_kws={'label': 'Correlación'}, square=True)
ax1.set_title('Matriz de Correlaciones - Métricas Clave', fontsize=12, fontweight='bold')

# ROAS Distribution
ax2 = axes[0, 1]
ax2.hist(df['roas'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(df['roas'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["roas"].mean():.2f}')
ax2.axvline(df['roas'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["roas"].median():.2f}')
ax2.set_xlabel('ROAS')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de ROAS', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Evolución temporal de Revenue
ax3 = axes[1, 0]
df_sorted = df.sort_values('fecha_campana')
ax3.plot(range(len(df_sorted)), df_sorted['revenue_generado'].values, marker='o', linewidth=2, markersize=8, color='green')
ax3.fill_between(range(len(df_sorted)), df_sorted['revenue_generado'].values, alpha=0.3, color='green')
ax3.set_xlabel('Campaña (ordenada por fecha)')
ax3.set_ylabel('Revenue Generado ($)')
ax3.set_title('Evolución Temporal de Revenue', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Presupuesto vs ROAS vs Conversiones
ax4 = axes[1, 1]
scatter2 = ax4.scatter(df['presupuesto_diario'], df['conversiones'], 
                      s=df['roas']*50, c=df['revenue_generado'], 
                      cmap='plasma', alpha=0.6, edgecolors='black', linewidth=1.5)
ax4.set_xlabel('Presupuesto Diario ($)')
ax4.set_ylabel('Conversiones')
ax4.set_title('Presupuesto vs Conversiones (color=Revenue, tamaño=ROAS)', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax4)
cbar2.set_label('Revenue ($)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_detallado.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: analisis_detallado.png")

# ============================================================================
# Resumen impreso
# ============================================================================

print("=" * 80)
