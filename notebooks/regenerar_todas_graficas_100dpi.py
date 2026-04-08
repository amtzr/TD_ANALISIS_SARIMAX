#!/usr/bin/env python3
"""
Script para regenerar TODAS las gráficas del análisis a 100 DPI
con las mismas especificaciones:
- 100 DPI
- Sin encabezados principales
- Estilo negro con marcadores
- Área blanca
- Etiquetas correctas con unidades
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos - Especificaciones detalladas
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['black'])
plt.rcParams['lines.markersize'] = 4
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# Mantener ejes horizontal e inferior visibles
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
sns.set_palette(["black"])

# Definiciones de etiquetas para variables
ETIQUETAS_VARIABLES = {
    'Temperatura': 'T(°C)',
    'Humedad_Relativa': 'HR(%)',
    'O3': 'O3(ppbv)',
    'PM25': 'PM2.5(μg/m³)',
    'PM10': 'PM10(μg/m³)',
    'CO': 'CO(ppm)',
    'tiempo': 'tiempo'
}

class RegeneradorTodasGraficas:
    """
    Clase para regenerar todas las gráficas del análisis a 100 DPI
    """
    
    def __init__(self, ruta_datos):
        """
        Inicializar el regenerador con la ruta de los datos
        """
        self.ruta_datos = ruta_datos
        self.df = None
        self.df_procesado = None
        self.resultados = {}
    
    def cargar_datos(self):
        """
        Cargar y preparar datos
        """
        print("Cargando datos para regeneración...")
        self.df = pd.read_csv(self.ruta_datos)
        
        # Convertir timestamp a datetime
        self.df['TimeStamp'] = pd.to_datetime(self.df['TimeStamp'])
        self.df.set_index('TimeStamp', inplace=True)
        
        # Renombrar columnas para consistencia
        self.df = self.df.rename(columns={
            'carbon_monoxide_Data': 'CO',
            'ozone_Data': 'O3',
            'pm10_Data': 'PM10',
            'pm25_Data': 'PM25',
            'relative_humidity_Data': 'Humedad_Relativa',
            'temperature_Data': 'Temperatura'
        })
        
        # Seleccionar columnas numéricas
        columnas_numericas = ['CO', 'O3', 'PM10', 'PM25', 'Humedad_Relativa', 'Temperatura']
        self.df_procesado = self.df[columnas_numericas].copy()
        
        # Eliminar valores faltantes
        self.df_procesado = self.df_procesado.dropna()
        
        print(f"Datos cargados: {len(self.df_procesado)} registros")
        return self.df_procesado
    
    def generar_series_tiempo_100dpi(self):
        """
        Generar gráficas de series de tiempo a 100 DPI
        """
        print("\nGenerando gráficas de series de tiempo a 100 DPI...")
        
        variables = self.df_procesado.columns
        
        # Gráfica individual para cada variable
        for variable in variables:
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            # Graficar con marcador
            ax.plot(self.df_procesado.index, self.df_procesado[variable], 
                    color='black', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
            
            # Configurar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            
            ax.set_title(f'Evolución Temporal de {variable}', fontsize=12, fontweight='bold')
            ax.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            ax.set_ylabel(etiqueta_y, fontsize=10)
            
            # Sin grid pero mantener ejes horizontal e inferior
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            # Guardar en formato SVG con 100 dpi
            plt.savefig(f'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/serie_individual_{variable.lower()}_100dpi.svg', 
                       format='svg', dpi=100, bbox_inches='tight')
            plt.close()
        
        # Gráfica comparativa de todas las variables
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Series de Tiempo - Variables Ambientales', fontsize=12, fontweight='bold')
        
        # Definir marcadores diferentes para cada variable
        marcadores = ['o', 's', '^', 'v', 'D', 'p']
        
        for i, variable in enumerate(variables):
            row = i // 2
            col = i % 2
            
            # Usar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            marcador = marcadores[i % len(marcadores)]
            
            axes[row, col].plot(self.df_procesado.index, self.df_procesado[variable], 
                             color='black', linewidth=1.5, marker=marcador, 
                             markersize=3, alpha=0.8, label=variable)
            axes[row, col].set_title(variable, fontsize=10, fontweight='bold')
            axes[row, col].set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            axes[row, col].set_ylabel(etiqueta_y, fontsize=10)
            
            # Sin grid pero mantener ejes horizontal e inferior
            axes[row, col].grid(False)
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)
        
        # Añadir leyenda general
        fig.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        # Guardar en formato SVG con 100 dpi
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/series_tiempo_completas_100dpi.svg', 
                   format='svg', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Gráficas de series de tiempo guardadas en formato SVG con 100 DPI")
    
    def generar_matriz_correlacion_100dpi(self):
        """
        Generar matriz de correlación a 100 DPI
        """
        print("\nGenerando matriz de correlación a 100 DPI...")
        
        correlacion = self.df_procesado.corr()
        
        # Mapa de calor de correlación
        plt.figure(figsize=(10, 8))
        plt.gca().set_facecolor('white')
        
        # Crear mapa de calor personalizado
        im = plt.imshow(correlacion, cmap='gray_r', aspect='auto', vmin=-1, vmax=1)
        
        # Añadir valores numéricos
        for i in range(len(correlacion)):
            for j in range(len(correlacion.columns)):
                plt.text(j, i, f'{correlacion.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=10)
        
        # Configurar etiquetas
        plt.xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(len(correlacion.index)), correlacion.index, fontsize=10)
        
        plt.title('Matriz de Correlación - Variables Ambientales', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Variables', fontsize=10)
        plt.ylabel('Variables', fontsize=10)
        
        # Sin grid pero mantener ejes horizontal e inferior
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Añadir barra de color
        cbar = plt.colorbar(im)
        cbar.set_label('Coeficiente de Correlación', fontsize=10)
        
        plt.tight_layout()
        # Guardar en formato SVG con 100 dpi
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/matriz_correlacion_100dpi.svg', 
                   format='svg', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Matriz de correlación guardada en formato SVG con 100 DPI")
    
    def generar_agrupaciones_100dpi(self):
        """
        Generar agrupaciones analíticas a 100 DPI
        """
        print("\nGenerando agrupaciones analíticas a 100 DPI...")
        
        # Agrupación 1: Contaminantes (CO, O3, PM10, PM25)
        contaminantes = ['CO', 'O3', 'PM10', 'PM25']
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        fig1.patch.set_facecolor('white')
        
        marcadores = ['o', 's', '^', 'v']
        for i, cont in enumerate(contaminantes):
            marcador = marcadores[i % len(marcadores)]
            etiqueta_y = ETIQUETAS_VARIABLES.get(cont, cont)
            ax1.plot(self.df_procesado.index, self.df_procesado[cont], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=cont)
        
        ax1.set_title('Contaminantes Atmosféricos', fontsize=12, fontweight='bold')
        ax1.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax1.set_ylabel('Concentración', fontsize=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/agrupacion_contaminantes_100dpi.svg', 
                   format='svg', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Agrupación 2: Variables Meteorológicas (Temperatura, Humedad)
        meteo = ['Temperatura', 'Humedad_Relativa']
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig2.patch.set_facecolor('white')
        
        for i, var in enumerate(meteo):
            marcador = marcadores[i % len(marcadores)]
            etiqueta_y = ETIQUETAS_VARIABLES.get(var, var)
            ax2.plot(self.df_procesado.index, self.df_procesado[var], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=var)
        
        ax2.set_title('Variables Meteorológicas', fontsize=12, fontweight='bold')
        ax2.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax2.set_ylabel('Valor', fontsize=10)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/agrupacion_meteorologicas_100dpi.svg', 
                   format='svg', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Agrupaciones analíticas guardadas en formato SVG con 100 DPI")
    
    def regenerar_todas_graficas_100dpi(self):
        """
        Regenerar todas las gráficas a 100 DPI
        """
        if self.df_procesado is None:
            self.cargar_datos()
        
        print("\n" + "="*60)
        print("REGENERANDO TODAS LAS GRÁFICAS A 100 DPI")
        print("="*60)
        
        # Generar todas las gráficas
        self.generar_series_tiempo_100dpi()
        self.generar_matriz_correlacion_100dpi()
        self.generar_agrupaciones_100dpi()
        
        print("\n" + "="*60)
        print("REGENERACIÓN COMPLETADA A 100 DPI")
        print("="*60)
        print("Todas las gráficas han sido regeneradas con:")
        print("- 100 DPI")
        print("- Sin encabezados principales")
        print("- Estilo negro con marcadores")
        print("- Área blanca")
        print("- Etiquetas correctas con unidades")

def main():
    """
    Función principal para regenerar todas las gráficas
    """
    ruta_datos = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/DATA/DATA_LIMPIA.csv'
    
    # Crear regenerador
    regenerador = RegeneradorTodasGraficas(ruta_datos)
    
    # Regenerar todas las gráficas
    regenerador.regenerar_todas_graficas_100dpi()

if __name__ == "__main__":
    main()
