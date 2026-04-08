#!/usr/bin/env python3
"""
Script para regenerar gráficas de descomposición con especificaciones mejoradas:
- 200 DPI
- Sin encabezados principales
- Estilo negro con marcadores
- Área blanca
- Etiquetas correctas en eje vertical
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

class RegeneradorDescomposicion:
    """
    Clase para regenerar gráficas de descomposición con nuevas especificaciones
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
    
    def generar_descomposicion_mejorada(self, variable):
        """
        Generar descomposición con especificaciones mejoradas
        """
        print(f"\nRegenerando descomposición para {variable}...")
        
        try:
            # Descomposición estacional
            descomposicion = seasonal_decompose(
                self.df_procesado[variable], 
                model='additive', 
                period=24  # Período de 24 horas
            )
            
            # Extraer componentes
            tendencia = descomposicion.trend.dropna()
            estacionalidad = descomposicion.seasonal.dropna()
            residuo = descomposicion.resid.dropna()
            
            # Calcular componente cíclico usando HP filter
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                ciclo, tendencia_hp = hpfilter(self.df_procesado[variable], lamb=1600)
                variacion_ciclica = ciclo.dropna()
            except:
                # Si falla HP filter, usar media móvil para componente cíclico
                variacion_ciclica = self.df_procesado[variable].rolling(
                    window=168, center=True
                ).mean() - tendencia  # 168 horas = 1 semana
            
            # Variación irregular (residuo después de extraer otros componentes)
            variacion_irregular = residuo
            
            # Obtener etiqueta específica para la variable
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            
            # Crear gráfica completa con 4 componentes
            fig, axes = plt.subplots(5, 1, figsize=(15, 12))
            fig.patch.set_facecolor('white')
            # SIN ENCABEZADO PRINCIPAL
            
            # Serie original
            axes[0].plot(self.df_procesado[variable].index, self.df_procesado[variable], 
                        color='black', linewidth=1, alpha=0.8, marker='o', markersize=2)
            axes[0].set_title('Serie Original', fontweight='bold')
            axes[0].set_ylabel(etiqueta_y, fontsize=10)  # Etiqueta con unidades
            axes[0].grid(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            
            # Tendencia
            axes[1].plot(tendencia.index, tendencia, 
                        color='black', linewidth=2, marker='s', markersize=2)
            axes[1].set_title('Tendencia', fontweight='bold')
            axes[1].set_ylabel('Tendencia', fontsize=10)
            axes[1].grid(False)
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            
            # Estacionalidad
            axes[2].plot(estacionalidad.index, estacionalidad, 
                        color='black', linewidth=1, alpha=0.7, marker='^', markersize=2)
            axes[2].set_title('Estacionalidad', fontweight='bold')
            axes[2].set_ylabel('Estacionalidad', fontsize=10)
            axes[2].grid(False)
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
            
            # Variación Cíclica
            axes[3].plot(variacion_ciclica.index, variacion_ciclica, 
                        color='black', linewidth=1.5, marker='D', markersize=2, alpha=0.8)
            axes[3].set_title('Variación Cíclica', fontweight='bold')
            axes[3].set_ylabel('Ciclo', fontsize=10)
            axes[3].grid(False)
            axes[3].spines['top'].set_visible(False)
            axes[3].spines['right'].set_visible(False)
            
            # Variación Irregular
            axes[4].plot(variacion_irregular.index, variacion_irregular, 
                        color='black', linewidth=1, alpha=0.6, marker='v', markersize=2)
            axes[4].set_title('Variación Irregular (Ruido)', fontweight='bold')
            axes[4].set_ylabel('Irregular', fontsize=10)
            axes[4].set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            axes[4].grid(False)
            axes[4].spines['top'].set_visible(False)
            axes[4].spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Guardar en formato SVG con 100 DPI
            ruta_svg = f'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/descomposicion_completa_{variable.lower()}.svg'
            plt.savefig(ruta_svg, format='svg', dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Descomposición regenerada: {ruta_svg}")
            
            return True
            
        except Exception as e:
            print(f"Error regenerando descomposición para {variable}: {e}")
            return False
    
    def regenerar_todas_descomposiciones(self):
        """
        Regenerar todas las gráficas de descomposición
        """
        if self.df_procesado is None:
            self.cargar_datos()
        
        variables = ['CO', 'O3', 'PM10', 'PM25', 'Humedad_Relativa', 'Temperatura']
        
        print("\n" + "="*60)
        print("REGENERANDO GRÁFICAS DE DESCOMPOSICIÓN")
        print("="*60)
        
        for variable in variables:
            self.generar_descomposicion_mejorada(variable)
        
        print("\n" + "="*60)
        print("REGENERACIÓN COMPLETADA")
        print("="*60)
        print("Todas las gráficas de descomposición han sido regeneradas con:")
        print("- 100 DPI")
        print("- Sin encabezados principales")
        print("- Estilo negro con marcadores")
        print("- Área blanca")
        print("- Etiquetas correctas en eje vertical")

def main():
    """
    Función principal para regenerar gráficas
    """
    ruta_datos = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/DATA/DATA_LIMPIA.csv'
    
    # Crear regenerador
    regenerador = RegeneradorDescomposicion(ruta_datos)
    
    # Regenerar todas las descomposiciones
    regenerador.regenerar_todas_descomposiciones()

if __name__ == "__main__":
    main()
