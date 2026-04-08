#!/usr/bin/env python3
"""
Análisis de Series de Tiempo - Datos Ambientales

Este script analiza series de tiempo de datos ambientales incluyendo:
- Monóxido de carbono (CO)
- Ozono (O3)
- Material particulado PM10
- Material particulado PM2.5
- Humedad relativa
- Temperatura

Autor: Análisis Ambiental
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

class AnalisisSeriesTiempo:
    """
    Clase para el análisis completo de series de tiempo ambientales
    """
    
    def __init__(self, ruta_datos):
        """
        Inicializar el analizador con la ruta de los datos
        
        Args:
            ruta_datos (str): Ruta al archivo CSV con los datos
        """
        self.ruta_datos = ruta_datos
        self.df = None
        self.df_procesado = None
        self.resultados = {}
        
    def cargar_datos(self):
        """
        Cargar y preprocesar los datos del archivo CSV
        """
        print("Cargando datos...")
        
        # Cargar datos
        self.df = pd.read_csv(self.ruta_datos)
        
        # Convertir timestamp a datetime
        self.df['TimeStamp'] = pd.to_datetime(self.df['TimeStamp'])
        
        # Establecer timestamp como índice
        self.df.set_index('TimeStamp', inplace=True)
        
        # Ordenar por fecha
        self.df.sort_index(inplace=True)
        
        # Renombrar columnas para mejor legibilidad
        columnas_renombradas = {
            'carbon_monoxide_Data': 'CO',
            'ozone_Data': 'O3',
            'pm10_Data': 'PM10',
            'pm25_Data': 'PM25',
            'relative_humidity_Data': 'Humedad_Relativa',
            'temperature_Data': 'Temperatura'
        }
        self.df.rename(columns=columnas_renombradas, inplace=True)
        
        print(f"Datos cargados: {len(self.df)} registros")
        print(f"Rango de fechas: {self.df.index.min()} a {self.df.index.max()}")
        print(f"Columnas: {list(self.df.columns)}")
        
        return self.df
    
    def limpiar_datos(self):
        """
        Limpiar y preprocesar los datos
        """
        print("Limpiando datos...")
        
        # Crear copia para no modificar originales
        self.df_procesado = self.df.copy()
        
        # Estadísticas de valores faltantes
        print("\nValores faltantes por columna:")
        for col in self.df_procesado.columns:
            nulos = self.df_procesado[col].isnull().sum()
            porcentaje = (nulos / len(self.df_procesado)) * 100
            print(f"{col}: {nulos} ({porcentaje:.2f}%)")
        
        # Interpolación lineal para valores faltantes
        for col in self.df_procesado.columns:
            if self.df_procesado[col].isnull().sum() > 0:
                self.df_procesado[col] = self.df_procesado[col].interpolate(method='linear')
        
        # Eliminar filas que aún tengan valores nulos después de interpolación
        self.df_procesado.dropna(inplace=True)
        
        # Detectar y eliminar outliers usando método IQR
        for col in self.df_procesado.columns:
            Q1 = self.df_procesado[col].quantile(0.25)
            Q3 = self.df_procesado[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            # Reemplazar outliers con valores en los límites
            self.df_procesado[col] = np.clip(self.df_procesado[col], limite_inferior, limite_superior)
        
        print(f"Datos después de limpieza: {len(self.df_procesado)} registros")
        
        return self.df_procesado
    
    def analisis_estadistico(self):
        """
        Realizar análisis estadístico descriptivo
        """
        print("\nRealizando análisis estadístico...")
        
        # Estadísticas descriptivas
        estadisticas = self.df_procesado.describe()
        
        # Correlación entre variables
        correlacion = self.df_procesado.corr()
        
        # Guardar resultados
        self.resultados['estadisticas'] = estadisticas
        self.resultados['correlacion'] = correlacion
        
        return estadisticas, correlacion
    
    def graficar_series_tiempo(self):
        """
        Crear gráficas de series de tiempo con especificaciones SVG
        """
        print("\nGenerando gráficas de series de tiempo...")
        
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
            # Guardar en formato SVG con 300 dpi
            plt.savefig(f'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/serie_individual_{variable.lower()}.svg', 
                       format='svg', dpi=300, bbox_inches='tight')
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
        # Guardar en formato SVG con 300 dpi
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/series_tiempo_completas.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Gráficas de series de tiempo guardadas en formato SVG en ../results/")
    
    def descomposicion_series_completa(self):
        """
        Realizar descomposición completa de series de tiempo incluyendo:
        - Tendencia
        - Estacionalidad
        - Variación cíclica (cuando aplica)
        - Variación irregular
        """
        print("\nRealizando descomposición completa de series de tiempo...")
        
        variables = self.df_procesado.columns
        
        for variable in variables:
            try:
                print(f"\nAnalizando {variable}...")
                
                # Descomposición clásica (tendencia + estacionalidad + residuo)
                descomposicion = seasonal_decompose(
                    self.df_procesado[variable].dropna(),
                    model='additive',
                    period=24  # Asumiendo datos horarios
                )
                
                # Extraer componentes
                tendencia = descomposicion.trend
                estacionalidad = descomposicion.seasonal
                residuo = descomposicion.resid
                
                # Calcular variación cíclica (usando filtro Hodrick-Prescott)
                try:
                    import statsmodels.api as sm
                    ciclo, tendencia_hp = sm.tsa.filters.hpfilter(
                        self.df_procesado[variable].dropna(), lamb=1600
                    )
                    variacion_ciclica = ciclo
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
                # Eliminar encabezado principal
                # fig.suptitle(f'Descomposición Completa de {variable}', fontsize=12, fontweight='bold')
                
                # Serie original
                axes[0].plot(self.df_procesado[variable].index, self.df_procesado[variable], 
                            color='black', linewidth=1, alpha=0.8, marker='o', markersize=2)
                axes[0].set_title('Serie Original', fontweight='bold')
                axes[0].set_ylabel(etiqueta_y, fontsize=10)  # Usar etiqueta con unidades
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
                
                # Guardar en formato SVG con 200 dpi
                ruta_svg = f'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/descomposicion_completa_{variable.lower()}.svg'
                plt.savefig(ruta_svg, format='svg', dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Descomposición guardada: {ruta_svg}")
                
                # Guardar resultados numéricos
                resultados_componentes = {
                    'tendencia': tendencia,
                    'estacionalidad': estacionalidad,
                    'variacion_ciclica': variacion_ciclica,
                    'variacion_irregular': variacion_irregular,
                    'descomposicion_original': descomposicion
                }
                
                self.resultados[f'descomposicion_completa_{variable}'] = resultados_componentes
                
                # Estadísticas de componentes
                print(f"  Estadísticas de componentes:")
                print(f"    Tendencia - Media: {tendencia.mean():.2f}, Std: {tendencia.std():.2f}")
                print(f"    Estacionalidad - Media: {estacionalidad.mean():.2f}, Std: {estacionalidad.std():.2f}")
                print(f"    Variación Cíclica - Media: {variacion_ciclica.mean():.2f}, Std: {variacion_ciclica.std():.2f}")
                print(f"    Variación Irregular - Media: {variacion_irregular.mean():.2f}, Std: {variacion_irregular.std():.2f}")
                
            except Exception as e:
                print(f"  ✗ Error en descomposición de {variable}: {e}")
        
        print("\nDescomposiciones completas guardadas en formato SVG en ../results/")
    
    def prueba_estacionariedad(self):
        """
        Realizar prueba de Dickey-Fuller aumentada para estacionariedad
        """
        print("\nRealizando prueba de estacionariedad (ADF)...")
        
        resultados_adf = {}
        
        for variable in self.df_procesado.columns:
            try:
                resultado = adfuller(self.df_procesado[variable].dropna())
                
                resultados_adf[variable] = {
                    'estadistico': resultado[0],
                    'p_valor': resultado[1],
                    'valores_criticos': resultado[4],
                    'es_estacionaria': resultado[1] < 0.05
                }
                
                print(f"{variable}:")
                print(f"  Estadístico ADF: {resultado[0]:.4f}")
                print(f"  P-valor: {resultado[1]:.4f}")
                print(f"  Estacionaria: {'Sí' if resultado[1] < 0.05 else 'No'}")
                print()
                
            except Exception as e:
                print(f"Error en prueba ADF para {variable}: {e}")
        
        self.resultados['prueba_adf'] = resultados_adf
        
        return resultados_adf
    
    def analisis_correlacion_cruzada(self):
        """
        Analizar correlaciones cruzadas entre variables
        """
        print("\nAnalizando correlaciones cruzadas...")
        
        # Matriz de correlación
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
        # Guardar en formato SVG con 300 dpi
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/matriz_correlacion.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de correlación interactiva
        fig = px.imshow(correlacion, 
                        text_auto=True, 
                        aspect="auto",
                        title="Matriz de Correlación Interactiva")
        fig.write_html('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/matriz_correlacion_interactive.html')
        
        print("Análisis de correlación guardado en c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/")
        
        return correlacion
    
    def modelo_arima_pm25_simple(self):
        """
        Versión simplificada del modelo ARIMA para PM25 sin variables exógenas
        para ejecución rápida
        """
        print("\nDesarrollando modelo ARIMA simplificado para PM25...")
        
        try:
            # Variable objetivo
            serie_pm25 = self.df_procesado['PM25'].dropna().astype(float)
            
            # Usar muestra más pequeña para acelerar
            sample_size = min(10000, len(serie_pm25))  # Solo 10,000 registros
            serie_pm25_sample = serie_pm25.iloc[:sample_size]
            
            # Dividir datos
            split_point = int(len(serie_pm25_sample) * 0.8)
            y_train = serie_pm25_sample.iloc[:split_point]
            y_test = serie_pm25_sample.iloc[split_point:]
            
            # Modelo ARIMA simple sin variables exógenas
            modelo = ARIMA(y_train, order=(1,1,1))
            
            print("  Ajustando modelo ARIMA simple...")
            modelo_ajustado = modelo.fit()
            
            # Realizar predicciones
            print("  Realizando predicciones...")
            predicciones = modelo_ajustado.forecast(steps=len(y_test))
            
            # Calcular métricas
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test, predicciones)
            mae = mean_absolute_error(y_test, predicciones)
            r2 = r2_score(y_test, predicciones)
            
            # Crear gráfica
            plt.figure(figsize=(12, 6))
            plt.gca().set_facecolor('white')
            
            plt.plot(y_train.index, y_train, 
                    label='Datos Entrenamiento', color='black', 
                    linewidth=1, alpha=0.7, marker='o', markersize=2)
            plt.plot(y_test.index, y_test, 
                    label='Datos Reales', color='black', 
                    linewidth=1.5, marker='s', markersize=3)
            plt.plot(y_test.index, predicciones, 
                    label='Predicciones PM25', color='black', 
                    linestyle='--', marker='^', markersize=4, alpha=0.8)
            
            etiqueta_y = ETIQUETAS_VARIABLES['PM25']
            
            plt.title('Modelo ARIMA Simple para PM25', fontsize=12, fontweight='bold')
            plt.xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            plt.ylabel(etiqueta_y, fontsize=10)
            plt.legend(fontsize=10, loc='best')
            
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/modelo_arima_pm25_simple.svg', 
                       format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nModelo ARIMA simple para PM25 ajustado:")
            print(f"  Métricas de evaluación:")
            print(f"    MSE: {mse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    R²: {r2:.4f}")
            print(f"    RMSE: {np.sqrt(mse):.4f}")
            print(f"  AIC: {modelo_ajustado.aic:.2f}")
            print(f"  BIC: {modelo_ajustado.bic:.2f}")
            
            return modelo_ajustado, predicciones
            
        except Exception as e:
            print(f"Error ajustando modelo ARIMA simple para PM25: {e}")
            return None, None
    
    def modelo_arima_pm25_con_exogenas(self):
        """
        Desarrollar modelo ARIMA para PM25 considerando variables exógenas:
        - Temperatura
        - Humedad Relativa
        - O3
        - CO
        - Tiempo (componentes estacionales)
        """
        print("\nDesarrollando modelo ARIMA para PM25 con variables exógenas...")
        
        try:
            # Variable objetivo
            serie_pm25 = self.df_procesado['PM25'].dropna().astype(float)
            
            # Variables exógenas
            variables_exogenas = ['Temperatura', 'Humedad_Relativa', 'O3', 'CO']
            exogenas = self.df_procesado[variables_exogenas].loc[serie_pm25.index].dropna().astype(float)
            
            # Asegurar que todas las series tengan la misma longitud
            min_len = min(len(serie_pm25), len(exogenas))
            serie_pm25 = serie_pm25.iloc[:min_len]
            exogenas = exogenas.iloc[:min_len]
            
            # Crear componentes de tiempo como variables exógenas
            tiempo_index = pd.to_datetime(serie_pm25.index)
            exogenas = exogenas.copy()
            exogenas['hora'] = tiempo_index.hour
            exogenas['dia_semana'] = tiempo_index.dayofweek
            exogenas['mes'] = tiempo_index.month
            
            # Crear variables dummy para componentes estacionales
            exogenas = pd.get_dummies(exogenas, columns=['hora', 'dia_semana', 'mes'], drop_first=True)
            
            # Dividir datos en entrenamiento y prueba (usar menos datos para acelerar)
            split_point = int(len(serie_pm25) * 0.7)  # Reducir a 70% entrenamiento
            # Usar solo una muestra de datos para acelerar el proceso
            sample_size = min(50000, len(serie_pm25))  # Limitar a 50,000 registros
            serie_pm25_sample = serie_pm25.iloc[:sample_size]
            exogenas_sample = exogenas.iloc[:sample_size]
            
            # Recalcular split point con la muestra
            split_point = int(len(serie_pm25_sample) * 0.7)
            
            # Convertir a numpy arrays para evitar problemas de dtype
            y_train = np.array(serie_pm25_sample.iloc[:split_point], dtype=float)
            y_test = np.array(serie_pm25_sample.iloc[split_point:], dtype=float)
            X_train = np.array(exogenas_sample.iloc[:split_point], dtype=float)
            X_test = np.array(exogenas_sample.iloc[split_point:], dtype=float)
            
            # Crear DataFrames para mantener índices
            y_train_df = pd.Series(y_train, index=serie_pm25_sample.index[:split_point])
            y_test_df = pd.Series(y_test, index=serie_pm25_sample.index[split_point:])
            X_train_df = pd.DataFrame(X_train, index=serie_pm25_sample.index[:split_point], 
                                   columns=exogenas_sample.columns)
            X_test_df = pd.DataFrame(X_test, index=serie_pm25_sample.index[split_point:], 
                                  columns=exogenas_sample.columns)
            
            # Ajustar modelo ARIMA con variables exógenas (SARIMAX) - Simplificado
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Modelo SARIMAX simplificado para acelerar el proceso
            modelo = SARIMAX(y_train_df, 
                            exog=X_train_df,
                            order=(1,1,1),  # Simplificado (p,d,q)
                            seasonal_order=(0,1,1,24),  # Simplificado - sin componente P estacional
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=True)  # Simplificar diferenciación
            
            print("  Ajustando modelo SARIMAX optimizado...")
            # Usar método de optimización más rápido
            modelo_ajustado = modelo.fit(disp=False, method='lbfgs', maxiter=50)
            
            # Realizar predicciones
            print("  Realizando predicciones...")
            predicciones = modelo_ajustado.get_forecast(steps=len(y_test_df), exog=X_test_df)
            
            # Calcular métricas de evaluación
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_test_df, predicciones.predicted_mean)
            mae = mean_absolute_error(y_test_df, predicciones.predicted_mean)
            r2 = r2_score(y_test_df, predicciones.predicted_mean)
            
            # Crear gráfica de resultados
            plt.figure(figsize=(15, 8))
            plt.gca().set_facecolor('white')
            
            # Datos de entrenamiento
            plt.plot(y_train_df.index, y_train_df, 
                    label='Datos Entrenamiento', color='black', 
                    linewidth=1, alpha=0.7, marker='o', markersize=2)
            
            # Datos de prueba
            plt.plot(y_test_df.index, y_test_df, 
                    label='Datos Reales', color='black', 
                    linewidth=1.5, marker='s', markersize=3)
            
            # Predicciones
            plt.plot(y_test_df.index, predicciones.predicted_mean, 
                    label='Predicciones PM25', color='black', 
                    linestyle='--', marker='^', markersize=4, alpha=0.8)
            
            # Configurar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES['PM25']
            
            plt.title('Modelo ARIMA-SARIMAX para PM25 con Variables Exógenas', 
                     fontsize=12, fontweight='bold')
            plt.xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            plt.ylabel(etiqueta_y, fontsize=10)
            plt.legend(fontsize=10, loc='best')
            
            # Sin grid pero mantener ejes horizontal e inferior
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Guardar en formato SVG con 300 dpi
            plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/modelo_arima_pm25_exogenas.svg', 
                       format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Análisis de importancia de variables exógenas
            print("\n  Análisis de importancia de variables exógenas:")
            importancia = modelo_ajustado.params.dropna()
            for var, coef in importancia.items():
                if var != 'sigma2':  # Excluir varianza del ruido
                    print(f"    {var}: {coef:.4f}")
            
            # Guardar resultados
            resultados_pm25 = {
                'modelo': modelo_ajustado,
                'predicciones': predicciones,
                'y_train': y_train_df,
                'y_test': y_test_df,
                'X_train': X_train_df,
                'X_test': X_test_df,
                'metricas': {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': np.sqrt(mse)
                },
                'importancia_variables': importancia
            }
            
            self.resultados['modelo_arima_pm25_exogenas'] = resultados_pm25
            
            print(f"\nModelo ARIMA para PM25 con variables exógenas ajustado:")
            print(f"  Métricas de evaluación:")
            print(f"    MSE: {mse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    R²: {r2:.4f}")
            print(f"    RMSE: {np.sqrt(mse):.4f}")
            print(f"  AIC: {modelo_ajustado.aic:.2f}")
            print(f"  BIC: {modelo_ajustado.bic:.2f}")
            
            return modelo_ajustado, predicciones, resultados_pm25
            
        except Exception as e:
            print(f"Error ajustando modelo ARIMA para PM25 con variables exógenas: {e}")
            return None, None, None
    
    def modelo_arima(self, variable='CO'):
        """
        Ajustar modelo ARIMA a una variable específica
        """
        print(f"\nAjustando modelo ARIMA para {variable}...")
        
        try:
            # Preparar datos
            serie = self.df_procesado[variable].dropna()
            
            # Ajustar modelo ARIMA simple (p,d,q) = (1,1,1)
            modelo = ARIMA(serie, order=(1,1,1))
            modelo_ajustado = modelo.fit()
            
            # Predicciones
            predicciones = modelo_ajustado.forecast(steps=24)  # 24 pasos adelante
            
            # Graficar resultados
            plt.figure(figsize=(12, 6))
            plt.gca().set_facecolor('white')
            
            # Datos reales con marcador
            plt.plot(serie.index[-100:], serie.values[-100:], 
                     label='Datos Reales', color='black', linewidth=1.5, marker='o', markersize=3)
            
            # Predicciones con marcador diferente
            plt.plot(pd.date_range(start=serie.index[-1], periods=25, freq='H')[1:],
                    predicciones, label='Predicciones ARIMA', color='black', 
                    linestyle='--', marker='s', markersize=4)
            
            # Configurar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            
            plt.title(f'Modelo ARIMA - {variable}', fontsize=12, fontweight='bold')
            plt.xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            plt.ylabel(etiqueta_y, fontsize=10)
            plt.legend(fontsize=10, loc='best')
            
            # Sin grid pero mantener ejes horizontal e inferior
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            # Guardar en formato SVG con 300 dpi
            plt.savefig(f'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/modelo_arima_{variable.lower()}.svg', 
                       format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar modelo y resultados
            self.resultados[f'modelo_arima_{variable}'] = {
                'modelo': modelo_ajustado,
                'predicciones': predicciones,
                'aic': modelo_ajustado.aic,
                'bic': modelo_ajustado.bic
            }
            
            print(f"Modelo ARIMA para {variable} ajustado:")
            print(f"  AIC: {modelo_ajustado.aic:.2f}")
            print(f"  BIC: {modelo_ajustado.bic:.2f}")
            
            return modelo_ajustado, predicciones
            
        except Exception as e:
            print(f"Error ajustando modelo ARIMA para {variable}: {e}")
            return None, None
    
    def generar_agrupaciones_analiticas(self):
        """
        Generar agrupaciones de gráficas para análisis comparativo
        """
        if self.df_procesado is None:
            print("Error: No hay datos procesados. Ejecute limpiar_datos() primero.")
            return
        
        print("\nGenerando agrupaciones analíticas...")
        
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
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/agrupacion_contaminantes.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
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
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/agrupacion_meteorologicas.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Agrupación 3: Comparación Normalizada
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fig3.patch.set_facecolor('white')
        
        # Normalizar todas las variables
        df_normalizado = (self.df_procesado - self.df_procesado.min()) / (self.df_procesado.max() - self.df_procesado.min())
        
        for i, variable in enumerate(df_normalizado.columns):
            marcador = marcadores[i % len(marcadores)]
            ax3.plot(df_normalizado.index, df_normalizado[variable], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=variable)
        
        ax3.set_title('Comparación Normalizada de Todas las Variables', fontsize=12, fontweight='bold')
        ax3.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax3.set_ylabel('Valor Normalizado (0-1)', fontsize=10)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/agrupacion_normalizada_completa.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Agrupaciones analíticas guardadas")
    
    def generar_reporte(self):
        """
        Generar reporte resumen del análisis
        """
        print("\nGenerando reporte de análisis...")
        
        reporte = f"""
# Reporte de Análisis de Series de Tiempo Ambientales

## Resumen de Datos
- **Registros analizados**: {len(self.df_procesado)}
- **Rango temporal**: {self.df_procesado.index.min()} a {self.df_procesado.index.max()}
- **Variables analizadas**: {list(self.df_procesado.columns)}

## Estadísticas Descriptivas
{self.resultados['estadisticas'].to_string()}

## Pruebas de Estacionariedad
"""
        
        for variable, resultado in self.resultados['prueba_adf'].items():
            reporte += f"\n### {variable}\n"
            reporte += f"- Estacionaria: {'Sí' if resultado['es_estacionaria'] else 'No'}\n"
            reporte += f"- P-valor: {resultado['p_valor']:.4f}\n"
        
        # Guardar reporte
        with open('c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/reporte_analisis.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("Reporte guardado en c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/reporte_analisis.txt")
    
    def ejecutar_analisis_completo(self):
        """
        Ejecutar el análisis completo de series de tiempo
        """
        print("="*60)
        print("ANÁLISIS COMPLETO DE SERIES DE TIEMPO AMBIENTALES")
        print("="*60)
        
        # 1. Cargar datos
        self.cargar_datos()
        
        # 2. Limpiar datos
        self.limpiar_datos()
        
        # 3. Análisis estadístico
        self.analisis_estadistico()
        
        # 4. Graficar series de tiempo
        self.graficar_series_tiempo()
        
        # 5. Descomposición completa de series
        self.descomposicion_series_completa()
        
        # 6. Prueba de estacionariedad
        self.prueba_estacionariedad()
        
        # 7. Análisis de correlación cruzada
        self.analisis_correlacion_cruzada()
        
        # 8. Modelado ARIMA (para CO como ejemplo)
        self.modelo_arima('CO')
        
        # 9. Modelado ARIMA para PM25 - versión simple primero
        self.modelo_arima_pm25_simple()
        
        # 10. Generar agrupaciones analíticas
        self.generar_agrupaciones_analiticas()
        
        # 11. Generar reporte
        self.generar_reporte()
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("Todos los resultados han sido guardados en la carpeta 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/'")

def main():
    """
    Función principal para ejecutar el análisis
    """
    # Ruta al archivo de datos
    ruta_datos = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/DATA/DATA_LIMPIA.csv'
    
    # Crear instancia del analizador
    analizador = AnalisisSeriesTiempo(ruta_datos)
    
    # Ejecutar análisis completo
    analizador.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()
