#!/usr/bin/env python3
"""
Visualizador de Gráficas - Análisis de Series de Tiempo Ambientales

Este script permite manipular y visualizar las gráficas generadas por el análisis
de series de tiempo, con opciones de personalización, filtrado y exportación.

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
import os
import glob
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo - Especificaciones detalladas
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

class VisualizadorGraficas:
    """
    Clase para manipular y visualizar gráficas del análisis de series de tiempo
    """
    
    def __init__(self, ruta_datos='c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/DATA/DATA_LIMPIA.csv', ruta_resultados='c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results'):
        """
        Inicializar el visualizador
        
        Args:
            ruta_datos (str): Ruta al archivo de datos
            ruta_resultados (str): Ruta a la carpeta de resultados
        """
        self.ruta_datos = ruta_datos
        self.ruta_resultados = ruta_resultados
        self.df = None
        self.colores_personalizados = {
            'CO': '#FF6B6B',
            'O3': '#4ECDC4', 
            'PM10': '#45B7D1',
            'PM25': '#96CEB4',
            'Humedad_Relativa': '#FFEAA7',
            'Temperatura': '#DDA0DD'
        }
        
    def cargar_datos(self):
        """
        Cargar los datos originales para visualización
        """
        print("Cargando datos para visualización...")
        
        self.df = pd.read_csv(self.ruta_datos)
        self.df['TimeStamp'] = pd.to_datetime(self.df['TimeStamp'])
        self.df.set_index('TimeStamp', inplace=True)
        self.df.sort_index(inplace=True)
        
        # Renombrar columnas
        columnas_renombradas = {
            'carbon_monoxide_Data': 'CO',
            'ozone_Data': 'O3',
            'pm10_Data': 'PM10',
            'pm25_Data': 'PM25',
            'relative_humidity_Data': 'Humedad_Relativa',
            'temperature_Data': 'Temperatura'
        }
        self.df.rename(columns=columnas_renombradas, inplace=True)
        
        # Limpiar datos básicos
        for col in self.df.columns:
            self.df[col] = self.df[col].interpolate(method='linear')
        self.df.dropna(inplace=True)
        
        print(f"Datos cargados: {len(self.df)} registros")
        return self.df
    
    def grafica_interactiva_personalizada(self, variables=None, rango_fechas=None, 
                                        titulo="Gráfica Interactiva Personalizada"):
        """
        Crear gráfica interactiva con opciones de personalización
        
        Args:
            variables (list): Lista de variables a graficar
            rango_fechas (tuple): Rango de fechas (inicio, fin)
            titulo (str): Título de la gráfica
        """
        if self.df is None:
            self.cargar_datos()
        
        # Filtrar variables
        if variables is None:
            variables = list(self.df.columns)
        
        # Filtrar por rango de fechas
        df_filtrado = self.df[variables].copy()
        if rango_fechas:
            inicio, fin = rango_fechas
            df_filtrado = df_filtrado.loc[inicio:fin]
        
        # Crear subplots con matplotlib
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars))
        fig.patch.set_facecolor('white')
        fig.suptitle(titulo, fontsize=12, fontweight='bold')
        
        # Definir marcadores diferentes para cada variable
        marcadores = ['o', 's', '^', 'v', 'D', 'p']
        
        for i, variable in enumerate(variables):
            # Usar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            
            # Graficar con marcador específico
            marcador = marcadores[i % len(marcadores)]
            axes[i].plot(df_filtrado.index, df_filtrado[variable], 
                         color='black', linewidth=1.5, marker=marcador, 
                         markersize=3, alpha=0.8, label=variable)
            
            axes[i].set_title(variable, fontsize=10, fontweight='bold')
            axes[i].set_ylabel(etiqueta_y, fontsize=10)
            axes[i].set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            
            # Sin grid pero mantener ejes horizontal e inferior
            axes[i].grid(False)
            # Solo eliminar bordes superior y derecho
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        # Añadir leyenda general
        fig.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        return fig
    
    def grafica_comparativa(self, variables, rango_fechas=None, normalizar=False):
        """
        Crear gráfica comparativa de múltiples variables
        
        Args:
            variables (list): Variables a comparar
            rango_fechas (tuple): Rango de fechas
            normalizar (bool): Si se normalizan los datos (0-1)
        """
        if self.df is None:
            self.cargar_datos()
        
        # Filtrar datos
        df_filtrado = self.df[variables].copy()
        if rango_fechas:
            inicio, fin = rango_fechas
            df_filtrado = df_filtrado.loc[inicio:fin]
        
        # Normalizar si se solicita
        if normalizar:
            df_filtrado = (df_filtrado - df_filtrado.min()) / (df_filtrado.max() - df_filtrado.min())
            titulo_suffix = " (Normalizado)"
        else:
            titulo_suffix = ""
        
        # Crear figura con matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Definir marcadores diferentes para cada variable
        marcadores = ['o', 's', '^', 'v', 'D', 'p']
        
        # Graficar cada variable con marcador diferente
        for i, variable in enumerate(variables):
            marcador = marcadores[i % len(marcadores)]
            ax.plot(df_filtrado.index, df_filtrado[variable], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=variable)
        
        # Configurar etiquetas
        ax.set_title(f'Comparación de Variables{titulo_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax.set_ylabel('Valor' if not normalizar else 'Valor Normalizado (0-1)', fontsize=10)
        
        # Añadir leyenda
        ax.legend(fontsize=10, loc='best')
        
        # Sin grid pero mantener ejes horizontal e inferior
        ax.grid(False)
        # Solo eliminar bordes superior y derecho
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def grafica_distribucion(self, variables=None, tipo='histograma'):
        """
        Crear gráficas de distribución para variables seleccionadas
        
        Args:
            variables (list): Variables a analizar
            tipo (str): Tipo de gráfica ('histograma', 'boxplot', 'violin')
        """
        if self.df is None:
            self.cargar_datos()
        
        if variables is None:
            variables = list(self.df.columns)
        
        if tipo == 'histograma':
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Distribución de Variables - Histograma', fontsize=12, fontweight='bold')
            fig.patch.set_facecolor('white')
            
            for i, variable in enumerate(variables):
                row = i // 3
                col = i % 3
                
                axes[row, col].hist(self.df[variable].dropna(), bins=30, 
                                  color='black', alpha=0.7, edgecolor='black')
                axes[row, col].set_title(variable, fontsize=10, fontweight='bold')
                
                # Usar etiquetas específicas
                etiqueta_y = ETIQUETAS_VARIABLES.get(variable, 'Valor')
                axes[row, col].set_ylabel(etiqueta_y, fontsize=10)
                axes[row, col].set_xlabel('Valor', fontsize=10)
                
                # Sin grid pero mantener ejes horizontal e inferior
                axes[row, col].grid(False)
                # Solo eliminar bordes superior y derecho
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
        
        elif tipo == 'boxplot':
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            # Crear boxplot con especificaciones
            boxprops = dict(color='black')
            whiskerprops = dict(color='black')
            capprops = dict(color='black')
            medianprops = dict(color='black', linewidth=2)
            
            bp = ax.boxplot([self.df[var].dropna() for var in variables], 
                           labels=variables, patch_artist=True,
                           boxprops=boxprops, whiskerprops=whiskerprops,
                           capprops=capprops, medianprops=medianprops)
            
            # Colorear las cajas en negro
            for patch in bp['boxes']:
                patch.set_facecolor('black')
                patch.set_alpha(0.7)
            
            ax.set_title('Distribución de Variables - Boxplot', fontsize=12, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=10)
            ax.set_xlabel('Variables', fontsize=10)
            
            # Sin grid ni bordes
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        elif tipo == 'violin':
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            # Crear violin plot con especificaciones
            violin_parts = ax.violinplot([self.df[var].dropna() for var in variables], 
                                       positions=range(1, len(variables)+1))
            
            # Colorear en negro
            for pc in violin_parts['bodies']:
                pc.set_facecolor('black')
                pc.set_alpha(0.7)
            
            # Configurar líneas
            for partname in ('cmaxes', 'cmins', 'cbars'):
                if partname in violin_parts:
                    vp = violin_parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1.5)
            
            # Configurar medianas si existen
            if 'cmedians' in violin_parts:
                vp = violin_parts['cmedians']
                vp.set_edgecolor('black')
                vp.set_linewidth(2)
            
            ax.set_xticks(range(1, len(variables)+1))
            ax.set_xticklabels(variables, fontsize=10)
            ax.set_title('Distribución de Variables - Violin Plot', fontsize=12, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=10)
            ax.set_xlabel('Variables', fontsize=10)
            
            # Sin grid ni bordes
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def grafica_correlacion_dinamica(self, metodo='pearson'):
        """
        Crear matriz de correlación dinámica con diferentes métodos
        
        Args:
            metodo (str): Método de correlación ('pearson', 'spearman', 'kendall')
        """
        if self.df is None:
            self.cargar_datos()
        
        # Calcular correlación
        corr_matrix = self.df.corr(method=metodo)
        
        # Crear heatmap con matplotlib en blanco y negro
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Usar colormap en escala de grises
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Añadir valores en las celdas
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Configurar ejes
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Añadir colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coeficiente de Correlación', rotation=270, labelpad=20)
        
        ax.set_title(f'Matriz de Correlación - Método {metodo.title()}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def grafica_series_temporales_agrupadas(self, variable, periodo='D'):
        """
        Crear gráfica de series temporales agrupadas por período
        
        Args:
            variable (str): Variable a analizar
            periodo (str): Período de agrupación ('D', 'W', 'M', 'H')
        """
        if self.df is None:
            self.cargar_datos()
        
        # Agrupar datos
        df_agrupado = self.df[variable].resample(periodo).agg(['mean', 'std', 'min', 'max'])
        
        # Crear figura con matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Media con marcadores
        ax.plot(df_agrupado.index, df_agrupado['mean'], 
               color='black', linewidth=2, marker='o', markersize=3, 
               label='Media', alpha=0.8)
        
        # Bandas de desviación estándar
        ax.fill_between(df_agrupado.index, 
                       df_agrupado['mean'] - df_agrupado['std'],
                       df_agrupado['mean'] + df_agrupado['std'],
                       color='gray', alpha=0.3, label='±1 Std Dev')
        
        # Mínimos y máximos con marcadores diferentes
        ax.plot(df_agrupado.index, df_agrupado['max'], 
               color='black', linewidth=1, marker='^', markersize=4, 
               label='Máximo', alpha=0.7)
        ax.plot(df_agrupado.index, df_agrupado['min'], 
               color='black', linewidth=1, marker='v', markersize=4, 
               label='Mínimo', alpha=0.7)
        
        # Configurar etiquetas específicas
        etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
        titulo_periodo = {
            'D': 'Diario', 'W': 'Semanal', 
            'M': 'Mensual', 'H': 'Horario'
        }.get(periodo, periodo)
        
        ax.set_title(f'{variable} - Agrupado por {titulo_periodo}', fontsize=12, fontweight='bold')
        ax.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax.set_ylabel(etiqueta_y, fontsize=10)
        
        # Añadir leyenda
        ax.legend(fontsize=10, loc='best')
        
        # Sin grid pero mantener ejes horizontal e inferior
        ax.grid(False)
        # Solo eliminar bordes superior y derecho
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generar_graficas_individuales(self):
        """
        Generar gráficas individuales para cada variable con nombres representativos
        """
        if self.df is None:
            self.cargar_datos()
        
        print("Generando gráficas individuales...")
        
        variables = self.df.columns
        titulos_representativos = {
            'CO': 'Concentración de Monóxido de Carbono',
            'O3': 'Niveles de Ozono',
            'PM10': 'Material Particulado PM10',
            'PM25': 'Material Particulado PM2.5',
            'Humedad_Relativa': 'Humedad Relativa',
            'Temperatura': 'Temperatura Ambiental'
        }
        
        for variable in variables:
            # Gráfica individual
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            # Graficar con marcador
            ax.plot(self.df.index, self.df[variable], 
                    color='black', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
            
            # Configurar etiquetas específicas
            etiqueta_y = ETIQUETAS_VARIABLES.get(variable, variable)
            titulo = titulos_representativos.get(variable, variable)
            
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
            ax.set_ylabel(etiqueta_y, fontsize=10)
            
            # Sin grid ni bordes
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            plt.tight_layout()
            
            # Guardar en SVG
            nombre_archivo = f"individual_{variable.lower()}"
            self.exportar_grafica(fig, nombre_archivo, "svg")
            plt.close()
            
            print(f"  ✓ Gráfica individual guardada: {variable}")
    
    def generar_agrupaciones_analiticas(self):
        """
        Generar agrupaciones de gráficas para análisis comparativo
        """
        if self.df is None:
            self.cargar_datos()
        
        print("Generando agrupaciones analíticas...")
        
        # Agrupación 1: Contaminantes (CO, O3, PM10, PM25)
        contaminantes = ['CO', 'O3', 'PM10', 'PM25']
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        fig1.patch.set_facecolor('white')
        
        marcadores = ['o', 's', '^', 'v']
        for i, cont in enumerate(contaminantes):
            marcador = marcadores[i % len(marcadores)]
            ax1.plot(self.df.index, self.df[cont], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=cont)
        
        ax1.set_title('Contaminantes Atmosféricos', fontsize=12, fontweight='bold')
        ax1.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax1.set_ylabel('Concentración', fontsize=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(False)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        self.exportar_grafica(fig1, "agrupacion_contaminantes", "svg")
        plt.close()
        
        # Agrupación 2: Variables Meteorológicas (Temperatura, Humedad)
        meteo = ['Temperatura', 'Humedad_Relativa']
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig2.patch.set_facecolor('white')
        
        for i, var in enumerate(meteo):
            marcador = marcadores[i % len(marcadores)]
            etiqueta_y = ETIQUETAS_VARIABLES.get(var, var)
            ax2.plot(self.df.index, self.df[var], 
                    color='black', linewidth=1.5, marker=marcador, 
                    markersize=3, alpha=0.8, label=var)
        
        ax2.set_title('Variables Meteorológicas', fontsize=12, fontweight='bold')
        ax2.set_xlabel(ETIQUETAS_VARIABLES['tiempo'], fontsize=10)
        ax2.set_ylabel('Valor', fontsize=10)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        self.exportar_grafica(fig2, "agrupacion_meteorologicas", "svg")
        plt.close()
        
        # Agrupación 3: Comparación Normalizada
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fig3.patch.set_facecolor('white')
        
        # Normalizar todas las variables
        df_normalizado = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        
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
        for spine in ax3.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        self.exportar_grafica(fig3, "agrupacion_normalizada_completa", "svg")
        plt.close()
        
        print("  ✓ Agrupaciones analíticas guardadas")
    
    def dashboard_interactivo(self):
        """
        Crear dashboard interactivo completo con widgets
        """
        if self.df is None:
            self.cargar_datos()
        
        # Crear widgets
        widget_variables = widgets.SelectMultiple(
            options=list(self.df.columns),
            value=list(self.df.columns)[:3],
            description='Variables:',
            disabled=False
        )
        
        widget_fecha_inicio = widgets.DatePicker(
            value=self.df.index.min().date(),
            description='Fecha Inicio:',
            disabled=False
        )
        
        widget_fecha_fin = widgets.DatePicker(
            value=self.df.index.max().date(),
            description='Fecha Fin:',
            disabled=False
        )
        
        widget_normalizar = widgets.Checkbox(
            value=False,
            description='Normalizar datos',
            disabled=False
        )
        
        widget_tipo_grafico = widgets.RadioButtons(
            options=['Líneas', 'Comparación', 'Distribución'],
            value='Líneas',
            description='Tipo:',
            disabled=False
        )
        
        # Función de actualización
        def actualizar_grafica(change):
            variables = list(widget_variables.value)
            fecha_inicio = widget_fecha_inicio.value
            fecha_fin = widget_fecha_fin.value
            normalizar = widget_normalizar.value
            tipo = widget_tipo_grafico.value
            
            # Convertir fechas a datetime
            inicio = pd.to_datetime(fecha_inicio)
            fin = pd.to_datetime(fecha_fin)
            
            # Crear gráfica según tipo
            if tipo == 'Líneas':
                fig = self.grafica_interactiva_personalizada(
                    variables=variables,
                    rango_fechas=(inicio, fin),
                    titulo='Gráfica de Líneas Personalizada'
                )
            elif tipo == 'Comparación':
                fig = self.grafica_comparativa(
                    variables=variables,
                    rango_fechas=(inicio, fin),
                    normalizar=normalizar
                )
            else:  # Distribución
                fig = self.grafica_distribucion(variables=variables)
            
            fig.show()
        
        # Conectar widgets a función de actualización
        widget_variables.observe(actualizar_grafica, names='value')
        widget_fecha_inicio.observe(actualizar_grafica, names='value')
        widget_fecha_fin.observe(actualizar_grafica, names='value')
        widget_normalizar.observe(actualizar_grafica, names='value')
        widget_tipo_grafico.observe(actualizar_grafica, names='value')
        
        # Mostrar widgets
        display(widgets.VBox([
            widgets.HBox([widget_variables, widget_tipo_grafico]),
            widgets.HBox([widget_fecha_inicio, widget_fecha_fin]),
            widget_normalizar
        ]))
        
        # Mostrar gráfica inicial
        actualizar_grafica(None)
    
    def exportar_grafica(self, fig, nombre_archivo, formato='svg'):
        """
        Exportar gráfica a diferentes formatos
        
        Args:
            fig: Figura de Plotly o Matplotlib
            nombre_archivo (str): Nombre del archivo sin extensión
            formato (str): Formato de exportación ('svg', 'html', 'png', 'pdf')
        """
        ruta_completa = os.path.join(self.ruta_resultados, f"{nombre_archivo}.{formato}")
        
        if formato == 'svg':
            if hasattr(fig, 'write_image'):  # Plotly
                fig.write_image(ruta_completa, width=1200, height=800, scale=3)  # 300 dpi
            else:  # Matplotlib
                fig.savefig(ruta_completa, format='svg', dpi=300, bbox_inches='tight')
        elif formato == 'html':
            fig.write_html(ruta_completa)
        elif formato == 'png':
            if hasattr(fig, 'write_image'):  # Plotly
                fig.write_image(ruta_completa, width=1200, height=800, scale=3)  # 300 dpi
            else:  # Matplotlib
                fig.savefig(ruta_completa, format='png', dpi=300, bbox_inches='tight')
        elif formato == 'pdf':
            fig.write_image(ruta_completa)
        
        print(f"Gráfica exportada: {ruta_completa}")
    
    def generar_reporte_visual(self):
        """
        Generar un reporte visual completo con múltiples gráficas en formato SVG
        """
        if self.df is None:
            self.cargar_datos()
        
        print("Generando reporte visual completo en formato SVG...")
        
        # 1. Gráfica de series temporales completas
        fig1 = self.grafica_interactiva_personalizada(
            titulo="Series de Tiempo - Variables Ambientales"
        )
        self.exportar_grafica(fig1, "reporte_series_completas", "svg")
        
        # 2. Gráfica comparativa normalizada
        fig2 = self.grafica_comparativa(
            variables=['CO', 'O3', 'PM10', 'PM25'],
            normalizar=True
        )
        self.exportar_grafica(fig2, "reporte_comparacion_normalizada", "svg")
        
        # 3. Distribución de variables
        fig3 = self.grafica_distribucion(tipo='boxplot')
        self.exportar_grafica(fig3, "reporte_distribucion", "svg")
        
        # 4. Matriz de correlación
        fig4 = self.grafica_correlacion_dinamica()
        self.exportar_grafica(fig4, "reporte_correlacion", "svg")
        
        # 5. Series agrupadas por día para cada variable
        for variable in self.df.columns:
            fig5 = self.grafica_series_temporales_agrupadas(variable, 'D')
            self.exportar_grafica(fig5, f"reporte_{variable.lower()}_diario", "svg")
        
        print("Reporte visual completo generado en formato SVG en ../results/")

def main():
    """
    Función principal con ejemplos de uso
    """
    # Crear instancia del visualizador
    visualizador = VisualizadorGraficas()
    
    # Cargar datos
    visualizador.cargar_datos()
    
    print("="*60)
    print("VISUALIZADOR DE GRÁFICAS - SERIES DE TIEMPO AMBIENTALES")
    print("="*60)
    
    # 1. Gráfica interactiva personalizada
    print("\n1. Generando gráfica interactiva personalizada...")
    fig1 = visualizador.grafica_interactiva_personalizada(
        variables=['CO', 'O3', 'PM10'],
        titulo="Variables Principales - Contaminantes"
    )
    fig1.show()
    visualizador.exportar_grafica(fig1, "ejemplo_interactivo_personalizado", "svg")
    
    # 2. Gráfica comparativa normalizada
    print("\n2. Generando gráfica comparativa normalizada...")
    fig2 = visualizador.grafica_comparativa(
        variables=['CO', 'O3', 'PM10', 'PM25'],
        normalizar=True
    )
    fig2.show()
    visualizador.exportar_grafica(fig2, "ejemplo_comparacion_normalizada", "svg")
    
    # 3. Distribución de variables
    print("\n3. Generando gráfica de distribución...")
    fig3 = visualizador.grafica_distribucion(tipo='boxplot')
    fig3.show()
    visualizador.exportar_grafica(fig3, "ejemplo_distribucion_violin", "svg")
    
    # 4. Matriz de correlación
    print("\n4. Generando matriz de correlación...")
    fig4 = visualizador.grafica_correlacion_dinamica(metodo='spearman')
    fig4.show()
    visualizador.exportar_grafica(fig4, "ejemplo_correlacion_spearman", "svg")
    
    # 5. Series temporales agrupadas
    print("\n5. Generando series temporales agrupadas...")
    fig5 = visualizador.grafica_series_temporales_agrupadas('CO', 'H')
    fig5.show()
    visualizador.exportar_grafica(fig5, "ejemplo_co_horario", "svg")
    
    # 6. Gráficas individuales
    print("\n6. Generando gráficas individuales...")
    visualizador.generar_graficas_individuales()
    
    # 7. Agrupaciones analíticas
    print("\n7. Generando agrupaciones analíticas...")
    visualizador.generar_agrupaciones_analiticas()
    
    # 8. Generar reporte visual completo
    print("\n8. Generando reporte visual completo...")
    visualizador.generar_reporte_visual()
    
    print("\n" + "="*60)
    print("VISUALIZACIONES COMPLETADAS")
    print("="*60)
    print("Todas las gráficas han sido guardadas en formato SVG en la carpeta 'results/'")
    print("\nPara usar el dashboard interactivo, ejecuta:")
    print("visualizador.dashboard_interactivo()")

if __name__ == "__main__":
    main()
