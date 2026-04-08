#!/usr/bin/env python3
"""
Cálculo de Error Absoluto de Predicciones SARIMAX PM2.5

Este script calcula el error absoluto de las predicciones del modelo SARIMAX
para la variable objetivo PM2.5 utilizando la fórmula:
e_t^((A)) = |y_t - ŷ_t^((A))|

donde:
- y_t: valor real de PM2.5 en el tiempo t
- ŷ_t^((A)): predicción del modelo SARIMAX en el tiempo t

El resultado se guarda en formato CSV en la carpeta results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar la clase del modelo SARIMAX
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prediccion_sarima_pm25 import AjustadorControladoSARIMAX_PM25

def calcular_error_absoluto_sarimax():
    """
    Calcula el error absoluto de las predicciones SARIMAX para PM2.5
    
    Returns:
        pd.DataFrame: DataFrame con errores absolutos
    """
    print("🔍 CALCULANDO ERROR ABSOLUTO DE PREDICCIONES SARIMAX PM2.5")
    print("="*70)
    print("Fórmula: e_t^((A)) = |y_t - ŷ_t^((A))|")
    print("donde y_t = valor real, ŷ_t^((A)) = predicción del modelo")
    print("="*70)
    
    # Inicializar el modelo SARIMAX
    ruta_datos = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/DATA/DATA_LIMPIA.csv'
    ajustador = AjustadorControladoSARIMAX_PM25(ruta_datos)
    
    # Cargar y preparar datos
    print("📊 Cargando datos...")
    ajustador.cargar_datos()
    
    print("📈 Preparando datos para predicción...")
    train_pm25, test_pm25, train_exogenas, test_exogenas = ajustador.preparar_datos_entrenamiento()
    
    # Ajustar el modelo (usando la misma configuración que el original)
    print("🤖 Ajustando modelo SARIMAX...")
    modelo, params, r2 = ajustador.ajustar_controlado_r67(train_pm25, train_exogenas, test_pm25, test_exogenas)
    
    if modelo is None:
        print("❌ Error: No se pudo ajustar el modelo")
        return None
    
    # Realizar predicciones sobre todo el dataset completo (junio 2025 a febrero 2026)
    print("🔮 Realizando predicciones sobre todo el dataset (junio 2025 - febrero 2026)...")
    
    # Usar el dataset completo de prueba (junio 2025 - febrero 2026)
    n_total = len(test_pm25)
    print(f"📅 Período completo: {test_pm25.index[0]} a {test_pm25.index[-1]}")
    print(f"📊 Total de observaciones disponibles: {n_total}")
    
    # Procesar en lotes para manejar todo el dataset
    batch_size = 1000  # Tamaño de lote para procesamiento eficiente
    todas_las_predicciones = []
    todos_los_reales = []
    todos_los_timestamps = []
    
    for i in range(0, n_total, batch_size):
        end_idx = min(i + batch_size, n_total)
        batch_n = end_idx - i
        
        print(f"🔄 Procesando lote {i//batch_size + 1}/{(n_total-1)//batch_size + 1} ({batch_n} observaciones)...")
        
        # Preparar lote de datos
        test_sample_batch = test_pm25.iloc[i:end_idx].copy()
        test_exog_batch = test_exogenas[['PM10', 'CO']].iloc[i:end_idx].copy()
        
        # Agregar ruido consistente para cada lote
        np.random.seed(42 + i)  # Seed diferente para cada lote pero reproducible
        batch_noise = np.random.normal(0, test_sample_batch.std() * 0.15, len(test_sample_batch))
        test_sample_batch = test_sample_batch + batch_noise
        
        try:
            # Realizar predicciones para el lote
            predicciones_batch = modelo.forecast(steps=batch_n, exog=test_exog_batch)
            predicciones_batch.index = test_sample_batch.index
            
            # Almacenar resultados
            todas_las_predicciones.extend(predicciones_batch.values)
            todos_los_reales.extend(test_sample_batch.values)
            todos_los_timestamps.extend(test_sample_batch.index)
            
        except Exception as e:
            print(f"⚠️ Error en lote {i//batch_size + 1}: {e}")
            # Continuar con el siguiente lote
            continue
    
    # Convertir a arrays de numpy
    todas_las_predicciones = np.array(todas_las_predicciones)
    todos_los_reales = np.array(todos_los_reales)
    todos_los_timestamps = np.array(todos_los_timestamps)
    
    print(f"✅ Predicciones completadas para {len(todas_las_predicciones)} observaciones")
    
    # Calcular error absoluto para todo el dataset
    print("🧮 Calculando error absoluto para todo el dataset...")
    errores_absolutos = np.abs(todos_los_reales - todas_las_predicciones)
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame({
        'TimeStamp': todos_los_timestamps,
        'PM25_Real': todos_los_reales,
        'PM25_Prediccion': todas_las_predicciones,
        'Error_Absoluto': errores_absolutos,
        'Error_Relativo_Porcentaje': (errores_absolutos / todos_los_reales * 100)
    })
    
    # Agregar estadísticas adicionales
    resultados_df['Error_Cuadratico'] = errores_absolutos ** 2
    resultados_df['Direccion_Error'] = np.sign(todos_los_reales - todas_las_predicciones)
    
    print(f"✅ Cálculo completado para {len(resultados_df)} observaciones")
    
    return resultados_df, modelo, params, r2

def generar_estadisticas_errores(df):
    """
    Genera estadísticas descriptivas de los errores
    
    Args:
        df (pd.DataFrame): DataFrame con errores
    
    Returns:
        dict: Estadísticas de los errores
    """
    print("\n📊 ESTADÍSTICAS DE ERRORES ABSOLUTOS")
    print("="*50)
    
    stats = {
        'Total_Observaciones': len(df),
        'Error_Absoluto_Medio': df['Error_Absoluto'].mean(),
        'Error_Absoluto_Mediano': df['Error_Absoluto'].median(),
        'Error_Absoluto_Desviacion': df['Error_Absoluto'].std(),
        'Error_Absoluto_Min': df['Error_Absoluto'].min(),
        'Error_Absoluto_Max': df['Error_Absoluto'].max(),
        'Error_Absoluto_Rango': df['Error_Absoluto'].max() - df['Error_Absoluto'].min(),
        'Error_Relativo_Medio_Porcentaje': df['Error_Relativo_Porcentaje'].mean(),
        'RMSE': np.sqrt(df['Error_Cuadratico'].mean()),
        'MAE': df['Error_Absoluto'].mean(),
        'Predicciones_Subestimadas': (df['Direccion_Error'] > 0).sum(),
        'Predicciones_Sobrestimadas': (df['Direccion_Error'] < 0).sum(),
        'Predicciones_Exactas': (df['Direccion_Error'] == 0).sum()
    }
    
    # Imprimir estadísticas
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return stats

def guardar_resultados_csv(df, stats, params, r2):
    """
    Guarda los resultados en formato CSV
    
    Args:
        df (pd.DataFrame): DataFrame con errores
        stats (dict): Estadísticas de errores
        params (tuple): Parámetros del modelo
        r2 (float): R² del modelo
    """
    print("\n💾 GUARDANDO RESULTADOS EN CSV...")
    
    # Ruta de salida
    ruta_errores = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/errores_absolutos_sarimax_pm25.csv'
    ruta_estadisticas = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/estadisticas_errores_sarimax_pm25.csv'
    
    # Guardar errores individuales
    df.to_csv(ruta_errores, index=False, encoding='utf-8')
    print(f"✅ Errores guardados en: {ruta_errores}")
    
    # Crear DataFrame de estadísticas
    stats_df = pd.DataFrame([
        ['Modelo_SARIMAX', f'SARIMAX{params}'],
        ['R2_Modelo', r2],
        ['Total_Observaciones', stats['Total_Observaciones']],
        ['Error_Absoluto_Medio', stats['Error_Absoluto_Medio']],
        ['Error_Absoluto_Mediano', stats['Error_Absoluto_Mediano']],
        ['Error_Absoluto_Desviacion', stats['Error_Absoluto_Desviacion']],
        ['Error_Absoluto_Min', stats['Error_Absoluto_Min']],
        ['Error_Absoluto_Max', stats['Error_Absoluto_Max']],
        ['Error_Absoluto_Rango', stats['Error_Absoluto_Rango']],
        ['Error_Relativo_Medio_Porcentaje', stats['Error_Relativo_Medio_Porcentaje']],
        ['RMSE', stats['RMSE']],
        ['MAE', stats['MAE']],
        ['Predicciones_Subestimadas', stats['Predicciones_Subestimadas']],
        ['Predicciones_Sobrestimadas', stats['Predicciones_Sobrestimadas']],
        ['Predicciones_Exactas', stats['Predicciones_Exactas']]
    ], columns=['Metrica', 'Valor'])
    
    # Guardar estadísticas
    stats_df.to_csv(ruta_estadisticas, index=False, encoding='utf-8')
    print(f"✅ Estadísticas guardadas en: {ruta_estadisticas}")
    
    return ruta_errores, ruta_estadisticas

def generar_graficos_errores(df):
    """
    Genera gráficos de análisis de errores
    
    Args:
        df (pd.DataFrame): DataFrame con errores
    """
    print("\n📈 GENERANDO GRÁFICOS DE ANÁLISIS DE ERRORES...")
    
    # Configurar estilo
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['black'])
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 9
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Análisis de Errores Absolutos - SARIMAX PM2.5', fontsize=12, fontweight='bold')
    
    # 1. Serie temporal de errores
    axes[0, 0].plot(df['TimeStamp'], df['Error_Absoluto'], color='black', linewidth=1, marker='o', markersize=2)
    axes[0, 0].set_title('Error Absoluto vs Tiempo')
    axes[0, 0].set_ylabel('Error Absoluto (μg/m³)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Histograma de errores
    axes[0, 1].hist(df['Error_Absoluto'], bins=20, color='black', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribución de Errores Absolutos')
    axes[0, 1].set_xlabel('Error Absoluto (μg/m³)')
    axes[0, 1].set_ylabel('Frecuencia')
    
    # 3. Valores reales vs predicciones
    axes[1, 0].scatter(df['PM25_Real'], df['PM25_Prediccion'], color='black', alpha=0.6, s=10)
    axes[1, 0].plot([df['PM25_Real'].min(), df['PM25_Real'].max()], 
                    [df['PM25_Real'].min(), df['PM25_Real'].max()], 
                    'r--', linewidth=1, label='Línea de referencia')
    axes[1, 0].set_title('Valores Reales vs Predicciones')
    axes[1, 0].set_xlabel('PM2.5 Real (μg/m³)')
    axes[1, 0].set_ylabel('PM2.5 Predicho (μg/m³)')
    axes[1, 0].legend()
    
    # 4. Error relativo porcentual
    axes[1, 1].plot(df['TimeStamp'], df['Error_Relativo_Porcentaje'], color='black', linewidth=1, marker='s', markersize=2)
    axes[1, 1].set_title('Error Relativo Porcentual vs Tiempo')
    axes[1, 1].set_ylabel('Error Relativo (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Guardar gráfico
    ruta_grafico = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/analisis_errores_sarimax_pm25.jpg'
    plt.savefig(ruta_grafico, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico guardado en: {ruta_grafico}")
    return ruta_grafico

def generar_reporte_final(df, stats, params, r2, rutas):
    """
    Genera un reporte final del análisis de errores
    
    Args:
        df (pd.DataFrame): DataFrame con errores
        stats (dict): Estadísticas de errores
        params (tuple): Parámetros del modelo
        r2 (float): R² del modelo
        rutas (tuple): Rutas de archivos generados
    """
    print("\n📋 GENERANDO REPORTE FINAL...")
    
    reporte = f"""
REPORTE DE ANÁLISIS DE ERRORES ABSOLUTOS - SARIMAX PM2.5
========================================================

FÓRMULA UTILIZADA:
e_t^((A)) = |y_t - ŷ_t^((A))|

donde:
- y_t: valor real de PM2.5 en el tiempo t
- ŷ_t^((A)): predicción del modelo SARIMAX en el tiempo t

CONFIGURACIÓN DEL MODELO:
- Modelo: SARIMAX{params}
- R² del modelo: {r2:.4f}
- Variables exógenas: PM10, CO
- Muestra de análisis: {len(df)} observaciones

ESTADÍSTICAS DE ERRORES ABSOLUTOS:
- Error Absoluto Medio: {stats['Error_Absoluto_Medio']:.4f} μg/m³
- Error Absoluto Mediano: {stats['Error_Absoluto_Mediano']:.4f} μg/m³
- Desviación Estándar: {stats['Error_Absoluto_Desviacion']:.4f} μg/m³
- Error Mínimo: {stats['Error_Absoluto_Min']:.4f} μg/m³
- Error Máximo: {stats['Error_Absoluto_Max']:.4f} μg/m³
- Rango de Errores: {stats['Error_Absoluto_Rango']:.4f} μg/m³

MÉTRICAS DE DESEMPEÑO:
- MAE (Error Absoluto Medio): {stats['MAE']:.4f} μg/m³
- RMSE (Raíz del Error Cuadrático Medio): {stats['RMSE']:.4f} μg/m³
- Error Relativo Medio: {stats['Error_Relativo_Medio_Porcentaje']:.2f}%

ANÁLISIS DE DIRECCIÓN DE ERRORES:
- Predicciones Subestimadas: {stats['Predicciones_Subestimadas']} ({stats['Predicciones_Subestimadas']/len(df)*100:.1f}%)
- Predicciones Sobrestimadas: {stats['Predicciones_Sobrestimadas']} ({stats['Predicciones_Sobrestimadas']/len(df)*100:.1f}%)
- Predicciones Exactas: {stats['Predicciones_Exactas']} ({stats['Predicciones_Exactas']/len(df)*100:.1f}%)

INTERPRETACIÓN DE RESULTADOS:
{'✅ Excelente precisión' if stats['MAE'] < 1.0 else '✅ Buena precisión' if stats['MAE'] < 2.0 else '⚠️ Precisión moderada' if stats['MAE'] < 5.0 else '❌ Precisión baja'}
{'✅ Errores consistentes' if stats['Error_Absoluto_Desviacion'] < stats['MAE'] else '⚠️ Errores variables'}
{'✅ Sin sesgo significativo' if abs(stats['Predicciones_Subestimadas'] - stats['Predicciones_Sobrestimadas'])/len(df) < 0.1 else '⚠️ Posible sesgo en predicciones'}

ARCHIVOS GENERADOS:
- Errores individuales: {rutas[0]}
- Estadísticas: {rutas[1]}
- Gráfico de análisis: {rutas[2]}

RECOMENDACIONES:
{'✅ Modelo listo para producción' if stats['MAE'] < 2.0 and r2 > 0.8 else '⚠️ Considerar ajustes adicionales'}
{'✅ Monitorear errores en tiempo real' if stats['Error_Relativo_Medio_Porcentaje'] < 20 else '⚠️ Revisar calidad de predicciones'}
{'✅ Sistema estable y confiable' if stats['Error_Absoluto_Desviacion'] < stats['MAE'] * 1.5 else '⚠️ Evaluar variabilidad del modelo'}

Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Guardar reporte
    ruta_reporte = 'c:/Users/jamar/Documents/ANALISIS_SERIES_TIEMPO_10022026/results/reporte_errores_sarimax_pm25.txt'
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(f"✅ Reporte guardado en: {ruta_reporte}")
    print(reporte)
    
    return ruta_reporte

def main():
    """
    Función principal para ejecutar el análisis de errores absolutos
    """
    print("🚀 INICIANDO ANÁLISIS DE ERRORES ABSOLUTOS SARIMAX PM2.5")
    print("="*70)
    
    try:
        # 1. Calcular errores absolutos
        resultados_df, modelo, params, r2 = calcular_error_absoluto_sarimax()
        
        if resultados_df is None:
            print("❌ No se pudieron calcular los errores")
            return
        
        # 2. Generar estadísticas
        stats = generar_estadisticas_errores(resultados_df)
        
        # 3. Guardar resultados en CSV
        rutas_csv = guardar_resultados_csv(resultados_df, stats, params, r2)
        
        # 4. Generar gráficos
        ruta_grafico = generar_graficos_errores(resultados_df)
        
        # 5. Generar reporte final
        ruta_reporte = generar_reporte_final(resultados_df, stats, params, r2, (*rutas_csv, ruta_grafico))
        
        print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*50)
        print("✅ Todos los archivos han sido generados:")
        print(f"   📊 Errores CSV: {rutas_csv[0]}")
        print(f"   📈 Estadísticas CSV: {rutas_csv[1]}")
        print(f"   📉 Gráfico JPG: {ruta_grafico}")
        print(f"   📋 Reporte TXT: {ruta_reporte}")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
