# Análisis de Series de Tiempo - Datos Ambientales

## Descripción General

Este proyecto realiza un análisis completo de series de tiempo de datos ambientales, incluyendo monitoreo de calidad del aire y variables meteorológicas. El análisis procesa datos de múltiples sensores para identificar patrones, tendencias, estacionalidad y relaciones entre diferentes variables ambientales.

## Variables Analizadas

- **Monóxido de Carbono (CO)**: Gas tóxico medido en ppm
- **Ozono (O3)**: Gas contaminante a nivel del suelo
- **Material Particulado PM10**: Partículas menores a 10 micrómetros
- **Material Particulado PM2.5**: Partículas menores a 2.5 micrómetros
- **Humedad Relativa**: Porcentaje de humedad en el aire
- **Temperatura**: Temperatura ambiental en grados Celsius

## Estructura del Proyecto

```
ANALISIS_SERIES_TIEMPO_10022026/
├── DATA/
│   └── DATA_LIMPIA.csv          # Datos brutos de sensores
├── notebooks/
│   └── analisis_series_tiempo.py # Script principal de análisis
├── results/                     # Gráficos y reportes generados
├── src/                         # Código fuente adicional
├── requirements.txt             # Dependencias Python
├── README.md                    # Este archivo
└── LICENSE                      # Licencia del proyecto
```

## Características del Análisis

### 1. Preprocesamiento de Datos
- Carga y validación de datos desde archivo CSV
- Conversión de timestamps a formato datetime
- Manejo de valores faltantes mediante interpolación lineal
- Detección y tratamiento de outliers usando método IQR
- Ordenamiento cronológico de datos

### 2. Análisis Estadístico Descriptivo
- Estadísticas básicas (media, mediana, desviación estándar, cuartiles)
- Matriz de correlación entre variables
- Visualización de distribuciones
- Análisis de valores extremos

### 3. Análisis de Series de Tiempo
- **Visualización Temporal**: Gráficos de series de tiempo para cada variable
- **Descomposición**: Separación en tendencia, estacionalidad y componentes residuales
- **Estacionariedad**: Prueba de Dickey-Fuller aumentada (ADF)
- **Correlación Cruzada**: Análisis de relaciones entre variables

### 4. Modelado Predictivo
- Implementación de modelos ARIMA para predicción
- Evaluación de modelos usando criterios AIC/BIC
- Generación de pronósticos a corto plazo
- Validación de predicciones

### 5. Visualizaciones Generadas
- Gráficos estáticos en formato PNG (Matplotlib/Seaborn)
- Gráficos interactivos en formato HTML (Plotly)
- Mapas de calor de correlación
- Gráficos de descomposición temporal
- Visualizaciones de predicciones

## Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Clonar o descargar el proyecto
cd ANALISIS_SERIES_TIEMPO_10022026

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales
- **pandas**: Manipulación y análisis de datos
- **numpy**: Computación numérica
- **matplotlib**: Visualización de datos estática
- **seaborn**: Visualización estadística
- **plotly**: Gráficos interactivos
- **scipy**: Herramientas científicas y estadísticas
- **statsmodels**: Modelado estadístico y series de tiempo
- **scikit-learn**: Aprendizaje automático
- **jupyter**: Notebooks interactivos

## Uso

### Ejecución del Análisis Completo

```bash
# Navegar a la carpeta de notebooks
cd notebooks

# Ejecutar el script principal
python analisis_series_tiempo.py
```

### Uso Programático

```python
from notebooks.analisis_series_tiempo import AnalisisSeriesTiempo

# Crear instancia del analizador
analizador = AnalisisSeriesTiempo('../DATA/DATA_LIMPIA.csv')

# Ejecutar análisis completo
analizador.ejecutar_analisis_completo()

# O ejecutar pasos específicos
analizador.cargar_datos()
analizador.limpiar_datos()
analizador.graficar_series_tiempo()
```

## Resultados Generados

El análisis genera los siguientes archivos en la carpeta `results/`:

### Gráficos Estáticos (PNG)
- `series_tiempo_completas.png`: Todas las variables en el tiempo
- `descomposicion_{variable}.png`: Descomposición por variable
- `matriz_correlacion.png`: Mapa de calor de correlaciones
- `modelo_arima_{variable}.png`: Predicciones del modelo ARIMA

### Gráficos Interactivos (HTML)
- `series_tiempo_interactive.html`: Series de tiempo interactivas
- `matriz_correlacion_interactive.html`: Matriz de correlación interactiva

### Reportes
- `reporte_analisis.txt`: Resumen completo del análisis

## Metodología

### 1. Limpieza de Datos
- **Interpolación Lineal**: Para valores faltantes esporádicos
- **Método IQR**: Para detección y tratamiento de outliers
- **Validación**: Verificación de integridad de datos

### 2. Análisis de Estacionariedad
- **Prueba ADF**: Hipótesis nula de no estacionariedad
- **Nivel de Significancia**: α = 0.05
- **Interpretación**: p-valor < 0.05 indica serie estacionaria

### 3. Descomposición Temporal
- **Modelo Aditivo**: Y(t) = Tendencia + Estacionalidad + Residuo
- **Período**: 24 (asumiendo datos horarios)
- **Validación**: Inspección visual de componentes

### 4. Modelado ARIMA
- **Orden (1,1,1)**: Configuración simple pero efectiva
- **Validación**: Criterios de información AIC/BIC
- **Horizonte**: 24 pasos adelante (predicción a 24 horas)

## Interpretación de Resultados

### Estadísticas Descriptivas
- **Media**: Valor promedio de cada variable
- **Desviación Estándar**: Variabilidad de los datos
- **Cuartiles**: Distribución y dispersión

### Correlaciones
- **Positiva Fuerte (>0.7)**: Relación directa importante
- **Negativa Fuerte (<-0.7)**: Relación inversa importante
- **Débil (-0.3 a 0.3)**: Poca relación lineal

### Estacionariedad
- **Series Estacionarias**: Propiedades estadísticas constantes en tiempo
- **Series No Estacionarias**: Requieren diferenciación para modelado

### Descomposición
- **Tendencia**: Comportamiento a largo plazo
- **Estacionalidad**: Patrones periódicos regulares
- **Residuo**: Componente aleatorio no explicado

## Aplicaciones Prácticas

### Monitoreo Ambiental
- Identificación de patrones de contaminación
- Detección de eventos anómalos
- Evaluación de calidad del aire

### Toma de Decisiones
- Alertas tempranas de contaminación elevada
- Planificación de actividades al aire libre
- Políticas de control ambiental

### Investigación
- Estudio de relaciones entre variables
- Modelado de fenómenos atmosféricos
- Desarrollo de sistemas de predicción

## Limitaciones y Consideraciones

### Datos
- Calidad depende de sensores originales
- Interpolación puede introducir sesgos
- Outliers pueden representar eventos reales

### Modelos
- ARIMA asume linealidad
- Predicciones a corto plazo solo
- Requiere validación continua

### Interpretación
- Correlación no implica causalidad
- Factores externos no considerados
- Variabilidad espacial no incluida

## Extensiones Futuras

### Análisis Avanzado
- Modelos LSTM/Deep Learning
- Análisis de frecuencia (FFT)
- Detección de anomalías automática

### Variables Adicionales
- Dirección y velocidad del viento
- Presión atmosférica
- Radiación solar

### Integración
- API de datos en tiempo real
- Dashboard web interactivo
- Sistema de alertas

## Contribuciones

Las contribuciones son bienvenidas. Sugerencias:
- Mejorar algoritmos de limpieza
- Añadir nuevos modelos predictivos
- Optimizar visualizaciones
- Extender documentación

## Licencia

Este proyecto está licenciado bajo los términos descritos en el archivo `LICENSE`.

## Contacto

Para preguntas o sugerencias sobre este análisis, por favor contactar al equipo de desarrollo.

---

**Nota**: Este análisis es para fines educativos y de investigación. Para aplicaciones críticas de monitoreo ambiental, se recomienda validación adicional y consulta con expertos en el campo.
