@echo off
echo Activando entorno virtual para Analisis de Series de Tiempo...
call venv\Scripts\activate
echo.
echo ✅ Entorno virtual activado!
echo 📦 Python: 
python --version
echo.
echo 🚀 Para ejecutar el analisis:
echo    python notebooks\analisis_series_tiempo.py
echo.
echo 🎨 Para ejecutar el visualizador:
echo    python notebooks\visualizador_graficas.py
echo.
echo 💡 Para salir del entorno:
echo    deactivate
echo.
cmd /k
