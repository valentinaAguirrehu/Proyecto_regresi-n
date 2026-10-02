# Informe de Laboratorio: Regresión Lineal Múltiple con CRISP-DM

**Nombre:** ________  **Curso:** Minería de Datos  **Fecha:** ________
**Enlace del cuaderno de Colab:** ________  **Enlace de la app en Vercel:** ________

> Borrador con los resultados reales de tus datos. Agrega las gráficas de la carpeta `graficas/` y reescribe con tus palabras.

## 1. Introducción y objetivo
Aplicar la fase de modelado de CRISP-DM para construir tres modelos de regresión lineal múltiple que predicen variables cuantitativas (precio del dólar, nivel de glucosa y consumo de energía), evaluarlos, exportarlos y ponerlos en una interfaz web.

## 2. Comprensión de los datos
| Dataset | Filas | Variable dependiente | Variables independientes |
|---|---|---|---|
| dolar_data.csv | 500 | Precio_Dolar (COP) | Dia, Inflacion, Tasa_interes |
| glucosa_data.csv | 2000 | Nivel_Glucosa (mg/dL) | Edad, IMC, Actividad_Fisica |
| energia_data.csv | 10000 | Consumo_Energia (kWh) | Temperatura, Hora, Dia_Semana |

(Insertar `head()`, `describe()` y los mapas de correlación `*_correlacion.png`.)

## 3. Preparación de los datos
- **Tipos de dato:** todas las columnas son numéricas.
- **Nulos:** 0 en los tres datasets.
- **Duplicados:** 0 en los tres datasets.
- **Outliers (IQR):** dólar: 4 en Inflación y 7 en Tasa_interes; glucosa: 16 en IMC y 3 en Nivel_Glucosa; energía: 90 en Temperatura y 66 en Consumo_Energia (≤ 1,4 %). Se conservaron por ser pocos y plausibles; eliminarlos sin justificación habría sesgado el modelo. (Insertar `*_boxplots.png`.)
- **Partición:** 80 % entrenamiento y 20 % prueba (`random_state=42`).

## 4. Modelado
Se usó `LinearRegression` de scikit-learn. Ecuaciones obtenidas:

- **Dólar:** Precio = 3985,78 + 4,984·Dia − 870,73·Inflacion − 1,377·Tasa_interes
- **Glucosa:** Glucosa = 65,86 + 1,227·Edad + 0,933·IMC − 2,085·Actividad_Fisica
- **Energía:** Consumo = 101,29 + 9,953·Temperatura + 5,020·Hora − 3,031·Dia_Semana

## 5. Evaluación
| Ejercicio | MSE | RMSE | R² |
|---|---|---|---|
| Dólar | 2376,97 | 48,75 | 0,9963 |
| Glucosa | 233,69 | 15,29 | 0,6814 |
| Energía | 429,52 | 20,72 | 0,8968 |

## 6. Interpretación de coeficientes
Los coeficientes se interpretan *manteniendo las demás variables constantes*. Para comparar importancia se usaron coeficientes estandarizados.

**Dólar** (R² = 0,996). Cada día adicional sube el dólar en promedio 4,98 COP (estandarizado 0,995, el único que pesa de verdad). La inflación tiene signo negativo (−870 por unidad, es decir, −8,7 COP por cada 0,01) y la tasa de interés −1,38 COP por punto, pero ambos aportes son prácticamente nulos (estandarizados 0,006 y 0,001) y no tienen correlación con el precio. El precio se explica casi solo por la tendencia temporal. Un R² tan alto se debe a que `Dia` es una tendencia casi lineal; no implica que inflación o tasa de interés influyan.

**Glucosa** (R² = 0,681). Por cada año de edad la glucosa aumenta 1,23 mg/dL; por cada unidad de IMC, 0,93 mg/dL; y por cada hora semanal de ejercicio baja 2,09 mg/dL. **Mayor impacto: Edad** (estandarizado 0,787), seguida de Actividad_Fisica (−0,222) e IMC (0,136).

**Energía** (R² = 0,897). Cada °C adicional aumenta el consumo 9,95 kWh; cada hora más avanzada del día, 5,02 kWh; y cada día más avanzado de la semana lo reduce 3,03 kWh. **Mayor impacto: Temperatura** (estandarizado 0,780), luego Hora (0,544) y Dia_Semana (−0,095).

(Insertar `*_dispersion.png`, `*_real_vs_predicho.png` y `*_residuos.png`.)

## 7. Exportación y despliegue
- Modelos guardados con `joblib.dump` en `modelo_dolar.pkl`, `modelo_glucosa.pkl` y `modelo_energia.pkl`, y cargados con `joblib.load` para verificar que coinciden con la fórmula manual.
- Interfaz en Flask con selector de ejercicio, campos de entrada y predicción en pantalla. Se usa `config.json` (intercepto y coeficientes) porque scikit-learn es muy pesado para Vercel; el resultado es idéntico a `model.predict()`.
- Desplegada en Vercel: (URL y capturas).

## 8. Conclusiones y limitaciones
- Los tres modelos tienen buen ajuste; el de glucosa es el más débil (R² = 0,68), lo que sugiere factores no incluidos (genética, dieta) o relaciones no lineales.
- El modelo del dólar depende casi solo del tiempo; extrapolar a días futuros asume que la tendencia continúa.
- La regresión lineal supone linealidad, errores independientes y varianza constante. `Hora` es una variable cíclica (la 24 está junto a la 1); aquí el ajuste lineal funciona, pero se podría mejorar con transformaciones seno/coseno.
- Mejoras posibles: validación cruzada, regularización (Ridge/Lasso), modelos no lineales.

## Anexos
Cuaderno de Colab, repositorio de GitHub, capturas de la app.
