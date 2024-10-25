
# Análisis de Riesgo Financiero de Apple (AAPL)

En este proyecto, realicé un análisis  del riesgo financiero de las acciones de Apple (AAPL), aplicando tres métodos principales de cálculo del Value at Risk (VaR), backtesting y expected shortfall:
- VaR Histórico
- VaR Paramétrico (Normal)
- Simulaciones de Monte Carlo

El objetivo de este análisis es evaluar y visualizar el riesgo potencial de las acciones de Apple bajo diferentes modelos de análisis de riesgo.

Puedes acceder al cuaderno de Jupyter completo [aquí](https://github.com/CalebSamaniego/analisis-riesgo-financiero/blob/main/Análisis%20del%20Riesgo%20Financiero%20de%20Apple%20(AAPL)%20usando%20Value%20at%20Risk%20(VaR)%20y%20Expected%20Shortfall%20(ES).ipynb).

![Gráfico VaR](https://github.com/CalebSamaniego/analisis-riesgo-financiero/blob/main/VaR.PNG)

---

## Dataset

- Precios históricos de las acciones de Apple (AAPL) obtenidos en Yahoo finance.
- Variables utilizadas:
  - Precio de Cierre Ajustado
  - Precio de Apertura, Máximo, Mínimo, Cierre
  - Volumen de Transacciones
  - Retornos Diarios Calculados
- Periodo de tiempo: 2015-2023.

---

## Métodos de Análisis

### 1. VaR Histórico
El VaR histórico se basa en los datos históricos de los retornos para calcular el percentil del 5%, que indica el riesgo de pérdida en un solo día.

### 2. VaR Paramétrico
El VaR paramétrico asume que los retornos siguen una distribución normal, y el cálculo se realiza utilizando la media y la desviación estándar de los retornos.

### 3. Simulaciones de Monte Carlo
Utilizando la distribución normal de los retornos diarios, se generan simulaciones aleatorias para estimar el VaR. También se experimentó con distribuciones t de Student para obtener estimaciones más precisas de riesgos extremos.

---

## Resultados
Para quienes buscan invertir a largo plazo y tienen una tolerancia al riesgo media-alta, AAPL parece una opción viable. Sin embargo, es importante estar preparados para posibles pérdidas durante momentos de alta volatilidad, especialmente si surgen eventos inesperados en el mercado.
- **VaR Histórico al 95%**: -2.93%
- **VaR Paramétrico al 95%**: -3.03%
- **VaR Monte Carlo al 95%**: -3.62%.

---
---

## Requisitos

- Python 3.8+
- Librerías necesarias:
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - Statsmodels
  - yfinance
