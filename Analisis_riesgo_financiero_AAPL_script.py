# ----------------------------------------
# Paquetes a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install yfinance
import yfinance as yf

# ----------------------------------------
# Descarga de los datos de AAPL (Apple)
data = yf.download("AAPL", start="2015-01-01", end="2023-01-01")
data.head()  # Nos muestra los primeros registros

# ----------------------------------------
#  Primeras filas 
print(data.head())

# Información general 
print(data.info())

# Resumen estadístico
print(data.describe())


# ----------------------------------------
# Verificamos valores nulos
print(data.isnull().sum())


# ----------------------------------------
#Columnas del DataFrame
print(data.columns)


# ----------------------------------------
# Acceso a la columna 'Adj Close' de AAPL
adj_close = data[('Adj Close', 'AAPL')]

# Tipo de dato
print(adj_close.dtype)

# Valores nulos
print(adj_close.isnull().sum())


# ----------------------------------------
# Cálculo de los retornos logarítmicos
data[('Return', 'AAPL')] = np.log(data[('Adj Close', 'AAPL')] / data[('Adj Close', 'AAPL')].shift(1))

# Resultados
print(data[[('Adj Close', 'AAPL'), ('Return', 'AAPL')]].head())


# ----------------------------------------
# Gráfico del precio de cierre ajustado
plt.figure(figsize=(10,6))
plt.plot(data['Adj Close'], label='Precio de Cierre Ajustado')
plt.title('Precio de Cierre Ajustado de AAPL (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre Ajustado (USD)')
plt.legend()
plt.show()


# ----------------------------------------
# Gráfico del volumen de transacciones
plt.figure(figsize=(10,6))
plt.plot(data['Volume'], color='orange', label='Volumen de transacción')
plt.title('Volumen de Transacción de AAPL (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Volumen de Transacción')
plt.legend()
plt.show()


# ----------------------------------------
# Calculo de la matriz de correlación
corr = data.corr()

# matriz de correlación como un mapa de calor
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()


# ----------------------------------------
# Grafico de los retornos logarítmicos a lo largo del tiempo
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'AAPL')], label='Retornos Diarios')
plt.title('Retornos Diarios de AAPL (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Logarítmico')
plt.legend()
plt.show()


# ----------------------------------------
# Grafico de la distribución de los retornos diarios
plt.figure(figsize=(10,6))
sns.histplot(data[('Return', 'AAPL')].dropna(), bins=100, kde=True)
plt.title('Distribución de los Retornos Diarios de AAPL')
plt.xlabel('Retorno Diario Logarítmico')
plt.ylabel('Frecuencia')
plt.show()


# ----------------------------------------
import scipy.stats as stats
import matplotlib.pyplot as plt

#  Q-Q plot para los retornos
stats.probplot(data[('Return', 'AAPL')], dist="norm", plot=plt)
plt.title('Q-Q Plot para los Retornos Diarios de AAPL')
plt.show()


# ----------------------------------------
# Calculo de la volatilidad (desviación estándar móvil de 30 días)
data[('Volatility', 'AAPL')] = data[('Return', 'AAPL')].rolling(window=30).std()

# Grafico de la volatilidad
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Volatility', 'AAPL')], label='Volatilidad (30 días)')
plt.title('Volatilidad de AAPL (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.legend()
plt.show()


# ----------------------------------------
# Cálculo de sesgo (asimetría) y curtosis
skewness = data[('Return', 'AAPL')].skew()
kurtosis = data[('Return', 'AAPL')].kurtosis()
print(f"Sesgo: {skewness}")
print(f"Curtosis: {kurtosis}")


# ----------------------------------------
# Cálculo del VaR Historico

var_95 = np.percentile(data[('Return', 'AAPL')].dropna(), 5)
print(f"VaR Histórico al 95%: {var_95:.2%}")


# ----------------------------------------
# Cálculo del VaR parametrico

mean = data[('Return', 'AAPL')].mean()
std_dev = data[('Return', 'AAPL')].std()
var_parametrico_95 = mean - 1.65 * std_dev
print(f"VaR Paramétrico al 95%: {var_parametrico_95:.2%}")


# ----------------------------------------
# Cálculo del VaR Montecarlo

from scipy.stats import t
df = 5  # grados de libertad, ajustable
simulations = t.rvs(df, loc=mean, scale=std_dev, size=10000)
var_montecarlo_t_95 = np.percentile(simulations, 5)
print(f"VaR Monte Carlo con t al 95%: {var_montecarlo_t_95:.2%}")


# ----------------------------------------
# Cálculo del backtesting para método histórico
violations_historic = data[('Return', 'AAPL')] < -var_95
num_violations_historic = violations_historic.sum()
print(f"Número de violaciones del VaR: {num_violations_historic}")



# ----------------------------------------
# Gráfico del backtesting para método histórico
violations_historic = data[('Return', 'AAPL')] < -var_95
plt.plot(data.index, data[('Return', 'AAPL')], label='Retornos Diarios')
plt.axhline(y=-var_95, color='r', linestyle='--', label='VaR Histórico (95%)')
plt.scatter(data.index[violations_historic], data[('Return', 'AAPL')][violations_historic], color='red', label='Violaciones')
plt.legend()
plt.show()



# ----------------------------------------
# Cálculo del backtesting para método paramétrico
violations_parametrico = data[('Return', 'AAPL')] < -var_parametrico_95
num_violations_parametrico = violations_parametrico.sum()
print(f"Número de violaciones del VaR: {num_violations_parametrico}")

# ----------------------------------------
# Gráfico del backtesting para método paramétrico
violations_parametrico = data[('Return', 'AAPL')] < -var_parametrico_95
plt.plot(data.index, data[('Return', 'AAPL')], label='Retornos Diarios')
plt.axhline(y=-var_parametrico_95, color='r', linestyle='--', label='VaR Paramétrico (95%)')
plt.scatter(data.index[violations_parametrico], data[('Return', 'AAPL')][violations_parametrico], color='red', label='Violaciones')
plt.legend()
plt.show()



# ----------------------------------------
# Cálculo del backtesting para simulación de montecarlo
violations_montecarlo = data[('Return', 'AAPL')] < -var_montecarlo_t_95
num_violations_montecarlo = violations_montecarlo.sum()
print(f"Número de violaciones del VaR: {num_violations_montecarlo}")

# ----------------------------------------
# Gráfico del backtesting para simulación de montecarlo
violations_montecarlo = data[('Return', 'AAPL')] < -var_montecarlo_t_95
plt.plot(data.index, data[('Return', 'AAPL')], label='Retornos Diarios')
plt.axhline(y=-var_montecarlo_95, color='r', linestyle='--', label='VaR Monte Carlo (95%)')
plt.scatter(data.index[violations_montecarlo], data[('Return', 'AAPL')][violations_montecarlo], color='red', label='Violaciones')
plt.legend()
plt.show()


# ----------------------------------------
# Cálculo del Expected Shortfall para VaR Histórico
ES_historic = data[('Return', 'AAPL')][data[('Return', 'AAPL')] < -var_95].mean()
print(f"Expected Shortfall (ES) Histórico al 95%: {ES_historic:.2%}")


# ----------------------------------------
# Cálculo del Expected Shortfall para VaR Paramétrico
ES_parametric = data[('Return', 'AAPL')][data[('Return', 'AAPL')] < -var_parametrico_95].mean()
print(f"Expected Shortfall (ES) Paramétrico al 95%: {ES_parametric:.2%}")


# ----------------------------------------
# Cálculo del Expected Shortfall para Monte Carlo
ES_montecarlo = np.mean(simulations[simulations < -var_montecarlo_t_95])
print(f"Expected Shortfall (ES) Monte Carlo al 95%: {ES_montecarlo:.2%}")


# ----------------------------------------
# Grafica de VaR y Expected Shortfall para los tres métodos
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'AAPL')], label='Retornos Diarios')
plt.axhline(y=var_95, color='r', linestyle='--', label='VaR Histórico (95%)')
plt.axhline(y=var_parametrico_95, color='g', linestyle='--', label='VaR Paramétrico (95%)')
plt.axhline(y=var_montecarlo_t_95, color='b', linestyle='--', label='VaR Monte Carlo (95%)')
plt.axhline(y=ES_historic, color='r', linestyle=':', label='ES Histórico (95%)')
plt.axhline(y=ES_parametric, color='g', linestyle=':', label='ES Paramétrico (95%)')
plt.axhline(y=ES_montecarlo, color='b', linestyle=':', label='ES Monte Carlo (95%)')
plt.legend()
plt.title('Comparación del VaR y Expected Shortfall')
plt.show()


