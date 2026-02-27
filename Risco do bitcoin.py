# ==============================
# ANÁLISE SIMPLES DE RISCO DO BITCOIN
# ==============================

# 1) Importação das bibliotecas
import yfinance as yf          # Baixar dados financeiros
import pandas as pd            # Trabalhar com tabelas
import numpy as np             # Cálculos matemáticos
import matplotlib.pyplot as plt  # Gráficos


# 2) Baixando dados históricos do Bitcoin
# BTC-USD = Bitcoin em dólar
btc = yf.download("BTC-USD", start="2018-01-01")

# Mantemos apenas o preço de fechamento
btc = btc[["Close"]]

# Renomeamos para facilitar
btc = btc[["Close"]]
btc.columns = ["preco"] 
btc["preco"] = btc["preco"].astype(float)   


# 3) Cálculo do retorno diário
# retorno = (preço atual - preço anterior) / preço anterior
btc["retorno"] = btc["preco"].pct_change()

# Remove o primeiro valor (NaN)
btc.dropna(inplace=True)


# 4) Métricas de risco e retorno
retorno_medio_diario = btc["retorno"].mean()
volatilidade_diaria = btc["retorno"].std()

# Transformando em valores anuais (aprox. 252 dias úteis)
retorno_anual = retorno_medio_diario * 252
volatilidade_anual = volatilidade_diaria * np.sqrt(252)

# Taxa livre de risco (exemplo: 2% ao ano)
taxa_livre_risco = 0.02

# Sharpe Ratio
sharpe = (retorno_anual - taxa_livre_risco) / volatilidade_anual


# 5) Drawdown (queda máxima)
btc["max_preco"] = btc["preco"].cummax()
btc["drawdown"] = (btc["preco"] - btc["max_preco"]) / btc["max_preco"]
drawdown_maximo = btc["drawdown"].min()


# 6) Exibindo resultados numéricos
print("====== MÉTRICAS DO BITCOIN ======")
print(f"Retorno anual médio: {retorno_anual:.2%}")
print(f"Volatilidade anual: {volatilidade_anual:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Drawdown máximo: {drawdown_maximo:.2%}")


# 7) Gráfico do preço do Bitcoin
plt.figure()
plt.plot(btc["preco"])
plt.title("Preço do Bitcoin")
plt.xlabel("Data")
plt.ylabel("Preço (USD)")
plt.show()


# 8) Gráfico do Drawdown
plt.figure()
plt.plot(btc["drawdown"])
plt.title("Drawdown do Bitcoin")
plt.xlabel("Data")
plt.ylabel("Drawdown")
plt.show()


# 9) Simulação de Monte Carlo
dias = 365
simulacoes = 500

retornos_simulados = np.random.normal(
    retorno_medio_diario,
    volatilidade_diaria,
    (dias, simulacoes)
)

precos_simulados = (1 + retornos_simulados).cumprod(axis=0)

plt.figure()
plt.plot(precos_simulados)
plt.title("Simulação de Monte Carlo - Bitcoin (1 ano)")
plt.xlabel("Dias")
plt.ylabel("Crescimento relativo")
plt.show()