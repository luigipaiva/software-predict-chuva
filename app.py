import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# --- BLOCO DE FUNÇÕES DE ANÁLISE ---

def criar_features(df_com_precip):
    """Cria features de tempo e lag a partir da série temporal."""
    df_copy = df_com_precip.copy()
    df_copy['ano'] = df_copy.index.year
    df_copy['mês'] = df_copy.index.month
    for i in range(1, 13):
        df_copy[f'lag_{i}'] = df_copy['precip'].shift(i)
    return df_copy

@st.cache_data
def executar_validacao_xgb(_df_features_completas):
    """Executa o backtesting no último ano completo para gerar as métricas de performance."""
    ano_validacao = 2024
    
    # Dividir o DataFrame de features (que já não tem NaNs)
    treino = _df_features_completas[_df_features_completas.index.year < ano_validacao]
    validacao = _df_features_completas[_df_features_completas.index.year == ano_validacao]

    if validacao.empty:
        return None, None, None
    
    X_treino = treino.drop('precip', axis=1)
    y_treino = treino['precip']
    X_validacao = validacao.drop('precip', axis=1)
    y_validacao_real = validacao['precip']
    
    modelo_xgb_val = xgb.XGBRegressor(n_estimators=100, random_state=42)
    modelo_xgb_val.fit(X_treino, y_treino)
    previsao_val = modelo_xgb_val.predict(X_validacao)
    
    rmse = np.sqrt(mean_squared_error(y_validacao_real, previsao_val))
    r2 = r2_score(y_validacao_real, previsao_val)
    metricas = {'RMSE': rmse, 'R²': r2}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_validacao_real.index, y_validacao_real.values, label=f'Valores Reais ({ano_validacao})', color='blue', marker='o')
    ax.plot(y_validacao_real.index, previsao_val, label=f'Valores Previstos (XGBoost) ({ano_validacao})', color='purple', linestyle='--')
    ax.set_title(f'Validação do XGBoost: Previsão vs. Real para {ano_validacao}')
    ax.legend()
    ax.grid(True)
    
    return metricas, fig, ano_validacao

def gerar_previsao_futura_xgb(df_features_completas, ano_alvo):
    """Treina o modelo com os dados disponíveis e prevê o ano alvo."""
    treino = df_features_completas[df_features_completas.index.year < ano_alvo]
    
    X_treino = treino.drop('precip', axis=1)
    y_treino = treino['precip']
    
    modelo_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
    modelo_xgb.fit(X_treino, y_treino)
    
    # Para prever, precisamos criar as features para as datas futuras iterativamente
    datas_futuras = pd.date_range(start=f'{ano_alvo}-01-01', periods=12, freq='MS')
    
    # Começa com o histórico que o modelo conhece
    historico_recente = treino['precip'].copy()
    previsoes = []

    for data in datas_futuras:
        # Cria features para a última linha do histórico
        df_para_prever = criar_features(historico_recente.to_frame(name='precip')).tail(1)
        X_para_prever = df_para_prever.drop('precip', axis=1)
        
        # Faz a previsão para o próximo mês
        pred_proximo_mes = modelo_xgb.predict(X_para_prever)[0]
        previsoes.append(pred_proximo_mes)
        
        # Adiciona a previsão ao histórico para que ela seja usada no cálculo do próximo lag
        historico_recente.loc[data] = pred_proximo_mes
        
    return pd.Series(previsoes, index=datas_futuras)

# --- BLOCO DA INTERFACE ---
st.set_page_config(page_title="Software de Previsão", layout="wide")
st.title("Software de Previsão de Chuva (Motor XGBoost) 🚀")

arquivo_enviado = st.file_uploader("Carregue seu arquivo CSV de dados históricos", type=["csv"])

if arquivo_enviado is not None:
    try:
        df = pd.read_csv(arquivo_enviado, delimiter=';', decimal=',')
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        dados_completos = df['precip']
        st.success("Arquivo carregado com sucesso!")
        
        # --- LÓGICA DE PREPARAÇÃO CENTRALIZADA ---
        # Criamos as features UMA VEZ com o dataset completo
        df_features_completas = criar_features(dados_completos.to_frame(name='precip'))
        df_features_completas.dropna(inplace=True) # Remove NaNs do início da série

        ano_alvo_previsao = 2026

        if st.button(f"Gerar Previsão para {ano_alvo_previsao}"):
            with st.spinner("Treinando modelo XGBoost e gerando previsão..."):
                previsao = gerar_previsao_futura_xgb(df_features_completas, ano_alvo_previsao)
                
                fig_previsao, ax = plt.subplots(figsize=(14, 7))
                ax.plot(dados_completos, label='Dados Históricos')
                ax.plot(previsao, label=f'Previsão XGBoost para {ano_alvo_previsao}', color='purple', linestyle='--')
                ax.set_title(f'Previsão de Precipitação para {ano_alvo_previsao}')
                ax.legend(); ax.grid(True)
                ax.set_xlim(pd.Timestamp(f'{dados_completos.index.year.max() - 6}-01-01'), pd.Timestamp(f'{ano_alvo_previsao}-12-31'))

            st.success("Previsão concluída!")
            st.pyplot(fig_previsao)
            st.write(f"Previsão para os 12 meses de {ano_alvo_previsao} (em mm):")
            st.dataframe(previsao.to_frame(name="Precipitação Prevista (mm)"))

        with st.expander("Ver Performance do Modelo (Validação)"):
            st.write("Aqui, testamos a acurácia do XGBoost prevendo o ano de 2024 e comparando com a realidade.")
            if st.button("Executar Validação XGBoost"):
                with st.spinner("Executando validação..."):
                    # Passamos o DataFrame com features já criado para a função
                    metricas, fig_validacao, ano_validado = executar_validacao_xgb(df_features_completas)
                
                if metricas:
                    st.success(f"Validação para o ano de {ano_validado} concluída!")
                    st.pyplot(fig_validacao)
                    st.subheader("Métricas de Performance do XGBoost")
                    col1, col2 = st.columns(2)
                    col1.metric("R² (Aderência do Modelo)", f"{metricas['R²']:.2f}")
                    col2.metric("RMSE (Erro Médio)", f"{metricas['RMSE']:.2f} mm")
                else:
                    st.error("Não foi possível executar a validação. Verifique se os dados para 2024 estão completos.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")