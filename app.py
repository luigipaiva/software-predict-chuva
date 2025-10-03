import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

# --- BLOCO DE FUNÇÕES DE ANÁLISE (MOTOR REDE NEURAL LSTM) ---

def criar_features(df_com_precip):
    df_copy = df_com_precip.copy()
    df_copy['ano'] = df_copy.index.year
    df_copy['mês'] = df_copy.index.month
    for i in range(1, 13):
        df_copy[f'lag_{i}'] = df_copy['precip'].shift(i)
    return df_copy

@st.cache_data
def executar_validacao_lstm(_df_features_completas):
    ano_validacao = 2024
    treino = _df_features_completas[_df_features_completas.index.year < ano_validacao]
    validacao = _df_features_completas[_df_features_completas.index.year == ano_validacao]

    if validacao.empty: return None, None, None
    
    X_treino_df = treino.drop('precip', axis=1)
    y_treino_df = treino[['precip']]
    X_validacao_df = validacao.drop('precip', axis=1)
    y_validacao_real = validacao['precip'].values
    
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_treino_scaled = scaler_X.fit_transform(X_treino_df)
    y_treino_scaled = scaler_y.fit_transform(y_treino_df)
    X_validacao_scaled = scaler_X.transform(X_validacao_df)
    
    X_treino_reshaped = X_treino_scaled.reshape((X_treino_scaled.shape[0], 1, X_treino_scaled.shape[1]))
    X_validacao_reshaped = X_validacao_scaled.reshape((X_validacao_scaled.shape[0], 1, X_validacao_scaled.shape[1]))
    
    modelo_lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(X_treino_reshaped.shape[1], X_treino_reshaped.shape[2])),
        Dense(1)
    ])
    modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')
    modelo_lstm.fit(X_treino_reshaped, y_treino_scaled, epochs=50, batch_size=32, verbose=0)
    
    previsao_scaled = modelo_lstm.predict(X_validacao_reshaped)
    previsao_val = scaler_y.inverse_transform(previsao_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_validacao_real, previsao_val))
    r2 = r2_score(y_validacao_real, previsao_val)
    metricas = {'RMSE': rmse, 'R²': r2}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(validacao.index, y_validacao_real, label=f'Reais ({ano_validacao})', color='blue', marker='o')
    ax.plot(validacao.index, previsao_val, label=f'Previstos (LSTM) ({ano_validacao})', color='orange', linestyle='--')
    ax.set_title(f'Validação da Rede Neural: Previsão vs. Real para {ano_validacao}'); ax.legend(); ax.grid(True)
    
    return metricas, fig, ano_validacao

def gerar_previsao_futura_lstm(df_features_completas, ano_alvo):
    treino = df_features_completas[df_features_completas.index.year < ano_alvo]
    X_treino_df = treino.drop('precip', axis=1)
    y_treino_df = treino[['precip']]

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_treino_scaled = scaler_X.fit_transform(X_treino_df)
    y_treino_scaled = scaler_y.fit_transform(y_treino_df)
    
    X_treino_reshaped = X_treino_scaled.reshape((X_treino_scaled.shape[0], 1, X_treino_scaled.shape[1]))

    modelo_lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(X_treino_reshaped.shape[1], X_treino_reshaped.shape[2])),
        Dense(1)
    ])
    modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')
    modelo_lstm.fit(X_treino_reshaped, y_treino_scaled, epochs=50, batch_size=32, verbose=0)
    
    historico_recente = df_features_completas['precip'].copy()
    previsoes = []
    datas_futuras = pd.date_range(start=f'{ano_alvo}-01-01', periods=12, freq='MS')

    for data in datas_futuras:
        df_para_prever = criar_features(historico_recente.to_frame(name='precip')).tail(1).drop('precip', axis=1)
        X_para_prever_scaled = scaler_X.transform(df_para_prever)
        X_para_prever_reshaped = X_para_prever_scaled.reshape((1, 1, X_para_prever_scaled.shape[1]))
        
        pred_scaled = modelo_lstm.predict(X_para_prever_reshaped)
        pred_proximo_mes = scaler_y.inverse_transform(pred_scaled)[0][0]
        previsoes.append(pred_proximo_mes)
        historico_recente.loc[data] = pred_proximo_mes
        
    return pd.Series(previsoes, index=datas_futuras)

# --- BLOCO DA INTERFACE (A APLICAÇÃO FINAL) ---
st.set_page_config(page_title="Software de Previsão", layout="wide")
st.title("Software de Previsão de Chuva (Motor Rede Neural LSTM) 🧠")

arquivo_enviado = st.file_uploader("Carregue seu arquivo CSV de dados históricos", type=["csv"])

if arquivo_enviado is not None:
    try:
        df = pd.read_csv(arquivo_enviado, delimiter=';', decimal=',')
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        dados_completos = df['precip']
        st.success("Arquivo carregado com sucesso!")
        
        df_features_completas = criar_features(dados_completos.to_frame(name='precip'))
        df_features_completas.dropna(inplace=True)

        ano_alvo_previsao = 2026

        if st.button(f"Gerar Previsão para {ano_alvo_previsao}"):
            with st.spinner("Treinando Rede Neural... Isso pode levar vários minutos, por favor aguarde."):
                previsao = gerar_previsao_futura_lstm(df_features_completas, ano_alvo_previsao)
                
                fig_previsao, ax = plt.subplots(figsize=(14, 7))
                ax.plot(dados_completos, label='Dados Históricos')
                ax.plot(previsao, label=f'Previsão LSTM para {ano_alvo_previsao}', color='orange', linestyle='--')
                ax.set_title(f'Previsão de Precipitação para {ano_alvo_previsao}'); ax.legend(); ax.grid(True)
                ax.set_xlim(pd.Timestamp(f'{dados_completos.index.year.max() - 6}-01-01'), pd.Timestamp(f'{ano_alvo_previsao}-12-31'))

            st.success("Previsão concluída!")
            st.pyplot(fig_previsao)
            st.write(f"Previsão para os 12 meses de {ano_alvo_previsao} (em mm):")
            st.dataframe(previsao.to_frame(name="Precipitação Prevista (mm)"))

        with st.expander("Ver Performance do Modelo (Validação)"):
            st.write("Aqui, testamos a acurácia da Rede Neural prevendo o ano de 2024 e comparando com a realidade.")
            if st.button("Executar Validação da Rede Neural"):
                with st.spinner("Executando validação... Esta operação é intensiva e pode demorar alguns minutos."):
                    metricas, fig_validacao, ano_validado = executar_validacao_lstm(df_features_completas)
                
                if metricas:
                    st.success(f"Validação para o ano de {ano_validado} concluída!")
                    st.pyplot(fig_validacao)
                    st.subheader("Métricas de Performance da Rede Neural")
                    col1, col2 = st.columns(2)
                    col1.metric("R² (Aderência do Modelo)", f"{metricas['R²']:.2f}")
                    col2.metric("RMSE (Erro Médio)", f"{metricas['RMSE']:.2f} mm")
                else:
                    st.error("Não foi possível executar a validação.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")