

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

st.title("🎓 Previsor de Notas com Inteligência Artificial")
st.write("Veja como a rede neural aprende com os seus dados que **mais horas de estudo levam a notas melhores**!")

# 1. Dados fornecidos pelo usuário via Pandas DataFrame
estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# Exibindo os dados na tabela e gráfico do Streamlit
st.write("📊 **Dados de Treinamento:**")
st.dataframe(estudos)
st.line_chart(estudos, x='horas', y='notas')

# Separando as variáveis (Features e Labels) para o TensorFlow
# (Nota: Convertemos para float para a rede neural processar corretamente)
horas = estudos['horas'].values.astype(float)
notas = estudos['notas'].values.astype(float)

# 2. Criando a Rede Neural (1 neurônio para Regressão Linear)
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# 3. Treinando o modelo (a IA "aprende" com os seus dados)
with st.spinner("Treinando a rede neural..."):
    model.fit(horas, notas, epochs=500, verbose=0)
st.success("✨ Modelo treinado com sucesso!")

# 4. Interação com o Aluno: Prever nota com base nas horas
st.subheader("Simulador de Desempenho")
horas_usuario = st.slider("Quantas horas você pretende estudar?", 0.0, 12.0, 5.0)

# Fazendo a previsão com o TensorFlow
nota_prevista = model.predict(np.array([horas_usuario]))[0][0]
nota_ajustada = min(max(nota_prevista, 0.0), 10.0)  # Limita entre 0 e 10

st.metric(label="Nota Prevista pela IA", value=f"{nota_ajustada:.2f}")

# 5. Descobrindo quantas horas são necessárias para tirar 10
pesos = model.layers[0].get_weights()
w, b = pesos[0][0], pesos[1][0]
if w > 0:
    horas_para_dez = (10.0 - b) / w
    st.info( f'{max(horas_para_dez):.1f} ')



