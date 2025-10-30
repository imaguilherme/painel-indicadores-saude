import streamlit as st, pandas as pd, plotly.express as px
from utils import read_csv_smart

st.title("🏥 Indicadores de UTI/CTI")

uti_file = st.sidebar.file_uploader("Base de UTI/CTI (CSV)", type=["csv"], key="uti1")
if not uti_file:
    st.info("Carregue a base de UTI/CTI.")
    st.stop()

df = read_csv_smart(uti_file)
if not {"DT_ENTRADA_CTI","DT_SAIDA_CTI"}.issubset(df.columns):
    st.error("Esperadas colunas DT_ENTRADA_CTI e DT_SAIDA_CTI.")
    st.stop()

df["LOS_UTI_dias"] = (df["DT_SAIDA_CTI"] - df["DT_ENTRADA_CTI"]).dt.days
st.metric("🎯 Média de dias em UTI", f"{df['LOS_UTI_dias'].mean():.1f}")
st.metric("📊 Mediana de dias em UTI", f"{df['LOS_UTI_dias'].median():.0f}")

fig = px.histogram(df, x="LOS_UTI_dias", nbins=40, title="Distribuição do tempo em UTI (dias)")
st.plotly_chart(fig, use_container_width=True)