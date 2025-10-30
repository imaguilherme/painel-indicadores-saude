import streamlit as st, pandas as pd, plotly.express as px
from utils import read_csv_smart, derive_common_fields

st.title("📈 Série Temporal de Internações e Óbitos")

pac_file = st.sidebar.file_uploader("Base de internações (CSV)", type=["csv"], key="st1")
if not pac_file:
    st.info("Carregue a base para ver as séries.")
    st.stop()

df = derive_common_fields(read_csv_smart(pac_file))
if "YM" not in df.columns:
    st.warning("Não foi possível derivar o Ano-Mês (coluna DATA_INTERNACAO ausente).")
    st.stop()

serie = df.groupby("YM").agg(
    Internacoes=("YM","size"),
    Obitos=("DATA_OBITO", lambda s: s.notna().sum())
).reset_index()

fig1 = px.line(serie, x="YM", y=["Internacoes","Obitos"], markers=True, title="Internações vs Óbitos por mês")
st.plotly_chart(fig1, use_container_width=True)