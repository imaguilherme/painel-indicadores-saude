import streamlit as st, pandas as pd, plotly.express as px
from utils import read_csv_smart

st.title("🧪 Procedimentos – Ranking e Séries")

proc_file = st.sidebar.file_uploader("Base de Procedimentos (CSV)", type=["csv"], key="proc1")
if not proc_file:
    st.info("Carregue a base de procedimentos.")
    st.stop()

df = read_csv_smart(proc_file)

# Try to identify the columns
name_col = None
for c in df.columns:
    if "PROCED" in c.upper() and not any(x in c.upper() for x in ["COD","CODE"]):
        name_col = c; break
if not name_col:
    st.error("Não identifiquei a coluna com o nome do procedimento (ex.: 'PROCEDIMENTO').")
    st.stop()

st.subheader("Top 30 procedimentos")
top = df[name_col].value_counts().head(30).reset_index()
top.columns = ["Procedimento","QTD"]
fig = px.bar(top, x="QTD", y="Procedimento", orientation="h")
st.plotly_chart(fig, use_container_width=True)