
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import read_csv_smart, derive_common_fields, percent

st.set_page_config(page_title="Painel de Indicadores de Saúde", page_icon="🩺", layout="wide")

st.markdown("""
# 🩺 Painel de Indicadores de Saúde – Perfil dos Pacientes
Este painel é um *template moderno* baseado no seu dashboard. Carregue seus CSVs na barra lateral para ver as visualizações.
""")

with st.sidebar:
    st.header("⚙️ Configurações")
    pacientes_file = st.file_uploader("Base principal de internações (CSV)", type=["csv"])
    procedimentos_file = st.file_uploader("Procedimentos (CSV)", type=["csv"])
    uti_file = st.file_uploader("UTI / CTI (CSV)", type=["csv"])
    st.markdown("---")
    st.caption("Dica: use export de consultas (2019–2025) nos três arquivos.")

@st.cache_data(show_spinner=False)
def load_data(p_file, proc_file, uti_file):
    dfs = {}
    if p_file: 
        dfp = read_csv_smart(p_file); dfs["pac"] = derive_common_fields(dfp)
    else:
        dfs["pac"] = pd.DataFrame()
    if proc_file: 
        dfs["proc"] = read_csv_smart(proc_file)
    else:
        dfs["proc"] = pd.DataFrame()
    if uti_file: 
        dfs["uti"] = read_csv_smart(uti_file)
    else:
        dfs["uti"] = pd.DataFrame()
    return dfs

dfs = load_data(pacientes_file, procedimentos_file, uti_file)
pac, proc, uti = dfs["pac"], dfs["proc"], dfs["uti"]

if pac.empty:
    st.info("⬅️ Carregue ao menos a base de internações para começar.")
    st.stop()

# ----- Filters -----
with st.sidebar:
    anos = sorted(pac["ANO"].dropna().unique().tolist())
    ano_sel = st.multiselect("Ano", anos, default=anos)
    sexos = pac["SEXO"].dropna().unique().tolist()
    sexo_sel = st.multiselect("Sexo", sexos, default=sexos)
    raca = pac.get("RACA_COR", pd.Series(dtype=str)).dropna().unique().tolist()
    raca_sel = st.multiselect("Raça/Cor", raca, default=raca)
    st.markdown("---")
    carater = pac.get("CARATER_ATENDIMENTO", pd.Series(dtype=str)).dropna().unique().tolist()
    carater_sel = st.multiselect("Caráter do Atendimento", carater, default=carater)

df = pac.copy()
if ano_sel: df = df[df["ANO"].isin(ano_sel)]
if sexo_sel: df = df[df["SEXO"].isin(sexo_sel)]
if raca_sel and "RACA_COR" in df.columns: df = df[df["RACA_COR"].isin(raca_sel)]
if carater_sel and "CARATER_ATENDIMENTO" in df.columns: df = df[df["CARATER_ATENDIMENTO"].isin(carater_sel)]

# ----- KPI ROW -----
col1, col2, col3, col4 = st.columns(4)
total_pac = df["ID_PACIENTE"].nunique() if "ID_PACIENTE" in df.columns else len(df)
internacoes = len(df)
estabs = df["UNIDADE_ADMISSAO"].nunique() if "UNIDADE_ADMISSAO" in df.columns else None
proceds = len(proc) if not proc.empty else None

col1.metric("👥 Pacientes únicos", f"{total_pac:,}".replace(",", "."))
col2.metric("🛏️ Internações", f"{internacoes:,}".replace(",", "."))
col3.metric("🏥 Estabelecimentos", estabs if estabs is not None else "—")
col4.metric("🧰 Procedimentos", proceds if proceds is not None else "—")

# ----- Row: Sexo & Caráter -----
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if "SEXO" in df.columns:
        s = df["SEXO"].value_counts().reset_index()
        s.columns = ["SEXO","QTD"]
        fig = px.pie(s, names="SEXO", values="QTD", hole=.45, title="Distribuição por Sexo")
        st.plotly_chart(fig, use_container_width=True)
with c2:
    if "CARATER_ATENDIMENTO" in df.columns:
        s = df["CARATER_ATENDIMENTO"].value_counts().reset_index()
        s.columns = ["CARÁTER","QTD"]
        fig = px.bar(s, x="CARÁTER", y="QTD", title="Caráter do Atendimento", text_auto='.2s')
        st.plotly_chart(fig, use_container_width=True)
with c3:
    if "RACA_COR" in df.columns:
        s = df["RACA_COR"].value_counts().reset_index()
        s.columns = ["Raça/Cor","QTD"]
        fig = px.bar(s, x="Raça/Cor", y="QTD", title="Raça e Cor", text_auto='.2s')
        st.plotly_chart(fig, use_container_width=True)

# ----- Age Pyramid -----
if "FAIXA_ETARIA" in df.columns and "SEXO" in df.columns:
    st.subheader("Pirâmide Etária")
    g = df.groupby(["FAIXA_ETARIA","SEXO"]).size().reset_index(name="QTD")
    # pivot to male negative values
    g["QTD_ADJ"] = g.apply(lambda r: -r["QTD"] if str(r["SEXO"]).upper().startswith("M") else r["QTD"], axis=1)
    fig = px.bar(g, x="QTD_ADJ", y="FAIXA_ETARIA", color="SEXO", orientation="h")
    fig.update_layout(bargap=0.05, xaxis_title="Pacientes (M negativo / F positivo)")
    st.plotly_chart(fig, use_container_width=True)

# ----- LOS & Mortalidade -----
st.subheader("Tempo de Internação e Desfechos")
c4, c5, c6 = st.columns(3)
with c4:
    if "LOS_dias" in df.columns:
        fig = px.histogram(df, x="LOS_dias", nbins=40, title="Distribuição do Tempo de Internação (dias)")
        st.plotly_chart(fig, use_container_width=True)
with c5:
    if "DATA_OBITO" in df.columns:
        mort = df["DATA_OBITO"].notna().mean()*100
        st.metric("⚰️ Mortalidade hospitalar (%)", f"{mort:.2f}%")
with c6:
    if "LOS_UTI_dias" in df.columns:
        fig = px.histogram(df.dropna(subset=["LOS_UTI_dias"]), x="LOS_UTI_dias", nbins=30, title="Tempo de UTI (dias)")
        st.plotly_chart(fig, use_container_width=True)

# ----- Geografia (por Município/UF se houver) -----
st.subheader("Distribuição Geográfica (Residência)")
geo_cols = [c for c in ["UF","MUNICIPIO_RESIDENCIA","CIDADE_MORADIA"] if c in df.columns]
if geo_cols:
    c_geo1, c_geo2 = st.columns([1,1])
    geo_col = st.selectbox("Coluna de localidade disponível:", geo_cols, index=0)
    g = df.groupby(geo_col).size().reset_index(name="Pacientes")
    fig = px.bar(g, x=geo_col, y="Pacientes", title=f"Pacientes por {geo_col}", text_auto='.2s')
    c_geo1.plotly_chart(fig, use_container_width=True)
    c_geo2.dataframe(g.sort_values("Pacientes", ascending=False), use_container_width=True)

# ----- Procedimentos & CID -----
st.subheader("Procedimentos e CID-10")
cols = st.columns(2)
with cols[0]:
    if not proc.empty:
        top_proc = proc.groupby(proc.columns[proc.columns.str.contains("PROCED", case=False, regex=True)][0]).size()
        top_proc = top_proc.sort_values(ascending=False).head(20).reset_index()
        top_proc.columns = ["Procedimento","QTD"]
        fig = px.bar(top_proc, x="QTD", y="Procedimento", orientation="h", title="Top 20 procedimentos")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Carregue a base de procedimentos para ver o ranking.")
with cols[1]:
    cid_cols = [c for c in df.columns if "CID" in c.upper() or "CID10" in c.upper()]
    if cid_cols:
        c = cid_cols[0]
        top_cid = df.groupby(c).size().sort_values(ascending=False).head(20).reset_index()
        top_cid.columns = ["CID-10","QTD"]
        fig = px.bar(top_cid, x="QTD", y="CID-10", orientation="h", title="Top 20 CID-10 (principal)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Inclua uma coluna de CID-10 na base de internações para este gráfico.")

st.markdown("—")
st.caption("Template criado por você + ChatGPT. Personalize nas páginas para novas análises (UTI, Readmissão 30d, etc.).")
