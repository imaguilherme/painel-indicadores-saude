# app.py — Perfil dos Pacientes (2019–2025) | suporta Parquet único OU 3 CSVs grandes (DuckDB)
# Requisitos: streamlit, pandas, numpy, plotly, python-dateutil, pyarrow, duckdb

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import duckdb, os, tempfile

st.set_page_config(page_title="Perfil dos Pacientes – ACC", layout="wide")

# -------------------- utilidades --------------------
def _to_dt(s):
    if pd.isna(s): return pd.NaT
    if isinstance(s, (pd.Timestamp, datetime)): return pd.to_datetime(s)
    try: return pd.to_datetime(parser.parse(str(s), dayfirst=True, fuzzy=True))
    except: return pd.NaT

def _post_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # datas
    for c in ["data_internacao","data_alta","data_obito","dthr_valida","dt_entrada_cti","dt_saida_cti","data_cirurgia_min","data_cirurgia_max"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # numéricos
    for c in ["idade","ano","ano_internacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # sexo
    if "sexo" in df.columns:
        df["sexo"] = (df["sexo"].astype(str).str.strip().str.upper()
                      .replace({"M":"Masculino","F":"Feminino","MASCULINO":"Masculino","FEMININO":"Feminino"}))
    # dias permanência
    if {"data_internacao","data_alta"}.issubset(df.columns):
        df["dias_permanencia"] = (df["data_alta"] - df["data_internacao"]).dt.days
    # faixas etárias
    if "idade" in df.columns:
        bins = [-1,0,4,11,17,24,34,44,54,64,74,84,120]
        labels = ["<1","1–4","5–11","12–17","18–24","25–34","35–44","45–54","55–64","65–74","75–84","85+"]
        df["faixa_etaria"] = pd.cut(pd.to_numeric(df["idade"], errors="coerce"), bins=bins, labels=labels, right=True)
    # dedup de eventos (AIH)
    keys = [c for c in ["codigo_internacao","prontuario_anonimo","data_internacao","data_alta"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)
    return df

@st.cache_data(show_spinner=False)
def load_parquet(file):
    return _post_load(pd.read_parquet(file))

@st.cache_resource(show_spinner=False)
def load_duckdb(csv_paths):
    """Registra 3 CSVs (EVOLUCOES, PROCED, CIDS/UTI) e cria view dataset unificada (lazy)."""
    con = duckdb.connect(database=":memory:")
    evo, proc, cti = csv_paths
    con.execute("CREATE VIEW evolu AS SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);", [evo])
    con.execute("CREATE VIEW proced AS SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);", [proc])
    con.execute("CREATE VIEW cids AS SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);", [cti])

    # normaliza chaves para lower/trim e agrega procedimentos para não explodir cardinalidade
    con.execute("""
        CREATE VIEW evolu_n AS
        SELECT
          lower(trim(prontuario_anonimo)) AS prontuario_anonimo,
          lower(trim(codigo_internacao)) AS codigo_internacao,
          *
        FROM evolu;

        CREATE VIEW cids_n AS
        SELECT
          lower(trim(prontuario_anonimo)) AS prontuario_anonimo,
          lower(trim(codigo_internacao)) AS codigo_internacao,
          *
        FROM cids;

        CREATE VIEW proc_agg AS
        SELECT
          lower(trim(codigo_internacao)) AS codigo_internacao,
          COUNT(DISTINCT codigo_procedimento) AS n_proced,
          ANY_VALUE(codigo_procedimento) AS proc_prim,
          ANY_VALUE(procedimento) AS proc_nome_prim,
          MIN(COALESCE(data_cirurgia, data_internacao)) AS data_cirurgia_min,
          MAX(COALESCE(data_cirurgia, data_internacao)) AS data_cirurgia_max
        FROM proced
        GROUP BY 1;

        CREATE VIEW dataset AS
        SELECT
          e.*,
          c.* EXCLUDE (codigo_internacao, prontuario_anonimo),
          p.*
        FROM evolu_n e
        LEFT JOIN cids_n c USING (codigo_internacao, prontuario_anonimo)
        LEFT JOIN proc_agg p USING (codigo_internacao);
    """)
    return con

def df_from_duckdb(con, sql):
    return con.execute(sql).df()

# -------------------- filtros e indicadores --------------------
def build_filters(df: pd.DataFrame):
    st.sidebar.header("Filtros")
    anos = sorted(df.get("ano_internacao", df.get("ano", pd.Series([], dtype=float))).dropna().unique().tolist())
    ano_sel = st.sidebar.multiselect("Ano da internação", anos, default=anos)
    sexos = sorted(df["sexo"].dropna().unique().tolist()) if "sexo" in df else []
    sexo_sel = st.sidebar.multiselect("Sexo", sexos, default=sexos)
    idade_min, idade_max = (int(np.nanmin(df["idade"])), int(np.nanmax(df["idade"]))) if "idade" in df and df["idade"].notna().any() else (0,120)
    idade_sel = st.sidebar.slider("Idade", min_value=0, max_value=max(idade_max,1), value=(idade_min, idade_max), step=1)
    carater_vals = sorted(df["tipo_evolucao"].dropna().unique().tolist()) if "tipo_evolucao" in df else []
    carater_sel = st.sidebar.multiselect("Caráter do atendimento", carater_vals, default=carater_vals)
    etnias = sorted(df["etnia"].dropna().unique().tolist()) if "etnia" in df else []
    etnia_sel = st.sidebar.multiselect("Raça/Cor", etnias, default=etnias)
    cidade_vals = sorted(df["cidade_moradia"].dropna().unique().tolist()) if "cidade_moradia" in df else []
    cidade_sel = st.sidebar.multiselect("Município de residência", cidade_vals, default=cidade_vals[:25])
    return {"ano": ano_sel, "sexo": sexo_sel, "idade": idade_sel, "carater": carater_sel, "etnia": etnia_sel, "cidade": cidade_sel}

def apply_filters(df, f):
    if "ano_internacao" in df.columns and f["ano"]:
        df = df[df["ano_internacao"].isin(f["ano"])]
    elif "ano" in df.columns and f["ano"]:
        df = df[df["ano"].isin(f["ano"])]
    if "sexo" in df.columns and f["sexo"]:
        df = df[df["sexo"].isin(f["sexo"])]
    if "idade" in df.columns:
        df = df[(df["idade"] >= f["idade"][0]) & (df["idade"] <= f["idade"][1])]
    if "tipo_evolucao" in df.columns and f["carater"]:
        df = df[df["tipo_evolucao"].isin(f["carater"])]
    if "etnia" in df.columns and f["etnia"]:
        df = df[df["etnia"].isin(f["etnia"])]
    if "cidade_moradia" in df.columns and f["cidade"]:
        df = df[df["cidade_moradia"].isin(f["cidade"])]
    return df

def pacientes_unicos(df: pd.DataFrame) -> pd.DataFrame:
    if {"prontuario_anonimo","data_internacao"}.issubset(df.columns):
        return (df.sort_values(["prontuario_anonimo","data_internacao"])
                  .groupby("prontuario_anonimo", as_index=False).tail(1))
    return df.drop_duplicates(subset=["prontuario_anonimo"])

def kpis(df_eventos: pd.DataFrame, df_pacientes: pd.DataFrame):
    pacientes = df_pacientes["prontuario_anonimo"].nunique() if "prontuario_anonimo" in df_pacientes else np.nan
    internacoes = df_eventos["codigo_internacao"].nunique() if "codigo_internacao" in df_eventos else len(df_eventos)
    tmi = df_eventos["dias_permanencia"].replace([np.inf,-np.inf], np.nan).dropna().mean() if "dias_permanencia" in df_eventos else np.nan

    mort_hosp = np.nan
    if {"data_internacao","data_alta"}.issubset(df_eventos.columns):
        e = df_eventos.copy()
        if "data_obito" in e.columns:
            e["obito_no_periodo"] = (e["data_obito"].notna()) & \
                (e["data_obito"] >= e["data_internacao"]) & \
                (e["data_obito"] <= (e["data_alta"] - pd.Timedelta(days=1)))
        elif "evolucao" in e.columns:
            e["obito_no_periodo"] = e["evolucao"].astype(str).str.contains("ÓBITO", case=False, na=False)
        else:
            e["obito_no_periodo"] = False
        denom = e["codigo_internacao"].nunique() if "codigo_internacao" in e else len(e)
        numer = e.loc[e["obito_no_periodo"]]
        numer = numer["codigo_internacao"].nunique() if "codigo_internacao" in e else len(numer)
        mort_hosp = (numer/denom*100) if denom else np.nan
    return pacientes, internacoes, tmi, mort_hosp

def reinternacao_30d_pos_proced(df: pd.DataFrame):
    ok = {"prontuario_anonimo","codigo_internacao","data_internacao","data_alta"}.issubset(df.columns)
    if not ok: return np.nan
    s = df.sort_values(["prontuario_anonimo","data_internacao","data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta_proc"] = (s["next_dt_internacao"] - s["data_internacao"]).dt.days
    s["delta_pos_alta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta_pos_alta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[(s["delta_proc"].between(0,30, inclusive="both")) & (~s["transfer"])]["codigo_internacao"].nunique()
    return (numer/base*100) if base else np.nan

def reinternacao_30d_pos_alta(df: pd.DataFrame):
    ok = {"prontuario_anonimo","codigo_internacao","data_internacao","data_alta"}.issubset(df.columns)
    if not ok: return np.nan
    s = df.sort_values(["prontuario_anonimo","data_internacao","data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[(s["delta"].between(0,30, inclusive="both")) & (~s["transfer"])]["codigo_internacao"].nunique()
    return (numer/base*100) if base else np.nan

# -------------------- UI: carregamento --------------------
st.title("Perfil dos Pacientes")

tab_parquet, tab_csv = st.tabs(["Parquet único (recomendado)", "3 CSVs grandes (DuckDB)"])

df = None
with tab_parquet:
    file_parquet = st.file_uploader("Carregue o Parquet único (dataset_unico_2019_2025.parquet)", type=["parquet"], key="pq")
    if file_parquet:
        df = load_parquet(file_parquet)

with tab_csv:
    c1, c2, c3 = st.columns(3)
    evo = c1.file_uploader("EVOLUÇÕES (csv)", type=["csv"], key="evo")
    proc = c2.file_uploader("PROCEDIMENTOS (csv)", type=["csv"], key="proc")
    cti = c3.file_uploader("CIDs/UTI (csv)", type=["csv"], key="cti")
    if evo and proc and cti:
        tmpdir = tempfile.mkdtemp()
        p_evo = os.path.join(tmpdir, "evo.csv"); open(p_evo, "wb").write(evo.getbuffer())
        p_proc = os.path.join(tmpdir, "proc.csv"); open(p_proc, "wb").write(proc.getbuffer())
        p_cti = os.path.join(tmpdir, "cti.csv"); open(p_cti, "wb").write(cti.getbuffer())
        con = load_duckdb((p_evo, p_proc, p_cti))
        # materializa um recorte inicial para filtros/gráficos; consultas adicionais podem ser puxadas conforme necessidade
        df = df_from_duckdb(con, "SELECT * FROM dataset")
        df = _post_load(df)

if df is None or df.empty:
    st.info("Carregue um Parquet único **ou** os 3 CSVs.")
    st.stop()

# -------------------- filtros e bases --------------------
f = build_filters(df)
df_f = apply_filters(df, f)                 # eventos (AIHs) filtrados
df_pac = pacientes_unicos(df_f)            # 1 linha por paciente filtrado

# toggle de análise
modo_perfil = st.toggle("Contar por **paciente único** (perfil). Desative para **internações**.", value=True)
base = df_pac if modo_perfil else df_f

# -------------------- KPIs --------------------
pacientes, internacoes, tmi, mort_hosp = kpis(df_f, df_pac)
ri_proc = reinternacao_30d_pos_proced(df_f)
ri_alta = reinternacao_30d_pos_alta(df_f)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Pacientes (distintos)", f"{int(pacientes):,}".replace(",",".") if pd.notna(pacientes) else "—")
k2.metric("Internações", f"{int(internacoes):,}".replace(",",".") if pd.notna(internacoes) else "—")
k3.metric("Tempo médio de internação (dias)", f"{tmi:.1f}" if pd.notna(tmi) else "—")
k4.metric("Reinternação 30d (procedimento)", f"{ri_proc:.1f}%" if pd.notna(ri_proc) else "—")
k5.metric("Reinternação 30d (alta)", f"{ri_alta:.1f}%" if pd.notna(ri_alta) else "—")
k6.metric("Mortalidade hospitalar", f"{mort_hosp:.1f}%" if pd.notna(mort_hosp) else "—")

st.divider()

# -------------------- gráficos --------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("Sexo")
    if "sexo" in base.columns:
        fig = px.bar(base.value_counts("sexo").rename("cont").reset_index(),
                     x="sexo", y="cont", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coluna 'sexo' não encontrada.")

with g2:
    st.subheader("Caráter do atendimento")
    col = "tipo_evolucao" if "tipo_evolucao" in df_f.columns else None
    if col:
        ordem = df_f[col].value_counts().index.tolist()
        fig = px.bar(df_f.value_counts(col).rename("cont").reset_index(),
                     x=col, y="cont", text_auto=True, category_orders={col:ordem})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coluna 'tipo_evolucao' não encontrada.")

st.subheader("Pirâmide etária")
if {"idade","sexo","faixa_etaria"}.issubset(base.columns):
    tmp = (base.dropna(subset=["faixa_etaria","sexo"])
                .groupby(["faixa_etaria","sexo"]).size().reset_index(name="n"))
    male = tmp[tmp["sexo"].eq("Masculino")].set_index("faixa_etaria")["n"].reindex(base["faixa_etaria"].cat.categories, fill_value=0)
    female = tmp[tmp["sexo"].eq("Feminino")].set_index("faixa_etaria")["n"].reindex(base["faixa_etaria"].cat.categories, fill_value=0)
    fig = go.Figure()
    fig.add_bar(y=male.index.astype(str), x=-male.values, name="Masculino", orientation="h")
    fig.add_bar(y=female.index.astype(str), x=female.values, name="Feminino", orientation="h")
    fig.update_layout(barmode="overlay", xaxis=dict(title="Pacientes (neg=M)"), yaxis=dict(title="Faixa etária"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Requer colunas 'idade' e 'sexo'.")

st.subheader("Raça/Cor × Sexo")
if {"etnia","sexo"}.issubset(base.columns):
    fig = px.bar(base.value_counts(["etnia","sexo"]).rename("cont").reset_index(),
                 x="etnia", y="cont", color="sexo", barmode="group", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Requer colunas 'etnia' e 'sexo'.")

st.subheader("Top municípios de residência")
if "cidade_moradia" in base.columns:
    topN = (base["cidade_moradia"].dropna().value_counts().head(20).reset_index())
    topN.columns = ["Município","Pacientes/Internações"]
    fig = px.treemap(topN, path=["Município"], values="Pacientes/Internações")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Coluna 'cidade_moradia' não encontrada.")

st.subheader("CID/Descrição (amostra)")
cid_col = [c for c in df_f.columns if ("cid" in c.lower() or "descricao" in c.lower() or "to_charsubstrievdescricao14000" in c.lower())]
if cid_col:
    col = cid_col[0]
    top = (df_f[col].dropna().astype(str).str.upper().str[:50]
           .value_counts().head(25).reset_index())
    top.columns = ["CID/Descrição (amostra)", "Frequência"]
    fig = px.bar(top, x="CID/Descrição (amostra)", y="Frequência", text_auto=True)
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Não encontrei coluna de CID/descrição no dataset.")

st.subheader("Amostra dos registros filtrados")
st.dataframe(df_f.head(200))
