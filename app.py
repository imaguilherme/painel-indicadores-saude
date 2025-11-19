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

# -------------------- funções de base --------------------
def pacientes_unicos(df: pd.DataFrame) -> pd.DataFrame:
    if {"prontuario_anonimo","data_internacao"}.issubset(df.columns):
        return (df.sort_values(["prontuario_anonimo","data_internacao"])
                  .groupby("prontuario_anonimo", as_index=False).tail(1))
    if "prontuario_anonimo" in df.columns:
        return df.drop_duplicates(subset=["prontuario_anonimo"])
    return df

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

# -------------------- filtros e indicadores --------------------
def build_filters(df: pd.DataFrame):
    st.sidebar.header("Filtros")

    # ano
    anos_col = "ano_internacao" if "ano_internacao" in df.columns else ("ano" if "ano" in df.columns else None)
    if anos_col:
        anos = sorted(df[anos_col].dropna().unique().tolist())
    else:
        anos = []
    ano_sel = st.sidebar.multiselect("Ano da internação", anos, default=anos)

    # idade
    if "idade" in df.columns and df["idade"].notna().any():
        idade_min, idade_max = int(np.nanmin(df["idade"])), int(np.nanmax(df["idade"]))
    else:
        idade_min, idade_max = 0, 120
    idade_sel = st.sidebar.slider("Idade", min_value=0, max_value=max(idade_max,1),
                                  value=(idade_min, idade_max), step=1)

    # estado (se existir)
    estado_col = next((c for c in df.columns if c.lower() in ["estado_residencia","uf_residencia","uf","estado","sigla_uf"]), None)
    estados_sel = []
    if estado_col:
        estados = sorted(df[estado_col].dropna().astype(str).unique().tolist())
        estados_sel = st.sidebar.multiselect("Estado de residência", estados, default=estados)

    # região de saúde (se existir)
    regiao_col = next((c for c in df.columns if "regiao" in c.lower() and "saude" in c.lower()), None)
    regioes_sel = []
    if regiao_col:
        regioes = sorted(df[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = st.sidebar.multiselect("Região de saúde", regioes, default=regioes)

    # município de residência (lista baseada em pacientes únicos)
    df_pac_ref = pacientes_unicos(df)
    cidade_col = "cidade_moradia" if "cidade_moradia" in df_pac_ref.columns else None
    cidades_sel = []
    if cidade_col:
        cidade_vals = sorted(df_pac_ref[cidade_col].dropna().astype(str).unique().tolist())
        # limitar default pra não ficar gigante
        default_cidades = cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        cidades_sel = st.sidebar.multiselect("Município de residência", cidade_vals, default=default_cidades)

    return {
        "ano": ano_sel,
        "idade": idade_sel,
        "estado": estados_sel,
        "regiao": regioes_sel,
        "cidade": cidades_sel,
    }

def apply_filters(df, f):
    # ano
    if "ano_internacao" in df.columns and f["ano"]:
        df = df[df["ano_internacao"].isin(f["ano"])]
    elif "ano" in df.columns and f["ano"]:
        df = df[df["ano"].isin(f["ano"])]

    # idade
    if "idade" in df.columns and f["idade"]:
        df = df[(df["idade"] >= f["idade"][0]) & (df["idade"] <= f["idade"][1])]

    # estado
    estado_col = next((c for c in df.columns if c.lower() in ["estado_residencia","uf_residencia","uf","estado","sigla_uf"]), None)
    if estado_col and f["estado"]:
        df = df[df[estado_col].isin(f["estado"])]

    # região de saúde
    regiao_col = next((c for c in df.columns if "regiao" in c.lower() and "saude" in c.lower()), None)
    if regiao_col and f["regiao"]:
        df = df[df[regiao_col].isin(f["regiao"])]

    # município
    if "cidade_moradia" in df.columns and f["cidade"]:
        df = df[df["cidade_moradia"].isin(f["cidade"])]

    return df

def show_active_filters(f):
    partes = []
    if f["ano"]:
        partes.append("**Ano:** " + ", ".join(str(a) for a in f["ano"]))
    if f["idade"]:
        partes.append(f"**Idade:** {f['idade'][0]}–{f['idade'][1]} anos")
    if f["estado"]:
        partes.append("**Estado:** " + ", ".join(f["estado"]))
    if f["regiao"]:
        partes.append("**Região de saúde:** " + ", ".join(f["regiao"]))
    if f["cidade"]:
        partes.append("**Município:** " + ", ".join(f["cidade"]))

    if partes:
        st.markdown("🔎 **Filtros ativos:** " + " | ".join(partes))
    else:
        st.markdown("🔎 **Filtros ativos:** nenhum filtro aplicado.")

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

# mostra filtros ativos no corpo da página
show_active_filters(f)
st.divider()

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

# -------------------- "Abas" de indicador + comparativo anual --------------------
st.markdown("### Indicadores principais")

indicador_top = st.radio(
    "Selecione o indicador para o comparativo anual:",
    ["Quantidade de pacientes", "Quantidade de internações"],
    horizontal=True,
    key="ind_top"
)

ano_col = "ano_internacao" if "ano_internacao" in df_f.columns else ("ano" if "ano" in df_f.columns else None)

if ano_col:
    df_year = df_f[~df_f[ano_col].isna()].copy()
    if not df_year.empty:
        grp = df_year.groupby(ano_col)

        if indicador_top == "Quantidade de pacientes":
            if "prontuario_anonimo" in df_year.columns:
                serie = grp["prontuario_anonimo"].nunique()
            else:
                serie = grp.size()
            y_label = "Pacientes distintos"
        else:  # Quantidade de internações
            if "codigo_internacao" in df_year.columns:
                serie = grp["codigo_internacao"].nunique()
            else:
                serie = grp.size()
            y_label = "Internações"

        df_plot = serie.reset_index(name="valor").sort_values(ano_col)
        fig_ano = px.bar(df_plot, x=ano_col, y="valor", text_auto=True)
        fig_ano.update_layout(
            xaxis_title="Ano",
            yaxis_title=y_label,
            height=280,
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_ano, use_container_width=True)
    else:
        st.info("Sem dados para o comparativo anual com os filtros atuais.")
else:
    st.info("Coluna de ano não encontrada no dataset.")

st.divider()

# -------------------- GRID PRINCIPAL--------------------
col_esq, col_meio, col_dir = st.columns([1.1, 1.3, 1.1])

# ========= COLUNA ESQUERDA =========
with col_esq:
    # linha Sexo + Caráter de Atendimento
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sexo")
        if "sexo" in base.columns:
            df_sexo = base.value_counts("sexo").rename("cont").reset_index()
            fig = px.bar(df_sexo, x="sexo", y="cont", text_auto=True)
            fig.update_layout(
                height=230,
                margin=dict(t=40, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna 'sexo' não encontrada.")

    with c2:
        # Caráter de atendimento (se existir)
        carater_col = None
        for cand in ["carater_atendimento","caracter_atendimento","carater","caráter_atendimento","carater_atend"]:
            if cand in df_f.columns:
                carater_col = cand
                break

        st.subheader("Caráter do atendimento")
        if carater_col:
            ordem = df_f[carater_col].value_counts().index.tolist()
            df_car = df_f.value_counts(carater_col).rename("cont").reset_index()
            fig = px.bar(df_car, x=carater_col, y="cont", text_auto=True,
                         category_orders={carater_col: ordem})
            fig.update_layout(
                height=230,
                xaxis_title="",
                margin=dict(t=40, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna de 'caráter do atendimento' não encontrada.")

    # Pirâmide etária (abaixo, ocupando a largura toda da coluna)
    st.subheader("Pirâmide etária")
    if {"idade","sexo","faixa_etaria"}.issubset(base.columns):
        tmp = (base.dropna(subset=["faixa_etaria","sexo"])
                    .groupby(["faixa_etaria","sexo"]).size().reset_index(name="n"))
        male = tmp[tmp["sexo"].eq("Masculino")].set_index("faixa_etaria")["n"].reindex(
            base["faixa_etaria"].cat.categories, fill_value=0
        )
        female = tmp[tmp["sexo"].eq("Feminino")].set_index("faixa_etaria")["n"].reindex(
            base["faixa_etaria"].cat.categories, fill_value=0
        )
        fig = go.Figure()
        fig.add_bar(y=male.index.astype(str), x=-male.values, name="Masculino", orientation="h")
        fig.add_bar(y=female.index.astype(str), x=female.values, name="Feminino", orientation="h")
        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Pacientes (neg=M)"),
            yaxis=dict(title="Faixa etária"),
            height=380,
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Requer colunas 'idade', 'sexo' e 'faixa_etaria'.")

# ========= COLUNA DO MEIO =========
with col_meio:
    st.subheader("Estado / Região de residência do paciente")

    if "cidade_moradia" in base.columns:
        df_geo = base.dropna(subset=["cidade_moradia"]).copy()
        df_geo["Pacientes/Internações"] = 1

        estado_col = next((c for c in df_geo.columns if c.lower() in ["estado_residencia","uf_residencia","uf","estado","sigla_uf"]), None)
        regiao_col = next((c for c in df_geo.columns if "regiao" in c.lower() and "saude" in c.lower()), None)

        path_cols = []
        if estado_col: path_cols.append(estado_col)
        if regiao_col: path_cols.append(regiao_col)
        path_cols.append("cidade_moradia")

        fig = px.treemap(
            df_geo,
            path=path_cols,
            values="Pacientes/Internações"
        )
        fig.update_layout(
            height=500,
            margin=dict(t=40, l=0, r=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Use os filtros de Estado / Região de saúde / Município para refinar a distribuição.")
    else:
        st.info("Coluna 'cidade_moradia' não encontrada.")

    # Raça × Sexo (embaixo)
    st.subheader("Raça/Cor × Sexo")
    if {"etnia","sexo"}.issubset(base.columns):
        df_etnia = (base.value_counts(["etnia","sexo"])
                         .rename("cont").reset_index())
        fig = px.bar(df_etnia, x="etnia", y="cont", color="sexo",
                     barmode="group", text_auto=True)
        fig.update_layout(
            xaxis_title="Raça/Cor",
            yaxis_title="Pacientes/Internações",
            height=320,
            margin=dict(t=40, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Requer colunas 'etnia' e 'sexo'.")

# ========= COLUNA DIREITA =========
with col_dir:
    # Card grande de quantidade de pacientes, como no print
    st.subheader("Quantidade de pacientes")
    st.markdown(
        f"<h2 style='text-align:center;'>{int(pacientes):,}</h2>".replace(",","."),
        unsafe_allow_html=True
    )
    st.caption("Pacientes distintos no período filtrado")

    st.markdown("---")

    # Procedimentos (top N)
    st.subheader("Procedimentos (amostra)")
    proc_cols = [c for c in base.columns if "proc_nome_prim" in c or "procedimento" == c.lower()]
    if proc_cols:
        pcol = proc_cols[0]
        top_proc = (base[pcol].dropna().astype(str)
                    .value_counts().head(10).reset_index())
        top_proc.columns = ["Procedimento", "Pacientes/Internações"]
        fig = px.bar(top_proc, x="Procedimento", y="Pacientes/Internações", text_auto=True)
        fig.update_layout(
            xaxis_tickangle=-35,
            height=260,
            margin=dict(t=40, b=120)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não encontrei coluna de procedimento agregada.")

    st.markdown("---")

    # Grupo / categorias CID-10
    st.subheader("Grupo e categorias de CID-10 (amostra)")
    cid_col = [c for c in df_f.columns if ("cid" in c.lower() or "descricao" in c.lower() or "to_charsubstrievdescricao14000" in c.lower())]
    if cid_col:
        col_cid = cid_col[0]
        top = (df_f[col_cid].dropna().astype(str).str.upper().str[:50]
               .value_counts().head(10).reset_index())
        top.columns = ["CID/Descrição (amostra)", "Frequência"]
        fig = px.bar(top, x="CID/Descrição (amostra)", y="Frequência", text_auto=True)
        fig.update_layout(
            xaxis_tickangle=-35,
            height=260,
            margin=dict(t=40, b=120)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não encontrei coluna de CID/descrição no dataset.")
