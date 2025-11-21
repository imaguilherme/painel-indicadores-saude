# app.py — Painel de Indicadores (Pacientes / Internações)
# Requisitos: streamlit, pandas, numpy, plotly, python-dateutil, pyarrow, duckdb

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import io
import os
import tempfile

st.set_page_config(page_title="Painel de Indicadores de Pacientes", layout="wide")


# ----------------------------------------
# Funções de suporte
# ----------------------------------------
@st.cache_data(show_spinner=False)
def _load_csv_or_excel(uploaded_file) -> pd.DataFrame:
    """Carrega CSV ou Excel, tentando inferir o separador."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    # caso seja CSV ou outro texto, usamos pandas.read_csv e heurística de separador
    # tenta ; depois , depois \t
    content = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(io.BytesIO(content), sep=";")
        if df.shape[1] == 1:
            # Possível separador diferente
            df = pd.read_csv(io.BytesIO(content), sep=",")
        if df.shape[1] == 1:
            df = pd.read_csv(io.BytesIO(content), sep="\t")
    except Exception:
        # tenta o default
        df = pd.read_csv(io.BytesIO(content))

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas (minúsculas, sem espaços extras)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _post_load(df: pd.DataFrame) -> pd.DataFrame:
    """Ajustes pós-carregamento: datas, tipos numéricos, faixas etárias, etc."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ----------------- datas -----------------
    date_cols = [
        "data_internacao",
        "data_alta",
        "data_obito",
        "dt_entrada_cti",
        "dt_saida_cti",
        "data_cirurgia",
        "data_procedimento",
        "data_cirurgia_min",
        "data_cirurgia_max",
        "data_nascimento",
        "dt_nascimento",
        "data_nasc",
        "dt_nasc",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # ----------------- numéricos básicos -----------------
    for c in ["idade", "ano", "ano_internacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------- dias de permanência -----------------
    if {"data_internacao", "data_alta"}.issubset(df.columns):
        df["dias_permanencia"] = (df["data_alta"] - df["data_internacao"]).dt.days

    # ----------------- faixas etárias customizadas -----------------
    if "idade" in df.columns:
        bins = [
            -1,
            0,  # < 1 ano
            8,  # 01 a 08 anos
            17,  # 09 a 17 anos
            26,  # 18 a 26 anos
            35,  # 27 a 35 anos
            44,  # 36 a 44 anos
            53,  # 45 a 53 anos
            62,  # 54 a 62 anos
            71,  # 63 a 71 anos
            80,  # 72 a 80 anos
            89,  # 81 a 89 anos
            200,  # 90 anos ou mais
        ]

        labels = [
            "< 1 ano",
            "01 a 08 anos",
            "09 a 17 anos",
            "18 a 26 anos",
            "27 a 35 anos",
            "36 a 44 anos",
            "45 a 53 anos",
            "54 a 62 anos",
            "63 a 71 anos",
            "72 a 80 anos",
            "81 a 89 anos",
            "90 anos ou mais",
        ]

        df["faixa_etaria"] = pd.cut(
            pd.to_numeric(df["idade"], errors="coerce"),
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

    # ----------------- deduplicação -----------------
    keys = [
        c
        for c in [
            "codigo_internacao",
            "prontuario_anonimo",
            "data_internacao",
            "data_alta",
        ]
        if c in df.columns
    ]
    if keys:
        df = df.drop_duplicates(subset=keys)

    return df


def _describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de percentuais de missing por coluna."""
    total = len(df)
    missing = df.isna().sum()
    perc = (missing / total) * 100
    out = pd.DataFrame(
        {"coluna": df.columns, "missing": missing.values, "perc_missing": perc.values}
    )
    out = out.sort_values("perc_missing", ascending=False)
    return out


# Para permitir que o usuário use DuckDB com 3 CSVs ou 1 parquet único
@st.cache_resource(show_spinner=False)
def _load_with_duckdb(files) -> pd.DataFrame:
    """
    Carrega múltiplos CSVs grandes ou um parquet usando duckdb e concatena.
    files: lista de UploadedFile
    """
    if not files:
        return pd.DataFrame()

    # Se for um arquivo só e for parquet
    if len(files) == 1 and files[0].name.lower().endswith(".parquet"):
        df = pd.read_parquet(files[0])
        return _post_load(_normalize_columns(df))

    # Caso geral: 1 ou mais CSV/XLSX utilizando DuckDB
    con = duckdb.connect(database=":memory:")
    dfs = []

    tmpdir = tempfile.mkdtemp()
    for idx, f in enumerate(files):
        fname = f.name.lower()
        full_path = os.path.join(tmpdir, f"file_{idx}_{fname}")
        with open(full_path, "wb") as fp:
            fp.write(f.read())

        if fname.endswith(".parquet"):
            q = f"SELECT * FROM read_parquet('{full_path}')"
        elif fname.endswith(".xlsx") or fname.endswith(".xls"):
            q = f"SELECT * FROM st_read('{full_path}')"
        else:
            # CSV — tenta ; como separador; se falhar, tentamos outras abordagens
            q = f"""
            SELECT * FROM read_csv_auto('{full_path}', header=True, sep=';')
            """

        try:
            df_tmp = con.execute(q).df()
        except Exception:
            # fallback: tenta inferência automática
            q2 = f"SELECT * FROM read_csv_auto('{full_path}', header=True)"
            df_tmp = con.execute(q2).df()

        dfs.append(df_tmp)

    if not dfs:
        return pd.DataFrame()

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = _normalize_columns(df_final)
    df_final = _post_load(df_final)

    return df_final


# ----------------------------------------
# Layout principal
# ----------------------------------------
st.title("📊 Painel de Indicadores de Pacientes / Internações")

with st.expander("ℹ️ Instruções de uso", expanded=False):
    st.markdown(
        """
- Carregue **1 arquivo parquet grande** OU até **3 arquivos CSV/XLSX** contendo dados de pacientes/internações.
- As colunas de referência esperadas (quando existirem) são:
  - `prontuario_anonimo`, `idade`, `sexo`, `municipio_residencia`, `data_internacao`, `data_alta`, `carater_atendimento`, etc.
- Os gráficos são atualizados automaticamente conforme os filtros.
"""
    )

uploaded_files = st.file_uploader(
    "Carregue o(s) arquivo(s) de dados (Parquet, CSV ou Excel)",
    type=["csv", "xlsx", "xls", "parquet"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("Carregue ao menos um arquivo para iniciar a análise.")
    st.stop()

with st.spinner("Carregando e processando dados..."):
    df = _load_with_duckdb(uploaded_files)

if df.empty:
    st.error("Não foi possível carregar dados a partir dos arquivos enviados.")
    st.stop()

st.success(f"Dados carregados com sucesso! Total de linhas: **{len(df):,}**.".replace(",", "."))

# Mostra algumas colunas
with st.expander("👀 Visualizar amostra dos dados carregados"):
    st.dataframe(df.head(50))

# ----------------------------------------
# Barra lateral — filtros globais
# ----------------------------------------
st.sidebar.header("🔎 Filtros Globais")

# Ano — detecta as colunas possíveis
possible_year_cols = [c for c in ["ano", "ano_internacao"] if c in df.columns]
year_col = None
if possible_year_cols:
    year_col = possible_year_cols[0]
    anos_unicos = sorted([int(a) for a in df[year_col].dropna().unique()])
    anos_sel = st.sidebar.multiselect("Ano da internação", anos_unicos, default=anos_unicos)
else:
    anos_sel = None

# Faixa etária
faixas_unicas = []
if "faixa_etaria" in df.columns:
    faixas_unicas = [str(x) for x in df["faixa_etaria"].dropna().unique()]
    faixas_unicas = sorted(
        faixas_unicas,
        key=lambda x: [
            "<" in x,
            "".join([d for d in x if d.isdigit()]) or "0",
        ],
    )

faixas_sel = (
    st.sidebar.multiselect(
        "Faixa etária",
        faixas_unicas,
        default=faixas_unicas if faixas_unicas else None,
    )
    if faixas_unicas
    else None
)

# Sexo
sexos_unicos = []
if "sexo" in df.columns:
    sexos_unicos = sorted(df["sexo"].dropna().astype(str).unique())

sexos_sel = (
    st.sidebar.multiselect(
        "Sexo",
        sexos_unicos,
        default=sexos_unicos if sexos_unicos else None,
    )
    if sexos_unicos
    else None
)

# Município de residência
mun_cols = [c for c in ["municipio_residencia", "mun_resid", "municipio"] if c in df.columns]
mun_col_atual = mun_cols[0] if mun_cols else None
mun_sel = None
if mun_col_atual is not None:
    municipios_unicos = sorted(df[mun_col_atual].dropna().astype(str).unique())
    # opção de selecionar todos
    st.sidebar.write("Municípios de residência")
    select_all_mun = st.sidebar.checkbox("Selecionar todos", value=True)
    if select_all_mun:
        mun_sel = municipios_unicos
    else:
        mun_sel = st.sidebar.multiselect(
            "Escolha município(s)", municipios_unicos, default=[]
        )

# Caráter de atendimento
carater_cols = [c for c in ["carater_atendimento", "carater_atend"] if c in df.columns]
carater_col = carater_cols[0] if carater_cols else None
carater_sel = None
if carater_col:
    carater_unicos = sorted(df[carater_col].dropna().astype(str).unique())
    carater_sel = st.sidebar.multiselect(
        "Caráter de atendimento",
        carater_unicos,
        default=carater_unicos,
    )

# Região de saúde / Estado se existirem
regiao_cols = [c for c in ["regiao_saude", "regiao"] if c in df.columns]
estado_cols = [c for c in ["estado", "uf"] if c in df.columns]

regiao_col = regiao_cols[0] if regiao_cols else None
estado_col = estado_cols[0] if estado_cols else None

regiao_sel = None
estado_sel = None

if regiao_col:
    regiao_unicas = sorted(df[regiao_col].dropna().astype(str).unique())
    regiao_sel = st.sidebar.multiselect(
        "Região de saúde",
        regiao_unicas,
        default=regiao_unicas,
    )

if estado_col:
    estado_unicas = sorted(df[estado_col].dropna().astype(str).unique())
    estado_sel = st.sidebar.multiselect(
        "Estado (UF)",
        estado_unicas,
        default=estado_unicas,
    )

# Aplica filtros globais
df_filtrado = df.copy()

if anos_sel is not None and year_col:
    df_filtrado = df_filtrado[df_filtrado[year_col].isin(anos_sel)]

if faixas_sel is not None and "faixa_etaria" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["faixa_etaria"].astype(str).isin(faixas_sel)]

if sexos_sel is not None and "sexo" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["sexo"].astype(str).isin(sexos_sel)]

if mun_sel is not None and mun_col_atual:
    df_filtrado = df_filtrado[df_filtrado[mun_col_atual].astype(str).isin(mun_sel)]

if carater_sel is not None and carater_col:
    df_filtrado = df_filtrado[df_filtrado[carater_col].astype(str).isin(carater_sel)]

if regiao_sel is not None and regiao_col:
    df_filtrado = df_filtrado[df_filtrado[regiao_col].astype(str).isin(regiao_sel)]

if estado_sel is not None and estado_col:
    df_filtrado = df_filtrado[df_filtrado[estado_col].astype(str).isin(estado_sel)]

st.sidebar.markdown("---")

# Mostrar resumo dos filtros
st.sidebar.subheader("📌 Resumo dos filtros aplicados")
filtros_ativos = []
if anos_sel:
    filtros_ativos.append(f"Ano(s): {', '.join(map(str, anos_sel))}")
if faixas_sel:
    filtros_ativos.append(f"Faixa(s) etária(s): {', '.join(map(str, faixas_sel))}")
if sexos_sel:
    filtros_ativos.append(f"Sexo(s): {', '.join(map(str, sexos_sel))}")
if mun_sel:
    filtros_ativos.append(f"Município(s): {', '.join(map(str, mun_sel))}")
if carater_sel:
    filtros_ativos.append(f"Caráter(es) de atendimento: {', '.join(map(str, carater_sel))}")
if regiao_sel:
    filtros_ativos.append(f"Região(ões) de saúde: {', '.join(map(str, regiao_sel))}")
if estado_sel:
    filtros_ativos.append(f"Estado(s): {', '.join(map(str, estado_sel))}")

if filtros_ativos:
    st.sidebar.write("\n".join(f"- {f}" for f in filtros_ativos))
else:
    st.sidebar.write("Nenhum filtro aplicado além do dataset completo.")

# ----------------------------------------
# Seções do painel
# ----------------------------------------
aba = st.tabs(
    [
        "📌 Visão Geral",
        "👤 Perfil dos Pacientes",
        "🏥 Internações e Evolução",
        "🧩 Características Clínicas / CID",
    ]
)

# ========================================
# ABA 1 — VISÃO GERAL
# ========================================
with aba[0]:
    st.header("📌 Visão Geral")

    col1, col2, col3, col4 = st.columns(4)

    # Total de pacientes (prontuários únicos)
    if "prontuario_anonimo" in df_filtrado.columns:
        total_pacientes = df_filtrado["prontuario_anonimo"].nunique()
    else:
        total_pacientes = df_filtrado.shape[0]

    # Total de internações
    if "codigo_internacao" in df_filtrado.columns:
        total_internacoes = df_filtrado["codigo_internacao"].nunique()
    else:
        total_internacoes = df_filtrado.shape[0]

    # Mortalidade hospitalar
    if "morte_hospitalar" in df_filtrado.columns:
        mort = df_filtrado["morte_hospitalar"].fillna(0)
        total_obitos = (mort == 1).sum()
        taxa_mortalidade = (total_obitos / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
    else:
        total_obitos = np.nan
        taxa_mortalidade = np.nan

    # Reinternação 30 dias
    if "reinternado_30dias" in df_filtrado.columns:
        rein = df_filtrado["reinternado_30dias"].fillna(0)
        total_reint = (rein == 1).sum()
        taxa_reint = (total_reint / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
    else:
        total_reint = np.nan
        taxa_reint = np.nan

    col1.metric("👤 Pacientes únicos", f"{total_pacientes:,}".replace(",", "."))
    col2.metric("🏥 Internações", f"{total_internacoes:,}".replace(",", "."))
    col3.metric(
        "⚰️ Óbitos hospitalares",
        f"{total_obitos if not np.isnan(total_obitos) else '-'}",
        f"{taxa_mortalidade:.1f}%" if not np.isnan(taxa_mortalidade) else None,
    )
    col4.metric(
        "🔁 Reinternações em 30 dias",
        f"{total_reint if not np.isnan(total_reint) else '-'}",
        f"{taxa_reint:.1f}%" if not np.isnan(taxa_reint) else None,
    )

    st.markdown("---")

    # Evolução temporal — comparando anos
    st.subheader("📈 Evolução anual de internações / pacientes")

    indicador_top = st.selectbox(
        "Indicador",
        [
            "Quantidade de internações",
            "Quantidade de pacientes",
            "Taxa de mortalidade hospitalar",
            "Taxa de reinternação em 30 dias",
        ],
    )

    # escolhe a melhor coluna de ano
    ano_col = year_col
    if ano_col is None:
        # tenta extrair o ano de data_internacao se existir
        if "data_internacao" in df_filtrado.columns:
            df_filtrado["ano_tmp"] = df_filtrado["data_internacao"].dt.year
            ano_col = "ano_tmp"

    if ano_col is None:
        st.info("Não há coluna de ano ou data para construir a evolução temporal.")
    else:
        df_year = df_filtrado.copy()
        df_year = df_year[df_year[ano_col].notna()]
        df_year[ano_col] = df_year[ano_col].astype(int)

        if not df_year.empty:
            grp = df_year.groupby(ano_col)

            if indicador_top == "Quantidade de pacientes":
                if "prontuario_anonimo" in df_year.columns:
                    serie = grp["prontuario_anonimo"].nunique()
                else:
                    serie = grp.size()
                label_y = "Pacientes únicos"
            elif indicador_top == "Taxa de mortalidade hospitalar" and "morte_hospitalar" in df_year.columns:
                mort = df_year["morte_hospitalar"].fillna(0)
                df_year["eh_obito"] = (mort == 1).astype(int)
                serie = grp["eh_obito"].mean() * 100
                label_y = "Taxa de mortalidade (%)"
            elif indicador_top == "Taxa de reinternação em 30 dias" and "reinternado_30dias" in df_year.columns:
                rein = df_year["reinternado_30dias"].fillna(0)
                df_year["eh_reint"] = (rein == 1).astype(int)
                serie = grp["eh_reint"].mean() * 100
                label_y = "Taxa de reinternação (%)"
            else:
                serie = grp.size()
                label_y = "Internações"

            fig_ano = px.line(
                serie.reset_index(),
                x=ano_col,
                y=0,
                markers=True,
            )
            fig_ano.update_layout(
                xaxis_title="Ano",
                yaxis_title=label_y,
                title=indicador_top,
            )
            st.plotly_chart(fig_ano, width="stretch")
        else:
            st.info("Não há dados suficientes para gerar a série temporal.")

# ========================================
# ABA 2 — PERFIL DOS PACIENTES
# ========================================
with aba[1]:
    st.header("👤 Perfil dos Pacientes")

    col_p1, col_p2 = st.columns([1, 1])

    # Distribuição por sexo
    with col_p1:
        st.subheader("Distribuição por sexo")

        if "sexo" in df_filtrado.columns:
            sexo_counts = (
                df_filtrado["sexo"].fillna("Ignorado").astype(str).value_counts().reset_index()
            )
            sexo_counts.columns = ["sexo", "n"]

            fig = px.bar(
                sexo_counts,
                x="sexo",
                y="n",
                text="n",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Sexo",
                yaxis_title="Quantidade",
                margin=dict(t=40),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna de sexo não encontrada no dataset.")

    # Faixa etária (pirâmide etária ou barras)
    with col_p2:
        st.subheader("Distribuição por faixa etária")

        if "faixa_etaria" in df_filtrado.columns:
            # Pirâmide etária (feminino/masculino)
            if "sexo" in df_filtrado.columns:
                df_pira = df_filtrado.copy()
                df_pira["sexo"] = df_pira["sexo"].fillna("Ignorado").astype(str)
                df_pira["faixa_etaria"] = df_pira["faixa_etaria"].astype(str)

                categorias = [
                    "< 1 ano",
                    "01 a 08 anos",
                    "09 a 17 anos",
                    "18 a 26 anos",
                    "27 a 35 anos",
                    "36 a 44 anos",
                    "45 a 53 anos",
                    "54 a 62 anos",
                    "63 a 71 anos",
                    "72 a 80 anos",
                    "81 a 89 anos",
                    "90 anos ou mais",
                ]
                df_pira["faixa_etaria"] = pd.Categorical(
                    df_pira["faixa_etaria"],
                    categories=categorias,
                    ordered=True,
                )
                df_pira = df_pira[df_pira["faixa_etaria"].isin(categorias)]

                tabela = (
                    df_pira.groupby(["faixa_etaria", "sexo"], observed=True)
                    .size()
                    .reset_index(name="n")
                )
                pivot = tabela.pivot(index="faixa_etaria", columns="sexo", values="n").fillna(0)

                for s in pivot.columns:
                    if "M" in s.upper():  # supondo masculino
                        pivot[s] = -pivot[s]

                fig = go.Figure()

                for s in pivot.columns:
                    fig.add_trace(
                        go.Bar(
                            y=pivot.index.astype(str),
                            x=pivot[s],
                            name=s,
                            orientation="h",
                        )
                    )

                fig.update_layout(
                    barmode="relative",
                    xaxis_title="Número de pacientes",
                    yaxis_title="Faixa etária",
                    title="Pirâmide etária por sexo",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                faixa_counts = (
                    df_filtrado["faixa_etaria"]
                    .astype(str)
                    .value_counts()
                    .reindex(
                        [
                            "< 1 ano",
                            "01 a 08 anos",
                            "09 a 17 anos",
                            "18 a 26 anos",
                            "27 a 35 anos",
                            "36 a 44 anos",
                            "45 a 53 anos",
                            "54 a 62 anos",
                            "63 a 71 anos",
                            "72 a 80 anos",
                            "81 a 89 anos",
                            "90 anos ou mais",
                        ]
                    )
                    .dropna()
                    .reset_index()
                )
                faixa_counts.columns = ["faixa_etaria", "n"]
                fig = px.bar(
                    faixa_counts,
                    x="faixa_etaria",
                    y="n",
                    text="n",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    xaxis_title="Faixa etária",
                    yaxis_title="Quantidade",
                    margin=dict(t=40),
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna de faixa etária não encontrada. Verifique se 'idade' está presente para cálculo.")

    st.markdown("---")

    # Distribuição por município de residência (treemap ou barra)
    st.subheader("📍 Distribuição por município de residência")

    if mun_col_atual:
        df_mun = df_filtrado.copy()
        df_mun[mun_col_atual] = df_mun[mun_col_atual].fillna("Ignorado").astype(str)

        # Somente top N municípios
        top_n = st.slider("Quantidade de municípios a exibir (por frequência)", 5, 50, 20)
        mun_counts = (
            df_mun[mun_col_atual]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        mun_counts.columns = ["municipio_residencia", "n"]

        tipo_mapa = st.radio(
            "Tipo de visualização",
            ["Treemap", "Barras"],
            horizontal=True,
        )

        if tipo_mapa == "Treemap":
            fig = px.treemap(
                mun_counts,
                path=["municipio_residencia"],
                values="n",
            )
            fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            fig = px.bar(
                mun_counts,
                x="municipio_residencia",
                y="n",
                text="n",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Município de residência",
                yaxis_title="Quantidade",
                margin=dict(t=40),
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Coluna de município de residência não encontrada no dataset.")

# ========================================
# ABA 3 — INTERNAÇÕES E EVOLUÇÃO
# ========================================
with aba[2]:
    st.header("🏥 Internações e Evolução")

    col_i1, col_i2 = st.columns(2)

    # Distribuição de dias de permanência
    with col_i1:
        st.subheader("Dias de permanência hospitalar")

        if "dias_permanencia" in df_filtrado.columns:
            df_dp = df_filtrado.copy()
            df_dp = df_dp[df_dp["dias_permanencia"].notna()]

            if not df_dp.empty:
                fig = px.histogram(
                    df_dp,
                    x="dias_permanencia",
                    nbins=30,
                )
                fig.update_layout(
                    xaxis_title="Dias de permanência",
                    yaxis_title="Quantidade de internações",
                )
                st.plotly_chart(fig, width="stretch")

                st.write(
                    f"Mediana de dias de permanência: **{df_dp['dias_permanencia'].median():.1f}**"
                )
            else:
                st.info("Não há valores válidos de dias de permanência após filtros.")
        else:
            st.info(
                "Dias de permanência não calculados. Verifique se as colunas 'data_internacao' e 'data_alta' estão presentes."
            )

    # Características do desfecho (mortalidade) ao longo do tempo
    with col_i2:
        st.subheader("Mortalidade hospitalar ao longo dos anos")

        if (
            "morte_hospitalar" in df_filtrado.columns
            and (year_col or "data_internacao" in df_filtrado.columns)
        ):
            df_mort = df_filtrado.copy()

            ano_col_mort = year_col
            if ano_col_mort is None and "data_internacao" in df_mort.columns:
                df_mort["ano_mort"] = df_mort["data_internacao"].dt.year
                ano_col_mort = "ano_mort"

            df_mort = df_mort[df_mort[ano_col_mort].notna()]
            df_mort[ano_col_mort] = df_mort[ano_col_mort].astype(int)

            if not df_mort.empty:
                df_mort["eh_obito"] = (df_mort["morte_hospitalar"].fillna(0) == 1).astype(int)
                serie = df_mort.groupby(ano_col_mort)["eh_obito"].mean() * 100

                fig = px.line(
                    serie.reset_index(),
                    x=ano_col_mort,
                    y="eh_obito",
                    markers=True,
                )
                fig.update_layout(
                    xaxis_title="Ano",
                    yaxis_title="Taxa de mortalidade (%)",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Não há dados suficientes para calcular a mortalidade anual após filtros.")
        else:
            st.info(
                "Não há coluna de mortalidade hospitalar ou informação de ano para a evolução temporal."
            )

    st.markdown("---")

    # Caráter de atendimento x ano (heatmap/barras)
    st.subheader("Internações por ano e caráter de atendimento")

    if carater_col and (year_col or "data_internacao" in df_filtrado.columns):
        df_car = df_filtrado.copy()
        carater_plot_col = carater_col
        ano_col_car = year_col

        if ano_col_car is None and "data_internacao" in df_car.columns:
            df_car["ano"] = df_car["data_internacao"].dt.year
            ano_col_car = "ano"

        df_car = df_car[df_car[ano_col_car].notna()]
        df_car[ano_col_car] = df_car[ano_col_car].astype(int)

        if not df_car.empty:
            tab = (
                df_car.groupby([ano_col_car, carater_plot_col])
                .size()
                .reset_index(name="n")
            )

            tipo_car_plot = st.radio(
                "Tipo de visualização",
                ["Heatmap", "Barras empilhadas"],
                horizontal=True,
            )

            if tipo_car_plot == "Heatmap":
                fig = px.density_heatmap(
                    tab,
                    x=ano_col_car,
                    y=carater_plot_col,
                    z="n",
                    histfunc="avg",
                )
                fig.update_layout(
                    xaxis_title="Ano",
                    yaxis_title="Caráter de atendimento",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                fig = px.bar(
                    tab,
                    x=ano_col_car,
                    y="n",
                    color=carater_plot_col,
                )
                fig.update_layout(
                    xaxis_title="Ano",
                    yaxis_title="Quantidade de internações",
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info(
                "Não há dados suficientes após filtros para montar a tabela de ano x caráter de atendimento."
            )
    else:
        st.info(
            "Não há coluna de caráter de atendimento ou de ano/datas para cruzar informações."
        )

# ========================================
# ABA 4 — CARACTERÍSTICAS CLÍNICAS / CID
# ========================================
with aba[3]:
    st.header("🧩 Características Clínicas / CID")

    # Aqui consideramos que há uma coluna de CID principal e possivelmente
    # uma tabela externa de CID -> descrição (pode ser mergeado no futuro)

    # Tenta encontrar colunas de CID
    possiveis_cid = [
        "cid_principal",
        "cid",
        "diagnostico_principal",
        "cid10",
        "cid_10",
    ]
    cid_col = None
    for c in possiveis_cid:
        if c in df_filtrado.columns:
            cid_col = c
            break

    if cid_col is None:
        st.info(
            "Não encontrei automaticamente uma coluna de CID principal. "
            "Verifique se seu dataset contém algo como 'cid_principal' ou 'cid10'."
        )
    else:
        st.subheader("Distribuição dos principais CIDs")

        df_cid = df_filtrado.copy()
        df_cid[cid_col] = df_cid[cid_col].fillna("Ignorado").astype(str)

        # top N CIDs
        top_n_cid = st.slider("Quantidade de CIDs principais a exibir", 5, 50, 15)
        cid_counts = (
            df_cid[cid_col]
            .value_counts()
            .head(top_n_cid)
            .reset_index()
        )
        cid_counts.columns = ["cid_principal", "n"]

        fig = px.bar(
            cid_counts,
            x="cid_principal",
            y="n",
            text="n",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="CID principal",
            yaxis_title="Quantidade de internações",
            margin=dict(t=40),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Se existir informação de reinternação e mortalidade, podemos ver top CIDs
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Top CIDs entre reinternações (30 dias)")

            if "reinternado_30dias" in df_filtrado.columns:
                df_reint = df_cid[df_cid["reinternado_30dias"] == 1]
                if not df_reint.empty:
                    cid_reint = (
                        df_reint[cid_col]
                        .value_counts()
                        .head(top_n_cid)
                        .reset_index()
                    )
                    cid_reint.columns = ["cid_principal", "n"]

                    fig = px.bar(
                        cid_reint,
                        x="cid_principal",
                        y="n",
                        text="n",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        xaxis_title="CID principal",
                        yaxis_title="Reinternações em 30 dias",
                        margin=dict(t=40),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Não há registros com reinternação em 30 dias após os filtros aplicados.")
            else:
                st.info("Coluna 'reinternado_30dias' não encontrada no dataset.")

        with col_c2:
            st.subheader("Top CIDs entre óbitos hospitalares")

            if "morte_hospitalar" in df_filtrado.columns:
                df_obito = df_cid[df_cid["morte_hospitalar"] == 1]
                if not df_obito.empty:
                    cid_morte = (
                        df_obito[cid_col]
                        .value_counts()
                        .head(top_n_cid)
                        .reset_index()
                    )
                    cid_morte.columns = ["cid_principal", "n"]

                    fig = px.bar(
                        cid_morte,
                        x="cid_principal",
                        y="n",
                        text="n",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        xaxis_title="CID principal",
                        yaxis_title="Óbitos hospitalares",
                        margin=dict(t=40),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Não há registros de óbito hospitalar após os filtros aplicados.")
            else:
                st.info("Coluna 'morte_hospitalar' não encontrada no dataset.")

        st.markdown("---")

        # Caso haja coluna de descrição de CID, podemos mostrar tabela
        possiveis_desc = [
            "descricao_cid",
            "descricao",
            "cid_descricao",
            "cid_desc",
        ]
        desc_col = None
        for c in possiveis_desc:
            if c in df_filtrado.columns:
                desc_col = c
                break

        if desc_col:
            st.subheader("Tabela de CIDs com descrição (amostra)")

            tabela_cid = (
                df_filtrado[[cid_col, desc_col]]
                .drop_duplicates()
                .rename(columns={cid_col: "CID", desc_col: "Descrição"})
                .head(200)
            )
            st.dataframe(tabela_cid)
        else:
            st.info("Não encontrei informações de CID com descrição textual no dataset.")

    st.markdown("---")

    st.subheader("📉 Análise adicional por CID e idade (opcional)")

    if cid_col and "idade" in df_filtrado.columns:
        df_cid_idade = df_filtrado.copy()
        df_cid_idade = df_cid_idade[df_cid_idade["idade"].notna()]

        top_n_scatter = st.slider(
            "Quantidade de CIDs para análise de idade",
            3,
            20,
            10,
        )
        principais_cids = (
            df_cid_idade[cid_col]
            .value_counts()
            .head(top_n_scatter)
            .index.tolist()
        )
        df_cid_idade = df_cid_idade[df_cid_idade[cid_col].isin(principais_cids)]

        fig = px.box(
            df_cid_idade,
            x=cid_col,
            y="idade",
        )
        fig.update_layout(
            xaxis_title="CID principal",
            yaxis_title="Idade",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "Para a análise adicional, são necessárias as colunas de CID principal e de idade."
        )

# Fim do app
