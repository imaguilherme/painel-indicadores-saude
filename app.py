import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

st.set_page_config(
    page_title="Painel Indicadores Saúde - ACC",
    page_icon="❤️",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Ajuste geral de fontes e cores */
        body, .stApp {
            background-color: #f5f5f5;
        }

        /* Títulos principais */
        h1, h2, h3 {
            color: #333333;
            font-family: "Helvetica", "Arial", sans-serif;
        }

        /* Cards métricas principais */
        .metric-card {
            padding: 1.2rem 1.5rem;
            border-radius: 0.8rem;
            background: linear-gradient(135deg, #ffffff, #f0f4ff);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        /* Subtítulos de seção */
        .section-subheader {
            font-size: 1.1rem;
            font-weight: 600;
            color: #444444;
            margin-bottom: 0.5rem;
        }

        /* Reduzir padding dos radios horizontais */
        div.row-widget.stRadio > div {
            flex-direction: row;
            gap: 1.5rem;
        }

        /* Esconder o texto 'Made with Streamlit' do rodapé (opcional) */
        footer {visibility: hidden;}
        .stDeployButton {display: none;}

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO DE DADOS (DUCKDB / PARQUET / CSV)
# ---------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_duckdb(path_duckdb: str):
    if not os.path.exists(path_duckdb):
        raise FileNotFoundError(f"Arquivo DuckDB não encontrado: {path_duckdb}")
    con = duckdb.connect(path_duckdb, read_only=True)
    return con


@st.cache_data(show_spinner=True)
def load_base(con: duckdb.DuckDBPyConnection, tabela: str) -> pd.DataFrame:
    query = f"SELECT * FROM {tabela}"
    df = con.execute(query).df()
    return df


@st.cache_data(show_spinner=True)
def load_csv_or_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")


# ---------------------------------------------------------------
# CAMINHOS E CARREGAMENTO EFETIVO
# ---------------------------------------------------------------

CAMINHO_DUCKDB = "dados/acc_db.duckdb"
TABELA_PRINCIPAL = "fato_internacoes"

try:
    con = load_duckdb(CAMINHO_DUCKDB)
    df_f = load_base(con, TABELA_PRINCIPAL)
except Exception as e:
    st.error(f"Erro ao carregar os dados principais: {e}")
    st.stop()

# ---------------------------------------------------------------
# PRÉ-PROCESSAMENTO BÁSICO
# ---------------------------------------------------------------

# Garantir colunas de datas em datetime
for col in ["dt_internacao", "dt_alta", "dt_obito"]:
    if col in df_f.columns:
        df_f[col] = pd.to_datetime(df_f[col], errors="coerce")

# Criar coluna de ano da internação, se não existir
if "ano_internacao" not in df_f.columns and "dt_internacao" in df_f.columns:
    df_f["ano_internacao"] = df_f["dt_internacao"].dt.year

# Criar coluna de mês/ano (AAAAMM) se fizer sentido
if "mes_ano" not in df_f.columns and "dt_internacao" in df_f.columns:
    df_f["mes_ano"] = df_f["dt_internacao"].dt.to_period("M").astype(str)

# Criar coluna de idade, se necessário
if "idade" not in df_f.columns and {"dt_nascimento", "dt_internacao"}.issubset(
    df_f.columns
):
    df_f["dt_nascimento"] = pd.to_datetime(df_f["dt_nascimento"], errors="coerce")
    df_f["idade"] = (
        (df_f["dt_internacao"] - df_f["dt_nascimento"]).dt.days // 365
    ).astype("float")

# ---------------------------------------------------------------
# FUNÇÕES DE INDICADORES E AGRUPAMENTOS
# ---------------------------------------------------------------

# Lista de indicadores disponíveis
indicadores_icardio = [
    "Quantidade de pacientes",
    "Quantidade de internações",
    "Quantidade de procedimentos",
    "Tempo médio de internação (dias)",
    "Internação em UTI (%)",
]

# Categorias de indicadores por tipo (para rótulo de eixo)
indicadores_percentual = ["Internação em UTI (%)"]
indicadores_media = ["Tempo médio de internação (dias)"]

# Função auxiliar para criar faixas etárias
def criar_faixas_etarias(df: pd.DataFrame) -> pd.DataFrame:
    if "idade" not in df.columns:
        return df

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
        df["idade"], bins=bins, labels=labels, right=True
    )

    return df


df_f = criar_faixas_etarias(df_f)


def calcular_tempo_medio_internacao(df: pd.DataFrame) -> float:
    if {"dt_internacao", "dt_alta"}.issubset(df.columns):
        dias = (df["dt_alta"] - df["dt_internacao"]).dt.days
        return dias.mean()
    return np.nan


def calcular_internacao_uti(df: pd.DataFrame) -> float:
    if "uti" in df.columns:
        return 100 * (df["uti"] == 1).mean()
    if "uti_flag" in df.columns:
        return 100 * (df["uti_flag"] == 1).mean()
    return np.nan


def agrega_para_grafico(
    df: pd.DataFrame, cols_groupby: list, indicador: str
) -> pd.DataFrame:
    """Agrupa df pelas cols_groupby e calcula o indicador selecionado."""

    if indicador == "Quantidade de pacientes":
        if "id_paciente" in df.columns:
            tmp = df.groupby(cols_groupby)["id_paciente"].nunique().reset_index()
            tmp.rename(columns={"id_paciente": "valor"}, inplace=True)
        else:
            tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
        return tmp

    if indicador == "Quantidade de internações":
        tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
        return tmp

    if indicador == "Quantidade de procedimentos":
        if "n_proced" in df.columns:
            tmp = df.groupby(cols_groupby)["n_proced"].sum().reset_index()
            tmp.rename(columns={"n_proced": "valor"}, inplace=True)
            return tmp
        else:
            tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
            return tmp

    if indicador == "Tempo médio de internação (dias)":
        if {"dt_internacao", "dt_alta"}.issubset(df.columns):
            tmp = df.groupby(cols_groupby).apply(calcular_tempo_medio_internacao)
            tmp = tmp.reset_index(name="valor")
            return tmp
        else:
            tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
            return tmp

    if indicador == "Internação em UTI (%)":
        if "uti" in df.columns or "uti_flag" in df.columns:
            tmp = df.groupby(cols_groupby).apply(calcular_internacao_uti)
            tmp = tmp.reset_index(name="valor")
            return tmp
        else:
            tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
            return tmp

    # Caso indicador não seja reconhecido:
    tmp = df.groupby(cols_groupby).size().reset_index(name="valor")
    return tmp


def format_val_for_card(indicador: str, valor: float) -> str:
    if pd.isna(valor):
        return "-"
    if indicador in indicadores_percentual:
        return f"{valor:,.1f}%".replace(",", ".")
    if indicador in indicadores_media:
        return f"{valor:,.1f}".replace(",", ".")
    # quantidade (inteiro)
    return f"{int(valor):,}".replace(",", ".")


def label_eixo_x(indicador):
    if indicador in indicadores_percentual:
        return "Taxa (%)"
    if indicador in indicadores_media:
        return "Média (dias)"
    return "Quantidade"



def get_sexo_color_map(categories):
    """Mapeia cores fixas para cada sexo.

    - Masculino: azul
    - Feminino: rosa
    - Outros: cinza neutro
    """
    mapa = {}
    for s in categories:
        s_raw = str(s)
        s_norm = s_raw.strip().upper()
        if s_norm in ["M", "MASCULINO"]:
            mapa[s_raw] = "#6794DC"  # azul
        elif s_norm in ["F", "FEMININO"]:
            mapa[s_raw] = "#E86F86"  # rosa
        else:
            mapa[s_raw] = "#A3A3A3"  # neutro para outros valores
    return mapa


st.markdown("### Indicadores disponíveis")

indicador_selecionado = st.radio(
    "Selecione o indicador para detalhar e para o comparativo anual:",
    indicadores_icardio,
    horizontal=True,
)


def calcular_indicador(nome):
    if nome == "Quantidade de pacientes":
        return pacientes
    if nome == "Quantidade de internações":
        return internacoes
    if nome == "Quantidade de procedimentos":
        return df_f["n_proced"].sum() if "n_proced" in df_f.columns else np.nan
    if nome == "Tempo médio de internação (dias)":
        return tmi
    if nome == "Internação em UTI (%)":
        return internacao_uti
    return np.nan


# ---------------------------------------------------------------
# INDICADORES GERAIS (CARD NO TOPO)
# ---------------------------------------------------------------

# Quantidade de pacientes
if "id_paciente" in df_f.columns:
    pacientes = df_f["id_paciente"].nunique()
else:
    pacientes = df_f.shape[0]

# Quantidade de internações
internacoes = df_f.shape[0]

# Quantidade de procedimentos
if "n_proced" in df_f.columns:
    qtd_procedimentos = df_f["n_proced"].sum()
else:
    qtd_procedimentos = np.nan

# Tempo médio de internação
tmi = calcular_tempo_medio_internacao(df_f)

# Internação em UTI (%)
internacao_uti = calcular_internacao_uti(df_f)

# Layout dos 4 cards principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Pacientes únicos",
        f"{pacientes:,}".replace(",", "."),
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Internações",
        f"{internacoes:,}".replace(",", "."),
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Procedimentos",
        "-" if pd.isna(qtd_procedimentos) else f"{qtd_procedimentos:,}".replace(
            ",", "."
        ),
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Tempo médio de internação (dias)",
        "-" if pd.isna(tmi) else f"{tmi:,.1f}".replace(",", "."),
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------
# FILTROS GERAIS NA PARTE SUPERIOR
# ---------------------------------------------------------------

st.markdown("### Filtros")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

anos_disponiveis = sorted(df_f["ano_internacao"].dropna().unique())
with col_f1:
    ano_selecionado = st.multiselect(
        "Ano de internação",
        options=anos_disponiveis,
        default=anos_disponiveis,
    )

sexo_opcoes = (
    ["F", "M"]
    if "sexo" in df_f["sexo"].dropna().unique()
    else sorted(df_f["sexo"].dropna().unique())
    if "sexo" in df_f.columns
    else []
)
with col_f2:
    sexo_filtro = st.multiselect(
        "Sexo",
        options=sexo_opcoes,
        default=sexo_opcoes,
    )

etnia_opcoes = (
    sorted(df_f["etnia"].dropna().unique())
    if "etnia" in df_f.columns
    else []
)
with col_f3:
    etnia_filtro = st.multiselect(
        "Raça/Cor",
        options=etnia_opcoes,
        default=etnia_opcoes,
    )

carater_opcoes = (
    sorted(df_f["carater_atendimento"].dropna().unique())
    if "carater_atendimento" in df_f.columns
    else []
)
with col_f4:
    carater_filtro = st.multiselect(
        "Caráter do atendimento",
        options=carater_opcoes,
        default=carater_opcoes,
    )

# Aplicar filtros no df_f
base_filtros = df_f.copy()

if ano_selecionado:
    base_filtros = base_filtros[base_filtros["ano_internacao"].isin( ano_selecionado )]

if sexo_filtro and "sexo" in base_filtros.columns:
    base_filtros = base_filtros[base_filtros["sexo"].isin(sexo_filtro)]

if etnia_filtro and "etnia" in base_filtros.columns:
    base_filtros = base_filtros[base_filtros["etnia"].isin(etnia_filtro)]

if carater_filtro and "carater_atendimento" in base_filtros.columns:
    base_filtros = base_filtros[
        base_filtros["carater_atendimento"].isin(carater_filtro)
    ]

# Essa será a base usada pelos gráficos
base_charts = base_filtros.copy()

# ---------------------------------------------------------------
# GRÁFICOS E VISUALIZAÇÕES PRINCIPAIS (ICARDIO)
# ---------------------------------------------------------------

st.markdown("## Visão Geral dos Pacientes (Perfil Demográfico)")

col_esq, col_dir = st.columns([1.2, 1.5])

# ---------------------------------------------------------------
# FUNÇÃO DE CARD (BARRA ÚNICA) PADRÃO
# ---------------------------------------------------------------

def card_bar_fig(
    df_cat: pd.DataFrame,
    cat_col: str,
    indicador: str,
    colors=None,
    color_map=None,
    height: int = 90,
):
    """Retorna um gráfico Plotly em forma de card (barra única segmentada).

    Agora aceita um ``color_map`` opcional para fixar cores por categoria.
    """
    if df_cat.empty:
        return go.Figure()

    df_plot = df_cat.copy()
    df_plot["dummy"] = "Total"
    df_plot["text"] = (
        df_plot[cat_col].astype(str).str.upper()
        + "<br>"
        + df_plot["valor"].apply(lambda v: format_val_for_card(indicador, v))
    )

    bar_kwargs = {}
    if color_map is not None:
        bar_kwargs["color_discrete_map"] = color_map
    elif colors is not None:
        bar_kwargs["color_discrete_sequence"] = colors

    fig = px.bar(
        df_plot,
        x="valor",
        y="dummy",
        color=cat_col,
        orientation="h",
        text="text",
        **bar_kwargs,
    )

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        marker_line_width=0,
    )

    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)

    fig.update_layout(
        height=height,
        margin=dict(l=1, r=1, t=5, b=5),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# --------------------------------------------------------------------
# BASE PARA GRÁFICOS E GRID PRINCIPAL
# --------------------------------------------------------------------

st.divider()

if indicador_selecionado == "Quantidade de pacientes":
    df_plot_global = agrega_para_grafico(base_charts, ["ano_internacao"], indicador_selecionado)
else:
    df_plot_global = agrega_para_grafico(base_charts, ["ano_internacao"], indicador_selecionado)

df_plot_global = df_plot_global.sort_values("ano_internacao")

st.markdown("### Evolução Anual do Indicador Selecionado")

fig_global = px.line(
    df_plot_global,
    x="ano_internacao",
    y="valor",
    markers=True,
)

fig_global.update_traces(
    text=df_plot_global["valor"].round(2),
    textposition="top center",
)

fig_global.update_xaxes(title="Ano")

fig_global.update_yaxes(
    title=label_eixo_x(indicador_selecionado),
    showgrid=True,
    zeroline=True,
)

st.plotly_chart(fig_global, use_container_width=True)

st.divider()

# --------------------------------------------------------------------
# GRID COM DOIS BLOCOS: ESQUERDA (PERFIL) E DIREITA (CARÁTER/LOCAL)
# --------------------------------------------------------------------

with col_esq:
    # ----------------- Sexo (CARD) -----------------
    st.subheader("Sexo")
    if "sexo" in base_charts.columns:
        df_sexo = agrega_para_grafico(base_charts, ["sexo"], indicador_selecionado)
        df_sexo = df_sexo.sort_values("valor", ascending=False)

        sexo_color_map = get_sexo_color_map(df_sexo["sexo"].unique())

        fig = card_bar_fig(
            df_sexo,
            cat_col="sexo",
            indicador=indicador_selecionado,
            color_map=sexo_color_map,
            height=90,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Coluna 'sexo' não encontrada.")


    # ----------------- Raça/Cor × Sexo (abaixo, maior) -----------------
    st.subheader("Raça/Cor × Sexo")
    if {"etnia", "sexo"}.issubset(base_charts.columns):
        df_etnia = agrega_para_grafico(
            base_charts, ["etnia", "sexo"], indicador_selecionado
        )

        # campo numérico arredondado
        df_etnia["valor_fmt"] = df_etnia["valor"].round(2)

        sexo_color_map = get_sexo_color_map(df_etnia["sexo"].unique())

        fig = px.bar(
            df_etnia,
            y="etnia",
            x="valor",
            color="sexo",
            barmode="group",
            orientation="h",
            text="valor_fmt",
            color_discrete_map=sexo_color_map,
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        fig.update_xaxes(
            title=label_eixo_x(indicador_selecionado),
        )
        fig.update_yaxes(
            title="Raça/Cor",
        )

        fig.update_layout(
            height=350,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Requer colunas 'etnia' e 'sexo'.")

    # ----------------- Pirâmide Etária -----------------
    st.subheader("Pirâmide Etária")
    if {"faixa_etaria", "sexo"}.issubset(base_charts.columns):
        categorias = [
            "90 anos ou mais",
            "81 a 89 anos",
            "72 a 80 anos",
            "63 a 71 anos",
            "54 a 62 anos",
            "45 a 53 anos",
            "36 a 44 anos",
            "27 a 35 anos",
            "18 a 26 anos",
            "09 a 17 anos",
            "01 a 08 anos",
            "< 1 ano",
        ]

        df_pira = base_charts.copy()
        df_pira["sexo"] = df_pira["sexo"].astype(str).str.upper()
        df_pira["faixa_etaria"] = df_pira["faixa_etaria"].astype(str)
        df_pira["faixa_etaria"] = pd.Categorical(
            df_pira["faixa_etaria"],
            categories=categorias,
            ordered=True,
        )

        df_pira["faixa_etaria"] = df_pira["faixa_etaria"].astype(str).str.title()
        df_pira = df_pira[df_pira["faixa_etaria"].isin(categorias)]

        tabela = agrega_para_grafico(
            df_pira, ["faixa_etaria", "sexo"], indicador_selecionado
        ).rename(columns={"valor": "n"})

        pivot = tabela.pivot(index="faixa_etaria", columns="sexo", values="n").fillna(0)
        pivot = pivot.reindex(categorias).fillna(0)

        fig = go.Figure()

        for idx, sexo_cat in enumerate(pivot.columns):
            values = pivot[sexo_cat]
            # Primeiro sexo vai para o lado esquerdo (valores negativos),
            # os demais para o lado direito, mantendo o formato de pirâmide.
            x_vals = -values if idx == 0 else values

            s_norm = str(sexo_cat).strip().upper()
            if s_norm in ["M", "MASCULINO"]:
                cor_sexo = "#6794DC"  # azul para masculino
            elif s_norm in ["F", "FEMININO"]:
                cor_sexo = "#E86F86"  # rosa para feminino
            else:
                cor_sexo = "#A3A3A3"  # cinza para outros

            fig.add_bar(
                y=pivot.index,
                x=x_vals,
                name=str(sexo_cat),
                orientation="h",
                marker_color=cor_sexo,
                text=np.round(values, 2),
                textposition="outside",
            )


        max_abs = float(np.nanmax(np.abs(pivot.values))) if pivot.values.size > 0 else 0.0
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
        tick_vals = np.linspace(-max_abs, max_abs, 5)
        tick_text = [f"{abs(v):.1f}" for v in tick_vals]

        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_text,
            title=label_eixo_x(indicador_selecionado),
        )

        fig.update_yaxes(title="Faixa etária")

        fig.update_layout(
            barmode="relative",
            bargap=0.1,
            height=400,
            margin=dict(t=20, b=40),
            legend_title_text="Sexo",
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Necessário ter colunas 'faixa_etaria' e 'sexo'.")

with col_dir:
    # ----------------- Caráter do atendimento (CARD horizontal) -----------------
    st.subheader("Caráter do atendimento")
    if "carater_atendimento" in base_charts.columns:
        df_carater = agrega_para_grafico(
            base_charts,
            ["carater_atendimento"],
            indicador_selecionado,
        )
        df_carater = df_carater.sort_values("valor", ascending=False)
        fig_carater = card_bar_fig(
            df_carater,
            cat_col="carater_atendimento",
            indicador=indicador_selecionado,
            colors=["#6CC5C6", "#E39A8D", "#C28CCB", "#91A8C5"],
            height=120,
        )
        st.plotly_chart(
            fig_carater,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Coluna 'carater_atendimento' não encontrada.")

    st.subheader("Mapa de Internações por Região de Saúde")
    if {"uf", "regiao_saude", "cidade_moradia"}.issubset(base_charts.columns):
        df_geo = agrega_para_grafico(
            base_charts,
            ["uf", "regiao_saude", "cidade_moradia"],
            indicador_selecionado,
        ).rename(columns={"valor": "Pacientes/Internações"})

        df_geo_plot = df_geo.copy()

        if not df_geo_plot.empty:
            fig_treemap = px.treemap(
                df_geo_plot,
                path=["uf", "regiao_saude", "cidade_moradia"],
                values="Pacientes/Internações",
                color="Pacientes/Internações",
                color_continuous_scale="Blues",
            )

            fig_treemap.update_layout(
                margin=dict(t=40, l=0, r=0, b=0),
                height=450,
            )

            st.plotly_chart(
                fig_treemap,
                use_container_width=True,
                config={"displayModeBar": True},
            )
        else:
            st.info("Não há dados para o filtro selecionado.")
    else:
        st.info(
            "Necessário ter colunas 'uf', 'regiao_saude' e 'cidade_moradia' para o treemap."
        )

st.divider()
st.caption("Painel de Indicadores de Saúde – ACC • Dados anonimizados (2019–2025)")
