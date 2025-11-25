# app.py — Painel de Indicadores Cardiovasculares (2019–2025)
# Suporta: 1 arquivo Parquet consolidado OU 3 CSVs grandes (DuckDB)
# Requisitos: streamlit, pandas, numpy, plotly, python-dateutil, pyarrow, duckdb

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import plotly.express as px
import plotly.graph_objects as go
import duckdb

# --------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Painel de Indicadores Cardiovasculares",
    layout="wide",
    page_icon="❤️",
)

# CSS global (layout mais “limpo”)
st.markdown(
    """
    <style>
    /* diminuir padding geral */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    /* remover fundo cinza do sidebar */
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
    }
    /* títulos */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    /* cards de KPI */
    .kpi-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.6rem;
        border: 1px solid #eee;
        background: #ffffff;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #777;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 600;
        color: #222;
    }

    /* RADIO como “botões” para indicadores */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    div[data-testid="stRadio"] label {
        border-radius: 999px;
        border: 1px solid #e0e0e0;
        padding: 0.35rem 0.85rem;
        background-color: #f7f7f7;
        cursor: pointer;
        font-size: 0.78rem;
        transition: all 0.15s ease-in-out;
        white-space: nowrap;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #ff6b6b;
        background-color: #ffecec;
    }
    div[data-testid="stRadio"] input:checked + div {
        background-color: #ff6b6b !important;
        color: white !important;
        border-color: #ff6b6b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# UTILITÁRIOS
# --------------------------------------------------------------------


def fmt_int(x):
    if pd.isna(x):
        return "-"
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, casas=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{casas}f}".replace(".", ",")


def fmt_pct(x, casas=1):
    if pd.isna(x):
        return "-"
    return f"{x:.{casas}f}%".replace(".", ",")


def parse_date(col):
    def _parse_one(v):
        if pd.isna(v):
            return pd.NaT
        if isinstance(v, (datetime, pd.Timestamp)):
            return pd.to_datetime(v)
        try:
            return parser.parse(str(v), dayfirst=True)
        except Exception:
            return pd.NaT

    return col.apply(_parse_one)


# --------------------------------------------------------------------
# CARGA DE DADOS
# --------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _post_load(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # normaliza nomes de colunas
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # datas
    date_cols = [c for c in df.columns if c.lower().startswith("data_") or c.lower().startswith("dt_")]
    for c in date_cols:
        df[c] = parse_date(df[c])

    # ano da internação
    if "ano_internacao" not in df.columns:
        if "data_internacao" in df.columns:
            df["ano_internacao"] = df["data_internacao"].dt.year
        elif "data_entrada" in df.columns:
            df["ano_internacao"] = df["data_entrada"].dt.year

    if "ano" not in df.columns and "ano_internacao" in df.columns:
        df["ano"] = df["ano_internacao"]

    # idade como numérica
    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")

        # faixas etárias
        bins = [-1, 0, 8, 17, 26, 35, 44, 53, 62, 71, 80, 89, 200]
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
        df["faixa_etaria"] = pd.cut(df["idade"], bins=bins, labels=labels)

    # dias de permanência
    if {"data_internacao", "data_alta"}.issubset(df.columns):
        dias = (df["data_alta"] - df["data_internacao"]).dt.days
        dias = dias.replace([np.inf, -np.inf], np.nan)
        df["dias_permanencia"] = dias

    # sexo em categoria ordenada (M/F)
    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].astype(str).str.upper().str.strip()
        ordem = ["M", "F"]
        df["sexo_cat"] = pd.Categorical(df["sexo"], categories=ordem, ordered=True)

    # normaliza carater_atendimento
    cand_carater = [c for c in df.columns if "carater" in c.lower()]
    if cand_carater:
        c = cand_carater[0]
        df[c] = df[c].astype(str).str.upper().str.strip()
        df["carater_atendimento"] = df[c]

    return df


@st.cache_data(show_spinner=False)
def load_parquet(file) -> pd.DataFrame:
    return _post_load(pd.read_parquet(file))


@st.cache_resource(show_spinner=False)
def load_duckdb(csv_paths):
    """
    Carrega EVOLUÇÕES, PROCEDIMENTOS e CIDS via DuckDB
    e monta a view 'dataset' ligada por PRONTUARIO_ANONIMO.
    """
    con = duckdb.connect(database=":memory:")
    evo, proc, cti = csv_paths

    def make_view(view_name: str, path: str):
        if path is None:
            return
        path_esc = str(path).replace("'", "''")
        con.execute(
            f"""
            CREATE VIEW {view_name} AS
            SELECT *
            FROM read_csv_auto(
                '{path_esc}',
                header      = TRUE,
                delim       = ',',
                encoding    = 'utf-8',
                auto_detect = TRUE,
                ignore_errors = TRUE,
                sample_size = -1
            );
            """
        )

    make_view("evolu", evo)
    make_view("proced", proc)
    make_view("cids", cti)

    con.execute(
        """
        CREATE VIEW evolu_n AS
        SELECT
            lower(trim(CAST(prontuario_anonimo AS VARCHAR))) AS prontuario_anonimo,
            *
        FROM evolu;

        CREATE VIEW cids_n AS
        SELECT
            lower(trim(CAST(prontuario_anonimo AS VARCHAR))) AS prontuario_anonimo,
            *
        FROM cids;

        CREATE VIEW proc_n AS
        SELECT
            lower(trim(CAST(prontuario_anonimo AS VARCHAR))) AS prontuario_anonimo,
            *
        FROM proced;

        CREATE VIEW proc_agg AS
        SELECT
            prontuario_anonimo,
            COUNT(*)                        AS n_proced,
            ANY_VALUE(codigo_procedimento)  AS proc_prim,
            ANY_VALUE(procedimento)         AS proc_nome_prim,
            ANY_VALUE(natureza_agend)       AS natureza_agend
        FROM proc_n
        GROUP BY prontuario_anonimo;

        CREATE VIEW dataset AS
        SELECT
            e.*,
            c.* EXCLUDE (prontuario_anonimo),
            p.*
        FROM evolu_n e
        LEFT JOIN cids_n  c USING (prontuario_anonimo)
        LEFT JOIN proc_agg p USING (prontuario_anonimo);
        """
    )

    return con


def df_from_duckdb(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


# --------------------------------------------------------------------
# ENRIQUECIMENTO COM TABELAS AUXILIARES (CID, SIGTAP, GEO)
# --------------------------------------------------------------------


def enrich_with_aux_tables(df: pd.DataFrame, cid_file=None, sigtap_file=None, geo_file=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df_enriched = df.copy()

    # ----- CID-10 -----
    if cid_file is not None:
        try:
            cid_df = pd.read_csv(cid_file, dtype=str)
            cid_df.columns = [c.lower() for c in cid_df.columns]
            cid_cols = cid_df.columns

            cid_code_col = next((c for c in cid_cols if c.startswith("cid") and "cid3" not in c), None)
            if cid_code_col is None:
                cid_code_col = "cid"

            # normaliza código de 3 caracteres
            cid_df["cid3"] = cid_df[cid_code_col].str.upper().str.strip().str[:3]

            # assume que existem colunas de capítulo / grupo
            cap_col = next((c for c in cid_cols if "capitulo" in c), None)
            grp_col = next((c for c in cid_cols if "grupo" in c), None)

            merge_cols = ["cid3"]
            use_cols = ["cid3"]
            if cap_col:
                use_cols.append(cap_col)
            if grp_col:
                use_cols.append(grp_col)

            cid_small = cid_df[use_cols].drop_duplicates(subset=merge_cols)

            if "cid" in df_enriched.columns:
                df_enriched["cid3"] = df_enriched["cid"].astype(str).str.upper().str.strip().str[:3]
                df_enriched = df_enriched.merge(cid_small, on="cid3", how="left")
                if cap_col:
                    df_enriched = df_enriched.rename(columns={cap_col: "cid_capitulo"})
                if grp_col:
                    df_enriched = df_enriched.rename(columns={grp_col: "cid_grupo"})
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com tabela CID: {e}")

    # ----- SIGTAP -----
    if sigtap_file is not None and "proc_prim" in df_enriched.columns:
        try:
            stp = pd.read_csv(sigtap_file, dtype=str, sep=";")
            stp.columns = [c.lower() for c in stp.columns]
            stp["codigo"] = stp["codigo"].str.strip()

            sig_small = stp[
                [
                    "codigo",
                    "grupo",
                    "subgrupo",
                    "forma_organizacao",
                    "descricao",
                ]
            ].drop_duplicates(subset=["codigo"])

            df_enriched["proc_prim"] = df_enriched["proc_prim"].astype(str).str.strip()
            df_enriched = df_enriched.merge(
                sig_small,
                how="left",
                left_on="proc_prim",
                right_on="codigo",
            )
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com SIGTAP: {e}")

    # ----- GEO -----
    if geo_file is not None:
        try:
            geo_df = pd.read_csv(geo_file, sep=";", dtype=str)
            geo_df.columns = [c.lower() for c in geo_df.columns]

            if "cidade_moradia" in df_enriched.columns and {"no_municipio", "sg_uf"}.issubset(geo_df.columns):
                partes = df_enriched["cidade_moradia"].astype(str).str.split(",", n=1, expand=True)
                df_enriched["cidade_nome_norm"] = partes[0].str.upper().str.strip()
                if partes.shape[1] > 1:
                    df_enriched["uf_from_cidade"] = partes[1].str.upper().str.strip()
                else:
                    df_enriched["uf_from_cidade"] = np.nan

                geo_df["no_municipio_norm"] = geo_df["no_municipio"].astype(str).str.upper().str.strip()
                geo_df["sg_uf"] = geo_df["sg_uf"].astype(str).str.upper().str.strip()

                geo_small = geo_df[
                    ["no_municipio_norm", "sg_uf", "no_macrorregional", "no_cir_padrao"]
                ].drop_duplicates(subset=["no_municipio_norm", "sg_uf"])

                df_enriched = df_enriched.merge(
                    geo_small,
                    how="left",
                    left_on=["cidade_nome_norm", "uf_from_cidade"],
                    right_on=["no_municipio_norm", "sg_uf"],
                )
                df_enriched = df_enriched.rename(
                    columns={
                        "sg_uf": "uf",
                        "no_macrorregional": "macroregiao",
                        "no_cir_padrao": "regiao_saude",
                    }
                )

                df_enriched = df_enriched.drop(
                    columns=["cidade_nome_norm", "no_municipio_norm"],
                    errors="ignore",
                )
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com regiões de saúde: {e}")

    return df_enriched


# --------------------------------------------------------------------
# MÉTRICAS / KPI
# --------------------------------------------------------------------


def pacientes_unicos(df: pd.DataFrame) -> pd.DataFrame:
    if {"prontuario_anonimo", "data_internacao"}.issubset(df.columns):
        return (
            df.sort_values(["prontuario_anonimo", "data_internacao"])
            .groupby("prontuario_anonimo", as_index=False)
            .tail(1)
        )
    if "prontuario_anonimo" in df.columns:
        return df.drop_duplicates(subset=["prontuario_anonimo"])
    return df


def marcar_obito_periodo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"data_internacao", "data_alta"}.issubset(df.columns):
        e = df.copy()
        if "data_obito" in e.columns:
            e["obito_no_periodo"] = (
                (e["data_obito"].notna())
                & (e["data_obito"] >= e["data_internacao"])
                & (e["data_obito"] <= (e["data_alta"] - pd.Timedelta(days=1)))
            )
        elif "evolucao" in e.columns:
            e["obito_no_periodo"] = e["evolucao"].astype(str).str.contains(
                "ÓBITO", case=False, na=False
            )
        else:
            e["obito_no_periodo"] = False

        if "codigo_internacao" in e.columns:
            aux = (
                e[["codigo_internacao", "obito_no_periodo"]]
                .groupby("codigo_internacao", as_index=False)["obito_no_periodo"]
                .max()
            )
            df = df.merge(aux, on="codigo_internacao", how="left")
        else:
            df["obito_no_periodo"] = e["obito_no_periodo"]
    else:
        df["obito_no_periodo"] = False
    df["obito_no_periodo"] = df["obito_no_periodo"].fillna(False)
    return df


def marcar_uti_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "codigo_internacao" not in df.columns:
        df["uti_flag"] = False
        return df

    e = df.copy()
    uti_flag = pd.Series(False, index=e.index)

    if "dt_entrada_cti" in e.columns:
        uti_flag = uti_flag | e["dt_entrada_cti"].notna()
    if "dt_saida_cti" in e.columns:
        uti_flag = uti_flag | e["dt_saida_cti"].notna()

    for cand in ["uti", "internacao_uti", "teve_uti"]:
        if cand in e.columns:
            uti_flag = uti_flag | e[cand].astype(str).str.upper().isin(
                ["1", "S", "SIM", "TRUE", "VERDADEIRO"]
            )

    e["uti_flag"] = uti_flag
    aux = (
        e[["codigo_internacao", "uti_flag"]]
        .groupby("codigo_internacao", as_index=False)["uti_flag"]
        .max()
    )
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["uti_flag"] = df["uti_flag"].fillna(False)
    return df


def marcar_reinternacoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ok = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}.issubset(
        df.columns
    )
    df["reint_30d_proc"] = False
    df["reint_30d_alta"] = False
    if not ok:
        return df

    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta_proc"] = (s["next_dt_internacao"] - s["data_internacao"]).dt.days
    s["delta_pos_alta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta_pos_alta"] <= 1

    s["reint_30d_proc"] = s["delta_proc"].between(0, 30, inclusive="both") & (~s["transfer"])
    s["reint_30d_alta"] = s["delta_pos_alta"].between(0, 30, inclusive="both") & (~s["transfer"])

    aux = (
        s[["codigo_internacao", "reint_30d_proc", "reint_30d_alta"]]
        .groupby("codigo_internacao", as_index=False)[["reint_30d_proc", "reint_30d_alta"]]
        .max()
    )
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["reint_30d_proc"] = df["reint_30d_proc"].fillna(False)
    df["reint_30d_alta"] = df["reint_30d_alta"].fillna(False)
    return df


def marcar_mort_30d_proc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "data_obito" not in df.columns or "codigo_internacao" not in df.columns:
        df["obito_30d_proc"] = False
        return df

    e = df.copy()
    cand_datas = [
        "data_procedimento",
        "data_cirurgia",
        "data_cirurgia_min",
        "data_internacao",
    ]
    data_proc_col = next((c for c in cand_datas if c in e.columns), None)
    if not data_proc_col:
        df["obito_30d_proc"] = False
        return df

    e = (
        e.sort_values(data_proc_col)
        .groupby("codigo_internacao", as_index=False)
        .first()
    )

    e["delta"] = (e["data_obito"] - e[data_proc_col]).dt.days
    e["obito_30d_proc"] = e["delta"].between(0, 30, inclusive="both")

    aux = e[["codigo_internacao", "obito_30d_proc"]]
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["obito_30d_proc"] = df["obito_30d_proc"].fillna(False)
    return df


def marcar_mort_30d_alta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"data_alta", "data_obito", "codigo_internacao"}.issubset(df.columns):
        df["obito_30d_alta"] = False
        return df

    e = (
        df.sort_values("data_alta")
        .groupby("codigo_internacao", as_index=False)
        .tail(1)
    )

    e["delta"] = (e["data_obito"] - e["data_alta"]).dt.days
    e["obito_30d_alta"] = e["delta"].between(0, 30, inclusive="both")

    aux = e[["codigo_internacao", "obito_30d_alta"]]
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["obito_30d_alta"] = df["obito_30d_alta"].fillna(False)
    return df


def kpis(df_eventos: pd.DataFrame, df_pacientes: pd.DataFrame):
    pacientes = (
        df_pacientes["prontuario_anonimo"].nunique()
        if "prontuario_anonimo" in df_pacientes
        else np.nan
    )
    internacoes = (
        df_eventos["codigo_internacao"].nunique()
        if "codigo_internacao" in df_eventos
        else len(df_eventos)
    )
    tmi = (
        df_eventos["dias_permanencia"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
        if "dias_permanencia" in df_eventos
        else np.nan
    )

    # mortalidade hospitalar (%)
    mort_hosp = np.nan
    if {"data_internacao", "data_alta"}.issubset(df_eventos.columns):
        e = df_eventos.copy()
        if "data_obito" in e.columns:
            e["obito_no_periodo"] = (
                (e["data_obito"].notna())
                & (e["data_obito"] >= e["data_internacao"])
                & (e["data_obito"] <= (e["data_alta"] - pd.Timedelta(days=1)))
            )
        elif "evolucao" in e.columns:
            e["obito_no_periodo"] = e["evolucao"].astype(str).str.contains(
                "ÓBITO", case=False, na=False
            )
        else:
            e["obito_no_periodo"] = False

        denom = (
            e["codigo_internacao"].nunique()
            if "codigo_internacao" in e
            else len(e)
        )
        numer_df = e.loc[e["obito_no_periodo"]]
        numer = (
            numer_df["codigo_internacao"].nunique()
            if "codigo_internacao" in e
            else len(numer_df)
        )
        mort_hosp = (numer / denom * 100) if denom else np.nan

    return pacientes, internacoes, tmi, mort_hosp


def reinternacao_30d_pos_proced(df: pd.DataFrame):
    ok = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}.issubset(
        df.columns
    )
    if not ok:
        return np.nan
    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta_proc"] = (s["next_dt_internacao"] - s["data_internacao"]).dt.days
    s["delta_pos_alta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta_pos_alta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[
        s["delta_proc"].between(0, 30, inclusive="both") & (~s["transfer"])
    ]["codigo_internacao"].nunique()
    return (numer / base * 100) if base else np.nan


def reinternacao_30d_pos_alta(df: pd.DataFrame):
    ok = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}.issubset(
        df.columns
    )
    if not ok:
        return np.nan
    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[
        s["delta"].between(0, 30, inclusive="both") & (~s["transfer"])
    ]["codigo_internacao"].nunique()
    return (numer / base * 100) if base else np.nan


def internacao_uti_pct(df: pd.DataFrame):
    if "codigo_internacao" not in df.columns:
        return np.nan

    e = df.copy()
    uti_flag = pd.Series(False, index=e.index)

    if "dt_entrada_cti" in e.columns:
        uti_flag = uti_flag | e["dt_entrada_cti"].notna()
    if "dt_saida_cti" in e.columns:
        uti_flag = uti_flag | e["dt_saida_cti"].notna()

    for cand in ["uti", "internacao_uti", "teve_uti"]:
        if cand in e.columns:
            uti_flag = uti_flag | e[cand].astype(str).str.upper().isin(
                ["1", "S", "SIM", "TRUE", "VERDADEIRO"]
            )

    e["uti_flag"] = uti_flag
    denom = e["codigo_internacao"].nunique()
    numer = e.loc[e["uti_flag"], "codigo_internacao"].nunique()
    return (numer / denom * 100) if denom else np.nan


def tempo_medio_uti_dias(df: pd.DataFrame):
    if not {"dt_entrada_cti", "dt_saida_cti"}.issubset(df.columns):
        return np.nan

    e = df.copy()
    dias_uti = (e["dt_saida_cti"] - e["dt_entrada_cti"]).dt.days
    dias_uti = dias_uti.replace([np.inf, -np.inf], np.nan).dropna()
    return dias_uti.mean() if len(dias_uti) else np.nan


def mortalidade_30d_pos_proced(df: pd.DataFrame):
    if "data_obito" not in df.columns or "codigo_internacao" not in df.columns:
        return np.nan

    e = df.copy()
    cand_datas = [
        "data_procedimento",
        "data_cirurgia",
        "data_cirurgia_min",
        "data_cirurgia_max",
        "data_internacao",
    ]
    data_proc_col = next((c for c in cand_datas if c in e.columns), None)
    if not data_proc_col:
        return np.nan

    e = (
        e.sort_values(data_proc_col)
        .groupby("codigo_internacao", as_index=False)
        .first()
    )

    e["delta"] = (e["data_obito"] - e[data_proc_col]).dt.days
    e["obito_30d_proc"] = e["delta"].between(0, 30, inclusive="both")

    denom = e["codigo_internacao"].nunique()
    numer = e.loc[e["obito_30d_proc"], "codigo_internacao"].nunique()
    return (numer / denom * 100) if denom else np.nan


def mortalidade_30d_pos_alta(df: pd.DataFrame):
    if not {"data_alta", "data_obito", "codigo_internacao"}.issubset(df.columns):
        return np.nan

    e = (
        df.sort_values("data_alta")
        .groupby("codigo_internacao", as_index=False)
        .tail(1)
    )

    e["delta"] = (e["data_obito"] - e["data_alta"]).dt.days
    e["obito_30d_alta"] = e["delta"].between(0, 30, inclusive="both")

    denom = e["codigo_internacao"].nunique()
    numer = e.loc[e["obito_30d_alta"], "codigo_internacao"].nunique()
    return (numer / denom * 100) if denom else np.nan


# --------------------------------------------------------------------
# AUXILIARES DO INDICADOR SELECIONADO (base + peso)
# --------------------------------------------------------------------


def definir_base_para_indicador(indicador, df_eventos, df_pacientes):
    if indicador == "Quantidade de pacientes":
        return df_pacientes.copy()
    return df_eventos.copy()


def _safe_bool_int(series: pd.Series) -> pd.Series:
    s = series.fillna(False)
    if s.dtype == bool:
        return s.astype(int)
    # fallback
    return s.astype(str).str.upper().isin(
        ["1", "TRUE", "VERDADEIRO", "S", "SIM"]
    ).astype(int)


def adicionar_peso_por_indicador(df: pd.DataFrame, indicador: str) -> pd.DataFrame:
    df = df.copy()
    df["peso"] = 0.0

    if indicador in ["Quantidade de pacientes", "Quantidade de internações"]:
        df["peso"] = 1.0

    elif indicador == "Quantidade de procedimentos":
        df["peso"] = df.get("n_proced", 0).fillna(0)

    elif indicador == "Tempo médio de internação (dias)":
        if "dias_permanencia" in df.columns:
            df["peso"] = df["dias_permanencia"].clip(lower=0).fillna(0)
        else:
            df["peso"] = 0.0

    elif indicador == "Internação em UTI (%)":
        df = marcar_uti_flag(df)
        df["peso"] = _safe_bool_int(df["uti_flag"])

    elif indicador == "Tempo médio de internação em UTI (dias)":
        if {"dt_entrada_cti", "dt_saida_cti"}.issubset(df.columns):
            dias_uti = (df["dt_saida_cti"] - df["dt_entrada_cti"]).dt.days
            df["peso"] = dias_uti.clip(lower=0).fillna(0)
        else:
            df["peso"] = 0.0

    elif indicador == "Reinternação em até 30 dias do procedimento (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = _safe_bool_int(df["reint_30d_proc"])

    elif indicador == "Reinternação em até 30 dias da alta (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = _safe_bool_int(df["reint_30d_alta"])

    elif indicador == "Mortalidade hospitalar (%)":
        df = marcar_obito_periodo(df)
        df["peso"] = _safe_bool_int(df["obito_no_periodo"])

    elif indicador == "Mortalidade em até 30 dias do procedimento (%)":
        df = marcar_mort_30d_proc(df)
        df["peso"] = _safe_bool_int(df["obito_30d_proc"])

    elif indicador == "Mortalidade em até 30 dias da alta (%)":
        df = marcar_mort_30d_alta(df)
        df["peso"] = _safe_bool_int(df["obito_30d_alta"])

    return df


# --------------------------------------------------------------------
# FILTROS
# --------------------------------------------------------------------


def build_filters(df: pd.DataFrame):
    if df is None or df.empty:
        st.error("Dataset não carregado.")
        st.stop()

    st.sidebar.header("Filtros")

    anos_col = "ano_internacao" if "ano_internacao" in df.columns else ("ano" if "ano" in df.columns else None)
    if anos_col:
        anos = sorted(df[anos_col].dropna().unique().tolist())
    else:
        anos = []
    ano_sel = st.sidebar.multiselect("Ano da internação", anos, default=anos)

    if "idade" in df.columns and df["idade"].notna().any():
        idade_min, idade_max = int(np.nanmin(df["idade"])), int(np.nanmax(df["idade"]))
    else:
        idade_min, idade_max = 0, 120
    idade_sel = st.sidebar.slider(
        "Idade", min_value=0, max_value=max(idade_max, 1), value=(idade_min, idade_max), step=1
    )

    estado_col = next(
        (c for c in df.columns if c.lower() in ["estado_residencia", "uf_residencia", "uf", "estado", "sigla_uf"]),
        None,
    )
    estados_sel = []
    if estado_col:
        estados = sorted(df[estado_col].dropna().astype(str).unique().tolist())
        estados_sel = st.sidebar.multiselect("Estado de residência", estados, default=estados)

    regiao_col = next(
        (c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()),
        None,
    )
    regioes_sel = []
    if regiao_col:
        regioes = sorted(df[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = st.sidebar.multiselect("Região de saúde", regioes, default=regioes)

    cidade_col = "cidade_moradia" if "cidade_moradia" in df.columns else None
    cidades_sel = []
    if cidade_col:
        cidade_vals = sorted(df[cidade_col].dropna().astype(str).unique().tolist())
        default_cidades = cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        cidades_sel = st.sidebar.multiselect(
            "Município de residência (amostra)", cidade_vals, default=default_cidades
        )

    return {
        "anos_col": anos_col,
        "ano_sel": ano_sel,
        "idade_sel": idade_sel,
        "estado_col": estado_col,
        "estados_sel": estados_sel,
        "regiao_col": regiao_col,
        "regioes_sel": regioes_sel,
        "cidade_col": cidade_col,
        "cidades_sel": cidades_sel,
    }


def apply_filters(df: pd.DataFrame, filtros: dict):
    df_f = df.copy()
    anos_col = filtros["anos_col"]
    if anos_col and filtros["ano_sel"]:
        df_f = df_f[df_f[anos_col].isin(filtros["ano_sel"])]

    if "idade" in df_f.columns:
        imin, imax = filtros["idade_sel"]
        df_f = df_f[df_f["idade"].between(imin, imax)]

    if filtros["estado_col"] and filtros["estados_sel"]:
        df_f = df_f[df_f[filtros["estado_col"]].astype(str).isin(filtros["estados_sel"])]

    if filtros["regiao_col"] and filtros["regioes_sel"]:
        df_f = df_f[df_f[filtros["regiao_col"]].astype(str).isin(filtros["regioes_sel"])]

    if filtros["cidade_col"] and filtros["cidades_sel"]:
        df_f = df_f[df_f[filtros["cidade_col"]].astype(str).isin(filtros["cidades_sel"])]

    return df_f


def describe_active_filters(filtros: dict):
    desc = []
    if filtros["ano_sel"]:
        desc.append(f"Ano: {', '.join(map(str, filtros['ano_sel']))}")
    imin, imax = filtros["idade_sel"]
    desc.append(f"Idade: {imin}–{imax} anos")
    if filtros["estados_sel"]:
        desc.append(f"UF: {', '.join(filtros['estados_sel'])}")
    if filtros["regioes_sel"]:
        desc.append(f"Região de saúde: {', '.join(filtros['regioes_sel'])}")
    if filtros["cidades_sel"]:
        desc.append(f"Municípios: {', '.join(filtros['cidades_sel'])}")
    return " | ".join(desc) if desc else "Nenhum filtro aplicado"


# --------------------------------------------------------------------
# INTERFACE – CARGA DE ARQUIVOS
# --------------------------------------------------------------------

st.title("Painel de Indicadores Cardiovasculares")

with st.sidebar.expander("📂 Fonte de dados", expanded=True):
    modo = st.radio(
        "Formato da base",
        ["Parquet único", "3 CSVs (evolução / procedimentos / CIDs)"],
        index=0,
    )

    parquet_file = None
    csv_evolu = csv_proc = csv_cids = None
    if modo == "Parquet único":
        parquet_file = st.file_uploader("Base consolidada (Parquet)", type=["parquet"])
    else:
        csv_evolu = st.file_uploader("Evoluções (CSV)", type=["csv"], key="csv_evolu")
        csv_proc = st.file_uploader("Procedimentos (CSV)", type=["csv"], key="csv_proc")
        csv_cids = st.file_uploader("CIDs (CSV)", type=["csv"], key="csv_cids")

    st.markdown("---")
    cid_file = st.file_uploader("Tabela CID-10 (opcional)", type=["csv"], key="cid_file")
    sigtap_file = st.file_uploader("SIGTAP (opcional)", type=["csv"], key="sigtap_file")
    geo_file = st.file_uploader("Tabela de municípios / regiões (opcional)", type=["csv"], key="geo_file")

if modo == "Parquet único":
    if not parquet_file:
        st.info("Carregue a base Parquet para começar.")
        st.stop()
    df_raw = load_parquet(parquet_file)
else:
    if not all([csv_evolu, csv_proc, csv_cids]):
        st.info("Carregue os 3 CSVs (evoluções, procedimentos e CIDs) para começar.")
        st.stop()
    con = load_duckdb((csv_evolu, csv_proc, csv_cids))
    df_raw = _post_load(df_from_duckdb(con, "SELECT * FROM dataset"))

df = enrich_with_aux_tables(df_raw, cid_file=cid_file, sigtap_file=sigtap_file, geo_file=geo_file)

# --------------------------------------------------------------------
# APLICAÇÃO DOS FILTROS
# --------------------------------------------------------------------

filtros = build_filters(df)
df_filt = apply_filters(df, filtros)
df_pacientes = pacientes_unicos(df_filt)

st.markdown(
    f"**Filtros atuais:** {describe_active_filters(filtros)}"
)

# --------------------------------------------------------------------
# KPIs GERAIS
# --------------------------------------------------------------------

pac, intern, tmi, mort_hosp = kpis(df_filt, df_pacientes)
reint_proc = reinternacao_30d_pos_proced(df_filt)
reint_alta = reinternacao_30d_pos_alta(df_filt)
uti_pct = internacao_uti_pct(df_filt)
tmi_uti = tempo_medio_uti_dias(df_filt)
mort_30d_proc = mortalidade_30d_pos_proced(df_filt)
mort_30d_alta = mortalidade_30d_pos_alta(df_filt)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Pacientes (distintos)</div>
            <div class="kpi-value">{fmt_int(pac)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Internações</div>
            <div class="kpi-value">{fmt_int(intern)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Tempo médio internação (dias)</div>
            <div class="kpi-value">{fmt_float(tmi)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Reinternação 30d (procedimento)</div>
            <div class="kpi-value">{fmt_pct(reint_proc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mortalidade hospitalar</div>
            <div class="kpi-value">{fmt_pct(mort_hosp)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# segunda linha de KPIs
c6, c7, c8, c9 = st.columns(4)
with c6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Reinternação 30d pós-alta</div>
            <div class="kpi-value">{fmt_pct(reint_alta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c7:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Internação em UTI</div>
            <div class="kpi-value">{fmt_pct(uti_pct)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c8:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Tempo médio em UTI (dias)</div>
            <div class="kpi-value">{fmt_float(tmi_uti)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c9:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mort. 30d pós-alta</div>
            <div class="kpi-value">{fmt_pct(mort_30d_alta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --------------------------------------------------------------------
# INDICADOR SELECIONADO – BOTÕES
# --------------------------------------------------------------------

st.subheader("Indicadores disponíveis")

lista_indicadores = [
    "Quantidade de pacientes",
    "Quantidade de internações",
    "Quantidade de procedimentos",
    "Tempo médio de internação (dias)",
    "Internação em UTI (%)",
    "Tempo médio de internação em UTI (dias)",
    "Reinternação em até 30 dias do procedimento (%)",
    "Reinternação em até 30 dias da alta (%)",
    "Mortalidade hospitalar (%)",
    "Mortalidade em até 30 dias do procedimento (%)",
    "Mortalidade em até 30 dias da alta (%)",
]

indicador_selecionado = st.radio(
    "Selecione o indicador para detalhar e para o comparativo anual:",
    lista_indicadores,
    horizontal=True,
)

# valor do indicador selecionado
valor_indicador = np.nan
if indicador_selecionado == "Quantidade de pacientes":
    valor_indicador = pac
elif indicador_selecionado == "Quantidade de internações":
    valor_indicador = intern
elif indicador_selecionado == "Quantidade de procedimentos":
    valor_indicador = df_filt.get("n_proced", pd.Series(dtype=float)).sum()
elif indicador_selecionado == "Tempo médio de internação (dias)":
    valor_indicador = tmi
elif indicador_selecionado == "Internação em UTI (%)":
    valor_indicador = uti_pct
elif indicador_selecionado == "Tempo médio de internação em UTI (dias)":
    valor_indicador = tmi_uti
elif indicador_selecionado == "Reinternação em até 30 dias do procedimento (%)":
    valor_indicador = reint_proc
elif indicador_selecionado == "Reinternação em até 30 dias da alta (%)":
    valor_indicador = reint_alta
elif indicador_selecionado == "Mortalidade hospitalar (%)":
    valor_indicador = mort_hosp
elif indicador_selecionado == "Mortalidade em até 30 dias do procedimento (%)":
    valor_indicador = mort_30d_proc
elif indicador_selecionado == "Mortalidade em até 30 dias da alta (%)":
    valor_indicador = mort_30d_alta

if "percentual" in indicador_selecionado.lower() or "(%)" in indicador_selecionado:
    val_fmt = fmt_pct(valor_indicador)
else:
    # se for tempo, ainda é float
    if "tempo médio" in indicador_selecionado.lower():
        val_fmt = fmt_float(valor_indicador)
    else:
        val_fmt = fmt_int(valor_indicador)

st.markdown(
    f"""
    <div class="kpi-card" style="margin-top:0.6rem;margin-bottom:0.2rem;">
        <div class="kpi-label">Valor do indicador selecionado</div>
        <div class="kpi-value">{val_fmt}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Base para gráficos de detalhe
base_ind = definir_base_para_indicador(indicador_selecionado, df_filt, df_pacientes)
base_charts = adicionar_peso_por_indicador(base_ind, indicador_selecionado)

st.markdown("---")

# --------------------------------------------------------------------
# GRÁFICOS PRINCIPAIS
# --------------------------------------------------------------------

# 1ª linha: Sexo | Caráter | Mapa por Estado/Região/Município | KPI indicador
csexo, ccar, cmap, c_kpi = st.columns([1.1, 1.1, 2.0, 1.1])

# Sexo
with csexo:
    st.markdown("### Sexo")
    if "sexo" in base_charts.columns:
        df_sexo = (
            base_charts.groupby("sexo", dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        fig = px.bar(
            df_sexo,
            x="sexo",
            y="valor",
            text="valor",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de sexo.")

# Caráter do atendimento
with ccar:
    st.markdown("### Caráter do atendimento")
    if "carater_atendimento" in base_charts.columns:
        df_car = (
            base_charts.groupby("carater_atendimento", dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        fig = px.bar(
            df_car,
            x="carater_atendimento",
            y="valor",
            text="valor",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de caráter de atendimento.")

# Treemap Estado → Região → Município
with cmap:
    st.markdown("### Estado ➜ Região de Saúde ➜ Município de residência")
    if any(c in base_charts.columns for c in ["uf", "regiao_saude", "cidade_moradia"]):
        tmp = base_charts.copy()
        if "uf" not in tmp.columns:
            tmp["uf"] = "Sem UF"
        if "regiao_saude" not in tmp.columns:
            tmp["regiao_saude"] = "Sem região"
        if "cidade_moradia" not in tmp.columns:
            tmp["cidade_moradia"] = "Sem município"

        df_geo = (
            tmp.groupby(["uf", "regiao_saude", "cidade_moradia"], dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        fig = px.treemap(
            df_geo,
            path=["uf", "regiao_saude", "cidade_moradia"],
            values="valor",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação geográfica para montar o mapa.")

# KPI em destaque (já mostramos acima – aqui deixo só o texto do selecionado)
with c_kpi:
    st.markdown("### Indicador ativo")
    st.write(indicador_selecionado)
    st.markdown(f"**Valor:** {val_fmt}")

st.markdown("---")

# 2ª linha: Pirâmide etária | Raça/Cor x Sexo
cpir, craca = st.columns([2, 2])

with cpir:
    st.markdown("### Pirâmide Etária")
    if {"faixa_etaria", "sexo"}.issubset(base_charts.columns):
        df_pir = (
            base_charts.groupby(["faixa_etaria", "sexo"], dropna=False)["peso"]
            .sum()
            .reset_index()
        )
        df_pir["valor_plot"] = df_pir.apply(
            lambda r: -r["peso"] if r["sexo"] == "M" else r["peso"], axis=1
        )
        fig = go.Figure()
        for sexo_val, cor in [("F", None), ("M", None)]:
            sub = df_pir[df_pir["sexo"] == sexo_val]
            fig.add_bar(
                y=sub["faixa_etaria"],
                x=sub["valor_plot"],
                name="Feminino" if sexo_val == "F" else "Masculino",
                orientation="h",
            )
        fig.update_layout(
            barmode="relative",
            height=320,
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Pacientes / internações (peso)",
            yaxis_title="Faixa etária",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de idade/sexo para a pirâmide.")

with craca:
    st.markdown("### Raça/Cor × Sexo")
    if {"raca_cor", "sexo"}.issubset(base_charts.columns):
        df_raca = (
            base_charts.groupby(["raca_cor", "sexo"], dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        fig = px.bar(
            df_raca,
            x="raca_cor",
            y="valor",
            color="sexo",
            barmode="group",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title=None,
            yaxis_title="Peso",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de raça/cor e sexo.")

st.markdown("---")

# 3ª linha: Procedimentos (amostra) | CID (capítulo/grupo) – amostra
cproc, ccid = st.columns([2, 2])

with cproc:
    st.markdown("### Procedimentos (amostra)")
    if "proc_nome_prim" in base_charts.columns:
        df_proc = (
            base_charts.groupby("proc_nome_prim", dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
            .sort_values("valor", ascending=False)
            .head(15)
        )
        fig = px.bar(
            df_proc,
            x="valor",
            y="proc_nome_prim",
            orientation="h",
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Peso (amostra)",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de procedimentos.")

with ccid:
    st.markdown("### CID (capítulo / grupo) – amostra")
    if {"cid_capitulo", "cid_grupo"}.issubset(base_charts.columns):
        df_cid = (
            base_charts.groupby(["cid_capitulo", "cid_grupo"], dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
            .sort_values("valor", ascending=False)
            .head(20)
        )
        df_cid["label"] = df_cid["cid_capitulo"].astype(str) + " - " + df_cid["cid_grupo"].astype(
            str
        )
        fig = px.bar(
            df_cid,
            x="valor",
            y="label",
            orientation="h",
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Peso (amostra)",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sem informação de capítulo/grupo CID-10.")

st.markdown("---")

# --------------------------------------------------------------------
# COMPARATIVO ANUAL DO INDICADOR
# --------------------------------------------------------------------

st.subheader("Comparativo anual do indicador selecionado")

anos_col = filtros["anos_col"]
if anos_col and anos_col in base_charts.columns:
    serie = (
        base_charts.groupby(anos_col, dropna=False)["peso"]
        .sum()
        .reset_index()
        .rename(columns={"peso": "valor"})
        .sort_values(anos_col)
    )
    fig = px.bar(
        serie,
        x=anos_col,
        y="valor",
        text="valor",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="Ano",
        yaxis_title="Peso do indicador",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("A base não possui coluna de ano para montar o comparativo anual.")
