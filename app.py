import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os
import tempfile

st.set_page_config(page_title="Perfil dos Pacientes", layout="wide")

# --------------------------------------------------------------------
# FUNÇÕES DE CARGA E PRÉ-PROCESSAMENTO
# --------------------------------------------------------------------


def _post_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Datas
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

    # Ano da internação
    if (
        "data_internacao" in df.columns
        and "ano_internacao" not in df.columns
        and "ano" not in df.columns
    ):
        df["ano_internacao"] = df["data_internacao"].dt.year

    # Numéricos básicos
    for c in ["idade", "ano", "ano_internacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dias de permanência
    if {"data_internacao", "data_alta"}.issubset(df.columns):
        df["dias_permanencia"] = (df["data_alta"] - df["data_internacao"]).dt.days

    # Faixa etária
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

    # Deduplicação básica
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


@st.cache_resource(show_spinner=False)
def load_duckdb(csv_paths):
    con = duckdb.connect(database=":memory:")
    evo, proc, cti = csv_paths

    def make_view(view_name: str, path: str):
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
            * EXCLUDE (prontuario_anonimo)
        FROM evolu;

        CREATE VIEW cids_n AS
        SELECT
            lower(trim(CAST(prontuario_anonimo AS VARCHAR))) AS prontuario_anonimo,
            * EXCLUDE (prontuario_anonimo)
        FROM cids;

        -- Base de procedimentos com código de internação (normalizado)
        CREATE VIEW proc_n AS
        SELECT
            lower(trim(CAST(prontuario_anonimo AS VARCHAR))) AS prontuario_anonimo,
            COALESCE(
                NULLIF(trim(CAST(codigo_internacao AS VARCHAR)), ''),
                'SEM_' || lower(trim(CAST(prontuario_anonimo AS VARCHAR))) || '_' ||
                COALESCE(CAST(CAST(data_internacao AS DATE) AS VARCHAR), 'NA') || '_' ||
                COALESCE(CAST(CAST(data_alta AS DATE) AS VARCHAR), 'NA')
            ) AS codigo_internacao,
            * EXCLUDE (prontuario_anonimo, codigo_internacao)
        FROM proced;

        -- Agregação por internação (1 linha por CODIGO_INTERNACAO)
        CREATE VIEW internacoes_base AS
        SELECT
            prontuario_anonimo,
            codigo_internacao,
            MIN(data_internacao)             AS data_internacao,
            MIN(data_cirurgia)               AS data_cirurgia_min,
            MAX(data_cirurgia)               AS data_cirurgia_max,
            MAX(data_alta)                   AS data_alta,
            MAX(data_obito)                  AS data_obito,
            ANY_VALUE(natureza_agend)        AS natureza_agend,
            COUNT(*)                         AS n_proced,
            ANY_VALUE(codigo_procedimento)   AS proc_prim,
            ANY_VALUE(procedimento)          AS proc_nome_prim
        FROM proc_n
        GROUP BY prontuario_anonimo, codigo_internacao;

        -- Dataset final (baseado em internações)
        CREATE VIEW dataset AS
        SELECT
            i.*,
            e.* EXCLUDE (prontuario_anonimo),
            c.* EXCLUDE (prontuario_anonimo)
        FROM internacoes_base i
        LEFT JOIN evolu_n e USING (prontuario_anonimo)
        LEFT JOIN cids_n  c USING (prontuario_anonimo);

        -- Base de pacientes (caracterização), independente de ter internação no período
        CREATE VIEW pacientes_base AS
        SELECT
            e.*,
            c.* EXCLUDE (prontuario_anonimo)
        FROM evolu_n e
        LEFT JOIN cids_n c USING (prontuario_anonimo);
        """
    )
    return con


def df_from_duckdb(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


@st.cache_data
def load_aux_tables():
    """Carrega tabelas auxiliares (se existirem no diretório do app)."""
    try:
        cid_df = pd.read_csv("listacids.csv", dtype=str)
    except FileNotFoundError:
        cid_df = None

    try:
        sigtap_df = pd.read_csv("listaprocedimentos.csv", dtype=str)
    except FileNotFoundError:
        sigtap_df = None

    try:
        geo_df = pd.read_csv("regioesdesaude.csv", dtype=str)
    except FileNotFoundError:
        geo_df = None

    return cid_df, sigtap_df, geo_df


def enrich_with_aux_tables(df: pd.DataFrame, cid_df=None, sigtap_df=None, geo_df=None) -> pd.DataFrame:
    """
    Enriquecer o dataframe principal com:
      - Capítulo / grupo CID-10
      - Grupos de procedimento SIGTAP
      - UF / Macrorregião / Região de Saúde
    """
    if df is None:
        return df
    df_enriched = df.copy()

    # ---------------- CID-10 ----------------
    if cid_df is not None:
        try:
            cid_df = cid_df.copy()
            cid_df.columns = [c.lower() for c in cid_df.columns]
            cid_code_col = next((c for c in cid_df.columns if "cid" in c), None)

            if cid_code_col and ("cid" in df_enriched.columns or "cids" in df_enriched.columns):
                if "cid" not in df_enriched.columns and "cids" in df_enriched.columns:
                    df_enriched["cid"] = (
                        df_enriched["cids"]
                        .astype(str)
                        .str.split(",")
                        .str[0]
                        .str.strip()
                        .str.upper()
                    )

                df_enriched["cid3"] = df_enriched["cid"].astype(str).str.strip().str.upper().str[:3]
                cid_df["cid3"] = cid_df[cid_code_col].astype(str).str.strip().str.upper().str[:3]

                keep_cols = ["cid3"]
                keep_cols += [c for c in cid_df.columns if any(k in c for k in ["capítulo", "capitulo", "grupo"])]
                cid_small = cid_df[keep_cols].drop_duplicates(subset=["cid3"])
                df_enriched = df_enriched.merge(cid_small, how="left", on="cid3")

                rename_cols = {}
                for c in df_enriched.columns:
                    cl = c.lower()
                    if "capítulo" in cl or "capitulo" in cl:
                        rename_cols[c] = "cid_capitulo"
                    elif cl == "grupo":
                        rename_cols[c] = "cid_grupo"
                if rename_cols:
                    df_enriched = df_enriched.rename(columns=rename_cols)
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com CID-10: {e}")

    # ---------------- SIGTAP ----------------
    if sigtap_df is not None:
        try:
            sig_df = sigtap_df.copy()
            sig_df.columns = [c.lower() for c in sig_df.columns]
            sig_code_col = next(
                (c for c in sig_df.columns if "proced" in c and ("cod" in c or "codigo" in c)),
                None,
            )
            proc_col = None
            for cand in ["proc_prim", "codigo_procedimento", "cod_procedimento"]:
                if cand in df_enriched.columns:
                    proc_col = cand
                    break

            if sig_code_col and proc_col:
                df_enriched[proc_col] = df_enriched[proc_col].astype(str).str.strip()
                sig_df[sig_code_col] = sig_df[sig_code_col].astype(str).str.strip()

                keep_cols = [sig_code_col]
                keep_cols += [c for c in sig_df.columns if any(k in c for k in ["grupo", "subgrupo", "forma", "nome"])]
                sig_small = sig_df[keep_cols].drop_duplicates(subset=[sig_code_col])
                df_enriched = df_enriched.merge(sig_small, how="left", left_on=proc_col, right_on=sig_code_col)
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com SIGTAP: {e}")

    # ---------------- Regiões de Saúde ----------------
    if geo_df is not None:
        try:
            geo_df = geo_df.copy()
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

                geo_small = geo_df[["no_municipio_norm", "sg_uf", "no_macrorregional", "no_cir_padrao"]].drop_duplicates(
                    subset=["no_municipio_norm", "sg_uf"]
                )

                df_enriched = df_enriched.merge(
                    geo_small,
                    how="left",
                    left_on=["cidade_nome_norm", "uf_from_cidade"],
                    right_on=["no_municipio_norm", "sg_uf"],
                )

                df_enriched = df_enriched.rename(
                    columns={"sg_uf": "uf", "no_macrorregional": "macroregiao", "no_cir_padrao": "regiao_saude"}
                )

                df_enriched = df_enriched.drop(columns=["cidade_nome_norm", "no_municipio_norm"], errors="ignore")
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com regiões de saúde: {e}")

    return df_enriched


# --------------------------------------------------------------------
# FUNÇÕES DE MÉTRICAS / KPI
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
    """Mortalidade hospitalar: data_obito entre data_internacao e data_alta (mesmo dia conta)."""
    if {"data_internacao", "data_alta"}.issubset(df.columns):
        e = df.copy()
        if "data_obito" in e.columns:
            data_int = pd.to_datetime(e["data_internacao"], errors="coerce").dt.floor("D")
            data_alta = pd.to_datetime(e["data_alta"], errors="coerce").dt.floor("D")
            data_obito = pd.to_datetime(e["data_obito"], errors="coerce").dt.floor("D")

            e["obito_no_periodo"] = (
                data_obito.notna()
                & (data_obito >= data_int)
                & (data_alta.isna() | (data_obito <= data_alta))
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
    return df


def marcar_uti_flag(df: pd.DataFrame) -> pd.DataFrame:
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
            uti_flag = uti_flag | e[cand].astype(str).str.upper().isin(["1", "S", "SIM", "TRUE", "VERDADEIRO"])

    e["uti_flag"] = uti_flag
    aux = e[["codigo_internacao", "uti_flag"]].groupby("codigo_internacao", as_index=False)["uti_flag"].max()
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["uti_flag"] = df["uti_flag"].fillna(False)
    return df


def marcar_reinternacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Marca reinternações em até 30 dias."""
    df = df.copy()
    required = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}
    if not required.issubset(df.columns):
        df["reint_30d_proc"] = False
        df["reint_30d_alta"] = False
        return df

    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"], kind="mergesort").copy()
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

    df = df.drop(columns=["reint_30d_proc", "reint_30d_alta"], errors="ignore")
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["reint_30d_proc"] = df["reint_30d_proc"].fillna(False)
    df["reint_30d_alta"] = df["reint_30d_alta"].fillna(False)
    return df


def marcar_mort_30d_proc(df: pd.DataFrame) -> pd.DataFrame:
    if "data_obito" not in df.columns or "codigo_internacao" not in df.columns:
        df["obito_30d_proc"] = False
        return df

    e = df.copy()
    cand_datas = ["data_procedimento", "data_cirurgia", "data_cirurgia_min", "data_internacao"]
    data_proc_col = next((c for c in cand_datas if c in e.columns), None)
    if not data_proc_col:
        df["obito_30d_proc"] = False
        return df

    e = e.sort_values(data_proc_col).groupby("codigo_internacao", as_index=False).first()
    e["delta"] = (e["data_obito"] - e[data_proc_col]).dt.days
    e["obito_30d_proc"] = e["delta"].between(0, 30, inclusive="both")

    aux = e[["codigo_internacao", "obito_30d_proc"]]
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["obito_30d_proc"] = df["obito_30d_proc"].fillna(False)
    return df


def marcar_mort_30d_alta(df: pd.DataFrame) -> pd.DataFrame:
    if not {"data_alta", "data_obito", "codigo_internacao"}.issubset(df.columns):
        df["obito_30d_alta"] = False
        return df

    e = df.sort_values("data_alta").groupby("codigo_internacao", as_index=False).tail(1)
    e["delta"] = (e["data_obito"] - e["data_alta"]).dt.days
    e["obito_30d_alta"] = e["delta"].between(0, 30, inclusive="both")

    aux = e[["codigo_internacao", "obito_30d_alta"]]
    df = df.merge(aux, on="codigo_internacao", how="left")
    df["obito_30d_alta"] = df["obito_30d_alta"].fillna(False)
    return df


def kpis(df_eventos: pd.DataFrame, df_pacientes: pd.DataFrame):
    pacientes = df_pacientes["prontuario_anonimo"].nunique() if "prontuario_anonimo" in df_pacientes else np.nan
    internacoes = df_eventos["codigo_internacao"].nunique() if "codigo_internacao" in df_eventos else len(df_eventos)

    tmi = (
        df_eventos["dias_permanencia"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        if "dias_permanencia" in df_eventos
        else np.nan
    )

    mort_hosp = np.nan
    if {"data_internacao", "data_alta"}.issubset(df_eventos.columns):
        e = df_eventos.copy()
        if "data_obito" in e.columns:
            data_int = pd.to_datetime(e["data_internacao"], errors="coerce").dt.floor("D")
            data_alta = pd.to_datetime(e["data_alta"], errors="coerce").dt.floor("D")
            data_obito = pd.to_datetime(e["data_obito"], errors="coerce").dt.floor("D")
            e["obito_no_periodo"] = (
                data_obito.notna()
                & (data_obito >= data_int)
                & (data_alta.isna() | (data_obito <= data_alta))
            )
        else:
            e["obito_no_periodo"] = False

        denom = e["codigo_internacao"].nunique() if "codigo_internacao" in e else len(e)
        numer_df = e.loc[e["obito_no_periodo"]]
        numer = numer_df["codigo_internacao"].nunique() if "codigo_internacao" in e else len(numer_df)
        mort_hosp = (numer / denom * 100) if denom else np.nan

    return pacientes, internacoes, tmi, mort_hosp


def reinternacao_30d_pos_proced(df: pd.DataFrame):
    ok = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}.issubset(df.columns)
    if not ok:
        return np.nan
    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta_proc"] = (s["next_dt_internacao"] - s["data_internacao"]).dt.days
    s["delta_pos_alta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta_pos_alta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[s["delta_proc"].between(0, 30, inclusive="both") & (~s["transfer"])]["codigo_internacao"].nunique()
    return (numer / base * 100) if base else np.nan


def reinternacao_30d_pos_alta(df: pd.DataFrame):
    ok = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}.issubset(df.columns)
    if not ok:
        return np.nan
    s = df.sort_values(["prontuario_anonimo", "data_internacao", "data_alta"]).copy()
    s["next_dt_internacao"] = s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    s["delta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[s["delta"].between(0, 30, inclusive="both") & (~s["transfer"])]["codigo_internacao"].nunique()
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
            uti_flag = uti_flag | e[cand].astype(str).str.upper().isin(["1", "S", "SIM", "TRUE", "VERDADEIRO"])

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
    cand_datas = ["data_procedimento", "data_cirurgia", "data_cirurgia_min", "data_cirurgia_max", "data_internacao"]
    data_proc_col = next((c for c in cand_datas if c in e.columns), None)
    if not data_proc_col:
        return np.nan

    e = e.sort_values(data_proc_col).groupby("codigo_internacao", as_index=False).first()
    e["delta"] = (e["data_obito"] - e[data_proc_col]).dt.days
    e["obito_30d_proc"] = e["delta"].between(0, 30, inclusive="both")

    denom = e["codigo_internacao"].nunique()
    numer = e.loc[e["obito_30d_proc"], "codigo_internacao"].nunique()
    return (numer / denom * 100) if denom else np.nan


def mortalidade_30d_pos_alta(df: pd.DataFrame):
    if not {"data_alta", "data_obito", "codigo_internacao"}.issubset(df.columns):
        return np.nan

    e = df.sort_values("data_alta").groupby("codigo_internacao", as_index=False).tail(1)
    e["delta"] = (e["data_obito"] - e["data_alta"]).dt.days
    e["obito_30d_alta"] = e["delta"].between(0, 30, inclusive="both")

    denom = e["codigo_internacao"].nunique()
    numer = e.loc[e["obito_30d_alta"], "codigo_internacao"].nunique()
    return (numer / denom * 100) if denom else np.nan


# --------------------------------------------------------------------
# AUXILIARES PARA INDICADOR
# --------------------------------------------------------------------


def definir_base_para_indicador(indicador, df_f, df_pac):
    """Define a base de linhas que alimenta os gráficos conforme o indicador."""
    if indicador == "Quantidade de pacientes":
        return df_pac.copy()

    if indicador == "Quantidade de internações":
        if df_f is not None and "codigo_internacao" in df_f.columns:
            return df_f.drop_duplicates(subset=["codigo_internacao"]).copy()
        return df_f.copy()

    return df_f.copy()


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
    elif indicador == "Internação em UTI (%)":
        df = marcar_uti_flag(df)
        df["peso"] = df["uti_flag"].astype(int)
    elif indicador == "Tempo médio de internação em UTI (dias)":
        if {"dt_entrada_cti", "dt_saida_cti"}.issubset(df.columns):
            dias_uti = (df["dt_saida_cti"] - df["dt_entrada_cti"]).dt.days
            df["peso"] = dias_uti.clip(lower=0).fillna(0)
    elif indicador == "Reinternação em até 30 dias do procedimento (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = df["reint_30d_proc"].fillna(False).astype(int)
    elif indicador == "Reinternação em até 30 dias da alta (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = df["reint_30d_alta"].fillna(False).astype(int)
    elif indicador == "Mortalidade hospitalar (%)":
        df = marcar_obito_periodo(df)
        df["peso"] = df["obito_no_periodo"].fillna(False).astype(int)
    elif indicador == "Mortalidade em até 30 dias do procedimento (%)":
        df = marcar_mort_30d_proc(df)
        df["peso"] = df["obito_30d_proc"].fillna(False).astype(int)
    elif indicador == "Mortalidade em até 30 dias da alta (%)":
        df = marcar_mort_30d_alta(df)
        df["peso"] = df["obito_30d_alta"].fillna(False).astype(int)

    return df


# --------------------------------------------------------------------
# FILTROS (com CASCATA Estado -> Região -> Município + sanitização)
# --------------------------------------------------------------------


def build_filters(df: pd.DataFrame):
    if df is None:
        st.error("Dataset não carregado.")
        st.stop()

    st.sidebar.header("Filtros")

    def _multiselect_com_todos(
        titulo: str,
        opcoes: list,
        key: str,
        default=None,
        help_text: str | None = None,
    ):
        """Multiselect com a opção 'Selecionar todos' dentro da lista + sanitização.

        Sanitização: se as opções mudarem (cascata), removemos seleções antigas que não existem mais.
        """
        all_token = "Selecionar todos"
        prev_key = f"__prev_{key}"

        opcoes = [str(x) for x in opcoes if pd.notna(x)]
        opcoes_set = set(opcoes)

        if default is None:
            default = opcoes
        default = [str(x) for x in default if str(x) in opcoes_set]

        # inicializa
        if key not in st.session_state:
            if len(opcoes) > 0 and set(default) == set(opcoes):
                st.session_state[key] = [all_token] + opcoes
            else:
                st.session_state[key] = default

        # sanitiza estado atual caso tenha sobras antigas
        curr0 = [str(x) for x in st.session_state.get(key, [])]
        curr0 = [x for x in curr0 if x == all_token or x in opcoes_set]
        if all_token in curr0 and set([x for x in curr0 if x != all_token]) != set(opcoes):
            # token ficou inválido porque não é mais "tudo"
            curr0 = [x for x in curr0 if x != all_token]
        if curr0 != st.session_state.get(key, []):
            st.session_state[key] = curr0

        if prev_key not in st.session_state:
            st.session_state[prev_key] = st.session_state.get(key, []).copy()

        def _on_change():
            prev = st.session_state.get(prev_key, [])
            curr = [str(x) for x in st.session_state.get(key, [])]

            # sanitiza no on_change também
            curr = [x for x in curr if x == all_token or x in opcoes_set]

            # 1) Token foi marcado agora -> selecionar tudo
            if all_token in curr and all_token not in prev:
                st.session_state[key] = [all_token] + opcoes

            # 2) Token já estava marcado e algo foi desmarcado -> remover token e manter parcial
            elif all_token in prev and all_token in curr and set(curr) != set([all_token] + opcoes):
                st.session_state[key] = [x for x in curr if x != all_token]

            else:
                curr_wo = [x for x in curr if x != all_token]
                # se todos marcados manualmente -> adiciona token
                if len(opcoes) > 0 and set(curr_wo) == set(opcoes):
                    st.session_state[key] = [all_token] + opcoes
                else:
                    st.session_state[key] = curr_wo

            st.session_state[prev_key] = st.session_state.get(key, []).copy()

        with st.sidebar.container(border=True):
            st.markdown(f"**{titulo}**")
            st.multiselect(
                titulo,
                options=[all_token] + opcoes,
                default=st.session_state.get(key, []),
                key=key,
                on_change=_on_change,
                label_visibility="collapsed",
                help=help_text,
            )

        st.session_state[prev_key] = st.session_state.get(key, []).copy()
        sel_final = st.session_state.get(key, [])
        return [x for x in sel_final if x != all_token]

    # ---- Período da internação (calendário) ----
    periodo_sel = None
    if "data_internacao" in df.columns:
        min_dt = pd.to_datetime(df["data_internacao"]).min().date()
        max_dt = pd.to_datetime(df["data_internacao"]).max().date()
        periodo_sel = st.sidebar.date_input(
            "Período da internação",
            value=(min_dt, max_dt),
            format="DD/MM/YYYY",
        )
        if not isinstance(periodo_sel, (list, tuple)):
            periodo_sel = (periodo_sel, periodo_sel)
    else:
        st.sidebar.info("Coluna 'data_internacao' não encontrada para filtro de período.")

    # ---- Idade ----
    if "idade" in df.columns and df["idade"].notna().any():
        idade_min, idade_max = int(np.nanmin(df["idade"])), int(np.nanmax(df["idade"]))
    else:
        idade_min, idade_max = 0, 120
    idade_sel = st.sidebar.slider(
        "Idade",
        min_value=0,
        max_value=max(idade_max, 1),
        value=(idade_min, idade_max),
        step=1,
    )

    # ---- Estado ----
    estado_col = next(
        (c for c in df.columns if c.lower() in ["estado_residencia", "uf_residencia", "uf", "estado", "sigla_uf"]),
        None,
    )
    estados_sel = []
    df_estado = df
    if estado_col:
        estados = sorted(df[estado_col].dropna().astype(str).unique().tolist())
        estados_sel = _multiselect_com_todos(
            "Estado de residência",
            estados,
            key="ms_estados",
            default=estados,
        )
        if estados_sel:
            df_estado = df[df[estado_col].astype(str).isin(estados_sel)]

    # ---- Região de saúde (dependente do Estado) ----
    regiao_col = next((c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()), None)
    regioes_sel = []
    if regiao_col and not df_estado.empty:
        regioes = sorted(df_estado[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = _multiselect_com_todos(
            "Região de saúde",
            regioes,
            key="ms_regioes",
            default=regioes,
        )

    # ---- Município (dependente do Estado e Região) ----
    cidade_col = "cidade_moradia" if "cidade_moradia" in df.columns else None
    cidades_sel = []
    df_cidade_base = df_estado
    if regiao_col and regioes_sel:
        df_cidade_base = df_cidade_base[df_cidade_base[regiao_col].astype(str).isin(regioes_sel)]

    if cidade_col and not df_cidade_base.empty:
        cidade_vals = sorted(df_cidade_base[cidade_col].dropna().astype(str).unique().tolist())
        default_cidades = cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        cidades_sel = _multiselect_com_todos(
            "Município de residência",
            cidade_vals,
            key="ms_cidades",
            default=default_cidades,
        )

    # ---- Sexo ----
    sexo_sel = []
    if "sexo" in df.columns:
        sexos = sorted(df["sexo"].dropna().astype(str).unique().tolist())
        sexo_sel = _multiselect_com_todos(
            "Sexo",
            sexos,
            key="ms_sexos",
            default=sexos,
        )

    return {
        "periodo": periodo_sel,
        "idade": idade_sel,
        "estado": estados_sel,
        "regiao": regioes_sel,
        "cidade": cidades_sel,
        "sexo": sexo_sel,
    }


def apply_filters(df: pd.DataFrame, f, include_period: bool = True):
    # Período
    if include_period and "data_internacao" in df.columns and f.get("periodo"):
        ini, fim = f["periodo"]
        ini = pd.to_datetime(ini)
        fim = pd.to_datetime(fim) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df["data_internacao"] >= ini) & (df["data_internacao"] <= fim)]

    # Idade
    if "idade" in df.columns and f["idade"]:
        df = df[(df["idade"] >= f["idade"][0]) & (df["idade"] <= f["idade"][1])]

    # Estado
    estado_col = next(
        (c for c in df.columns if c.lower() in ["estado_residencia", "uf_residencia", "uf", "estado", "sigla_uf"]),
        None,
    )
    if estado_col and f["estado"]:
        df = df[df[estado_col].astype(str).isin(f["estado"])]

    # Região de saúde
    regiao_col = next((c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()), None)
    if regiao_col and f["regiao"]:
        df = df[df[regiao_col].astype(str).isin(f["regiao"])]

    # Município
    if "cidade_moradia" in df.columns and f["cidade"]:
        df = df[df["cidade_moradia"].astype(str).isin(f["cidade"])]

    # Sexo
    if "sexo" in df.columns and f["sexo"]:
        df = df[df["sexo"].astype(str).isin(f["sexo"])]

    return df


# (Função mantida, mas não é chamada — “Filtros ativos” removido do layout)
def show_active_filters(f):
    partes = []
    if f.get("periodo"):
        ini, fim = f["periodo"]
        partes.append(f"**Período:** {ini.strftime('%d/%m/%Y')} – {fim.strftime('%d/%m/%Y')}")
    if f["idade"]:
        partes.append(f"**Idade:** {f['idade'][0]}–{f['idade'][1]} anos")
    if f["estado"]:
        partes.append("**Estado:** " + ", ".join(f["estado"]))
    if f["regiao"]:
        partes.append("**Região de saúde:** " + ", ".join(f["regiao"]))
    if f["cidade"]:
        partes.append("**Município:** " + ", ".join(f["cidade"]))
    if f["sexo"]:
        partes.append("**Sexo:** " + ", ".join(f["sexo"]))
    if partes:
        st.markdown("**Filtros ativos:** " + " | ".join(partes))
    else:
        st.markdown("**Filtros ativos:** nenhum filtro aplicado.")

# --------------------------------------------------------------------
# INTERFACE PRINCIPAL
# --------------------------------------------------------------------

st.title("Perfil dos Pacientes")

# ------------ Upload dos 3 CSVs (apenas na primeira vez) ------------
if "df" not in st.session_state:
    df_tmp = None

    st.subheader("Envie os 3 arquivos CSV")

    c1, c2, c3 = st.columns(3)
    evo = c1.file_uploader("CARACTERIZAÇÃO (csv)", type=["csv"], key="evo")
    proc = c2.file_uploader("PROCEDIMENTOS (csv)", type=["csv"], key="proc")
    cti = c3.file_uploader("CIDs/UTI (csv)", type=["csv"], key="cti")

    if evo and proc and cti:
        tmpdir = tempfile.mkdtemp()
        p_evo = os.path.join(tmpdir, "evo.csv")
        p_proc = os.path.join(tmpdir, "proc.csv")
        p_cti = os.path.join(tmpdir, "cti.csv")
        open(p_evo, "wb").write(evo.getbuffer())
        open(p_proc, "wb").write(proc.getbuffer())
        open(p_cti, "wb").write(cti.getbuffer())

        con = load_duckdb((p_evo, p_proc, p_cti))
        df_tmp = df_from_duckdb(con, "SELECT * FROM dataset")
        df_tmp = _post_load(df_tmp)

        cid_df, sigtap_df, geo_df = load_aux_tables()
        df_tmp = enrich_with_aux_tables(df_tmp, cid_df, sigtap_df, geo_df)

        # Base de pacientes (caracterização) para contagem alternativa
        df_base_tmp = df_from_duckdb(con, "SELECT * FROM pacientes_base")
        df_base_tmp = _post_load(df_base_tmp)
        df_base_tmp = enrich_with_aux_tables(df_base_tmp, cid_df, sigtap_df, geo_df)

        st.session_state["df"] = df_tmp
        st.session_state["df_base"] = df_base_tmp
        st.success("Arquivos carregados com sucesso! Painel inicializado.")

        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.stop()

# ------------ Depois de carregado, não mostra mais os uploaders ------------
df = st.session_state["df"]
df_base = st.session_state.get("df_base")

if df is None or df.empty:
    st.error("Dataset vazio ou não carregado corretamente.")
    st.stop()

# Filtros
f = build_filters(df)
df_f = apply_filters(df, f, include_period=True)
# Para a base de pacientes, aplicamos apenas filtros demográficos (sem período)
df_base_f = apply_filters(df_base, f, include_period=False) if df_base is not None else None
df_pac = pacientes_unicos(df_f)

pacientes_base_count = (
    df_base_f["prontuario_anonimo"].nunique()
    if (df_base_f is not None and "prontuario_anonimo" in df_base_f.columns)
    else np.nan
)

# REMOVIDO: show_active_filters(f)
st.divider()

modo_perfil = True

# KPIs globais
pacientes, internacoes, tmi, mort_hosp = kpis(df_f, df_pac)
ri_proc = reinternacao_30d_pos_proced(df_f)
ri_alta = reinternacao_30d_pos_alta(df_f)
uti_pct = internacao_uti_pct(df_f)
tmi_uti = tempo_medio_uti_dias(df_f)
mort_30_proc = mortalidade_30d_pos_proced(df_f)
mort_30_alta = mortalidade_30d_pos_alta(df_f)

# --------------------------------------------------------------------
# Estilo dos botões dos indicadores (chips)
# --------------------------------------------------------------------
st.markdown(
    """
<style>
div[data-baseweb="radio"] > div {
    flex-wrap: wrap;
    gap: 0.35rem;
}
div[data-baseweb="radio"] label > div:first-child {
    display: none;
}
div[data-baseweb="radio"] label > div:nth-child(2) {
    border-radius: 999px;
    border: 1px solid #d0d0d0;
    background-color: #ffffff;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: #333333;
    transition: 0.15s;
}
div[data-baseweb="radio"] label:hover > div:nth-child(2) {
    border-color: #ff4b4b;
    color: #ff4b4b;
}
div[data-baseweb="radio"] input:checked + div {
    background-color: #ff4b4b !important;
    border-color: #ff4b4b !important;
    color: #ffffff !important;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

indicadores_icardio = [
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

indicadores_quantidade = [
    "Quantidade de pacientes",
    "Quantidade de internações",
    "Quantidade de procedimentos",
]

indicadores_media = [
    "Tempo médio de internação (dias)",
    "Tempo médio de internação em UTI (dias)",
]

indicadores_percentual = [
    "Internação em UTI (%)",
    "Reinternação em até 30 dias do procedimento (%)",
    "Reinternação em até 30 dias da alta (%)",
    "Mortalidade hospitalar (%)",
    "Mortalidade em até 30 dias do procedimento (%)",
    "Mortalidade em até 30 dias da alta (%)",
]


def agrega_para_grafico(df, group_cols, indicador):
    """
    Usa a coluna 'peso':
    - Quantidade  -> soma
    - Tempo médio -> média
    - Percentual  -> média*100
    """
    g = df.groupby(group_cols, dropna=False)["peso"]

    if indicador in indicadores_percentual:
        out = g.mean().mul(100.0).reset_index(name="valor")
    elif indicador in indicadores_media:
        out = g.mean().reset_index(name="valor")
    else:
        out = g.sum().reset_index(name="valor")
    return out


def label_eixo_x(indicador):
    if indicador in indicadores_percentual:
        return "Taxa (%)"
    if indicador in indicadores_media:
        return "Média (dias)"
    return "Quantidade"


# ---------- mapa de cores fixas por sexo ----------
def get_sexo_color_map(categories):
    """
    Retorna um dict categoria -> cor, mantendo:
    - Masculino: azul
    - Feminino: rosa
    - Outros: cinza
    """
    mapa = {}
    for s in categories:
        chave = str(s)
        norm = chave.strip().upper()
        if norm in ["M", "MASCULINO"]:
            mapa[chave] = "#6794DC"  # azul
        elif norm in ["F", "FEMININO"]:
            mapa[chave] = "#E86F86"  # rosa
        else:
            mapa[chave] = "#A3A3A3"  # neutro
    return mapa


# --------------------------------------------------------------------
# Cards / Formatação
# --------------------------------------------------------------------
def format_val_for_card(indicador: str, v: float) -> str:
    """
    Formatação nos cards (barras internas):
    - Percentuais: 2 casas
    - Médias: 1 casa
    - Quantidades: inteiro (sem .00). Se >=1000, mostra em "Mil" inteiro.
    """
    if pd.isna(v):
        return "—"

    if indicador in indicadores_percentual:
        return f"{v:.2f}%".replace(".", ",")

    if indicador in indicadores_media:
        return f"{v:.1f}".replace(".", ",")

    # Quantidades inteiras
    if indicador in indicadores_quantidade:
        if v >= 1000:
            return f"{int(v/1000)} Mil"
        return f"{int(v)}"

    # fallback
    return f"{v:,.0f}".replace(",", ".")


def card_bar_fig(
    df_cat: pd.DataFrame,
    cat_col: str,
    indicador: str,
    colors=None,
    color_map=None,
    height: int = 90,
):
    """Retorna um gráfico Plotly em forma de card (barra única segmentada)."""
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
# Indicadores: valor + numerador (apenas para percentuais)
# --------------------------------------------------------------------
st.markdown("### Indicadores disponíveis")

indicador_selecionado = st.radio(
    "Selecione o indicador para detalhar e para o comparativo temporal:",
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
        return uti_pct
    if nome == "Tempo médio de internação em UTI (dias)":
        return tmi_uti
    if nome == "Reinternação em até 30 dias do procedimento (%)":
        return ri_proc
    if nome == "Reinternação em até 30 dias da alta (%)":
        return ri_alta
    if nome == "Mortalidade hospitalar (%)":
        return mort_hosp
    if nome == "Mortalidade em até 30 dias do procedimento (%)":
        return mort_30_proc
    if nome == "Mortalidade em até 30 dias da alta (%)":
        return mort_30_alta
    return np.nan


def calcular_indicador_ano(nome, df_eventos_ano: pd.DataFrame, df_pacientes_ano: pd.DataFrame):
    pac_ano, int_ano, tmi_ano, mort_hosp_ano = kpis(df_eventos_ano, df_pacientes_ano)
    ri_proc_ano = reinternacao_30d_pos_proced(df_eventos_ano)
    ri_alta_ano = reinternacao_30d_pos_alta(df_eventos_ano)
    uti_pct_ano = internacao_uti_pct(df_eventos_ano)
    tmi_uti_ano = tempo_medio_uti_dias(df_eventos_ano)
    mort_30_proc_ano = mortalidade_30d_pos_proced(df_eventos_ano)
    mort_30_alta_ano = mortalidade_30d_pos_alta(df_eventos_ano)
    qtd_proc_ano = df_eventos_ano["n_proced"].sum() if "n_proced" in df_eventos_ano.columns else np.nan

    if nome == "Quantidade de pacientes":
        return pac_ano
    if nome == "Quantidade de internações":
        return int_ano
    if nome == "Quantidade de procedimentos":
        return qtd_proc_ano
    if nome == "Tempo médio de internação (dias)":
        return tmi_ano
    if nome == "Internação em UTI (%)":
        return uti_pct_ano
    if nome == "Tempo médio de internação em UTI (dias)":
        return tmi_uti_ano
    if nome == "Reinternação em até 30 dias do procedimento (%)":
        return ri_proc_ano
    if nome == "Reinternação em até 30 dias da alta (%)":
        return ri_alta_ano
    if nome == "Mortalidade hospitalar (%)":
        return mort_hosp_ano
    if nome == "Mortalidade em até 30 dias do procedimento (%)":
        return mort_30_proc_ano
    if nome == "Mortalidade em até 30 dias da alta (%)":
        return mort_30_alta_ano
    return np.nan


valor_ind = calcular_indicador(indicador_selecionado)

# base para numerador (apenas nos percentuais): usa a mesma base do indicador
# e calcula peso (0/1) para somar numerador
def numerador_indicador_percentual(indicador_nome: str, df_eventos: pd.DataFrame, df_pac: pd.DataFrame) -> int | None:
    if indicador_nome not in indicadores_percentual:
        return None

    base = definir_base_para_indicador(indicador_nome, df_eventos, df_pac)
    base = adicionar_peso_por_indicador(base, indicador_nome)
    # numerador = soma dos "eventos positivos"
    num = int(base["peso"].fillna(0).sum()) if "peso" in base.columns else 0
    return num


numerador_pct = numerador_indicador_percentual(indicador_selecionado, df_f, df_pac)

# Formata valor principal do card da direita:
if pd.isna(valor_ind):
    texto_valor = "—"
elif indicador_selecionado in indicadores_quantidade:
    texto_valor = f"{int(valor_ind):,}".replace(",", ".")
elif indicador_selecionado in indicadores_percentual:
    # % + (numerador)
    pct_txt = f"{valor_ind:.2f}%".replace(".", ",")
    if numerador_pct is not None:
        texto_valor = f"{pct_txt} ({numerador_pct:,})".replace(",", ".")
    else:
        texto_valor = pct_txt
else:
    # médias
    texto_valor = f"{valor_ind:.2f}".replace(".", ",")


# --------------------------------------------------------------------
# BASE PARA GRÁFICOS E GRID PRINCIPAL
# --------------------------------------------------------------------

st.divider()

if indicador_selecionado == "Quantidade de pacientes" and modo_perfil:
    base_charts = df_pac.copy()
else:
    base_charts = definir_base_para_indicador(indicador_selecionado, df_f, df_pac)

base_charts = adicionar_peso_por_indicador(base_charts, indicador_selecionado)

col_esq, col_meio, col_dir = st.columns([1.1, 1.3, 1.1])
# --------------------------------------------------------------------
# COLUNA 1 — Sexo / Raça-Cor x Sexo / Pirâmide etária (simplificado)
# --------------------------------------------------------------------
with col_esq:
    st.markdown("## Sexo")

    if "sexo" in base_charts.columns:
        df_sexo = agrega_para_grafico(base_charts, ["sexo"], indicador_selecionado)
        sexo_categories = df_sexo["sexo"].astype(str).tolist()
        sexo_color_map = get_sexo_color_map(sexo_categories)

        fig_sexo = card_bar_fig(
            df_sexo,
            cat_col="sexo",
            indicador=indicador_selecionado,
            color_map=sexo_color_map,
            height=95,
        )
        st.plotly_chart(fig_sexo, use_container_width=True)
    else:
        st.info("Coluna 'sexo' não encontrada.")

# --------------------------------------------------------------------
# COLUNA 2 — Caráter do atendimento (ELET/URG/EMERG)
# --------------------------------------------------------------------
with col_meio:
    st.markdown("## Caráter do Atendimento")

    # tentativa de identificar coluna do caráter
    caracter_col = None
    for cand in ["carater_atendimento", "carater", "natureza_agend", "carater_do_atendimento"]:
        if cand in base_charts.columns:
            caracter_col = cand
            break

    if caracter_col:
        df_car = agrega_para_grafico(base_charts, [caracter_col], indicador_selecionado)
        df_car[caracter_col] = df_car[caracter_col].astype(str).str.upper()

        # Cores solicitadas:
        # Eletiva verde, Urgência verde e Emergência amarelo
        car_color_map = {
            "ELE": "#4CAF50",
            "ELETIVA": "#4CAF50",
            "URG": "#2E7D32",
            "URGÊNCIA": "#2E7D32",
            "URGENCIA": "#2E7D32",
            "EMG": "#FBC02D",
            "EMERGÊNCIA": "#FBC02D",
            "EMERGENCIA": "#FBC02D",
        }
        # fallback para qualquer outro valor
        for v in df_car[caracter_col].unique().tolist():
            if v not in car_color_map:
                car_color_map[v] = "#9E9E9E"

        fig_car = card_bar_fig(
            df_car,
            cat_col=caracter_col,
            indicador=indicador_selecionado,
            color_map=car_color_map,
            height=95,
        )
        st.plotly_chart(fig_car, use_container_width=True)
    else:
        st.info("Coluna de caráter do atendimento não encontrada.")

# --------------------------------------------------------------------
# COLUNA 3 — Card principal do indicador (valor)
# --------------------------------------------------------------------
with col_dir:
    st.markdown(f"## {indicador_selecionado}")
    st.markdown(
        f"""
        <div style="
            width:100%;
            background:#ffffff;
            border:1px solid #e6e6e6;
            border-radius:14px;
            padding:28px 18px;
            text-align:center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 44px; font-weight: 700; line-height: 1.1;">
                {texto_valor}
            </div>
            <div style="font-size: 13px; color: #666; margin-top: 12px;">
                Valor do indicador no período filtrado
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------------------
# Segunda linha: Procedimentos, CID, Geo (Treemap) e Boxplot Idade x Sexo
# --------------------------------------------------------------------
cA, cB, cC = st.columns([1.25, 1.25, 1.0])

# ----------------- Procedimentos (amostra) -----------------
with cA:
    st.markdown("### Procedimentos (amostra)")

    proc_nome_col = None
    for cand in ["proc_nome_prim", "procedimento", "procedimento_desc", "ds_procedimento", "proced_nome"]:
        if cand in base_charts.columns:
            proc_nome_col = cand
            break

    if proc_nome_col:
        df_proc = agrega_para_grafico(base_charts, [proc_nome_col], indicador_selecionado)
        df_proc = df_proc.sort_values("valor", ascending=False).head(15)
        fig_proc = px.bar(
            df_proc,
            x="valor",
            y=proc_nome_col,
            orientation="h",
            labels={"valor": label_eixo_x(indicador_selecionado), proc_nome_col: "Procedimento"},
        )
        fig_proc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_proc, use_container_width=True)
    else:
        st.info("Coluna de procedimentos não encontrada.")

# ----------------- CID (capítulo/grupo) — amostra -----------------
with cB:
    st.markdown("### CID (capítulo / grupo) – amostra")

    cid_col = "cid_capitulo" if "cid_capitulo" in base_charts.columns else None
    if cid_col is None and "cid_grupo" in base_charts.columns:
        cid_col = "cid_grupo"

    if cid_col:
        df_cid = agrega_para_grafico(base_charts, [cid_col], indicador_selecionado)
        df_cid = df_cid.sort_values("valor", ascending=False).head(15)
        fig_cid = px.bar(
            df_cid,
            x="valor",
            y=cid_col,
            orientation="h",
            labels={"valor": label_eixo_x(indicador_selecionado), cid_col: "CID"},
        )
        fig_cid.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cid, use_container_width=True)
    else:
        st.info("Coluna de CID (capítulo/grupo) não encontrada.")

# ----------------- Geo treemap (Estado -> Região -> Município) -----------------
with cC:
    st.markdown("### Estado → Região de Saúde → Município")

    df_geo_plot = base_charts.copy()

    # normaliza e garante níveis completos (evita ValueError do treemap)
    def _norm_str(s):
        return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # tenta detectar colunas
    uf_col = None
    for cand in ["uf", "uf_residencia", "estado_residencia", "sigla_uf", "estado"]:
        if cand in df_geo_plot.columns:
            uf_col = cand
            break

    reg_col = None
    for cand in ["regiao_saude", "regiao de saude", "regiao_saude_res", "no_cir_padrao"]:
        if cand in df_geo_plot.columns:
            reg_col = cand
            break
    if reg_col is None:
        # fallback: procura algo com regiao+saude
        reg_col = next((c for c in df_geo_plot.columns if "regiao" in c.lower() and "saud" in c.lower()), None)

    mun_col = "cidade_moradia" if "cidade_moradia" in df_geo_plot.columns else None

    if uf_col and reg_col and mun_col:
        df_geo_plot["uf"] = _norm_str(df_geo_plot[uf_col]).str.upper().fillna("Sem UF")
        df_geo_plot["regiao_saude"] = _norm_str(df_geo_plot[reg_col]).fillna("Sem região")
        df_geo_plot["cidade_moradia"] = _norm_str(df_geo_plot[mun_col]).fillna("Sem município")

        # Prefixos para evitar colisão de nomes entre níveis
        df_geo_plot["uf_p"] = "UF: " + df_geo_plot["uf"]
        df_geo_plot["rs_p"] = "RS: " + df_geo_plot["regiao_saude"]
        df_geo_plot["mun_p"] = "Mun: " + df_geo_plot["cidade_moradia"]

        # valor para treemap:
        if indicador_selecionado in indicadores_percentual:
            # treemap em percentuais não faz muito sentido em "soma".
            # Mantém numerador (eventos) como valor.
            df_geo_plot["valor_plot"] = df_geo_plot["peso"].fillna(0)
        elif indicador_selecionado in indicadores_media:
            # usa soma do peso, para representar volume do total de dias
            df_geo_plot["valor_plot"] = df_geo_plot["peso"].fillna(0)
        else:
            df_geo_plot["valor_plot"] = df_geo_plot["peso"].fillna(0)

        df_geo_plot = (
            df_geo_plot.groupby(["uf_p", "rs_p", "mun_p"], dropna=False)["valor_plot"]
            .sum()
            .reset_index()
        )

        fig = px.treemap(
            df_geo_plot,
            path=["uf_p", "rs_p", "mun_p"],
            values="valor_plot",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colunas geo (UF/Região/Município) não encontradas para o treemap.")

st.divider()

# --------------------------------------------------------------------
# Boxplot — Idade por sexo (coluna à direita originalmente)
# --------------------------------------------------------------------
st.markdown("### Boxplot – Idade por sexo")

if "idade" in base_charts.columns and "sexo" in base_charts.columns:
    df_box = base_charts.copy()
    df_box = df_box[df_box["idade"].notna()]
    fig_box = px.box(
        df_box,
        x="sexo",
        y="idade",
        points="outliers",
        labels={"sexo": "Sexo", "idade": "Idade"},
    )
    fig_box.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("Colunas necessárias ('idade' e 'sexo') não encontradas para o boxplot.")
# --------------------------------------------------------------------
# COMPARATIVO TEMPORAL (Ano / Meses)
# --------------------------------------------------------------------
st.divider()
st.markdown("## Comparativo temporal")

tipo_comp = st.selectbox("Comparar por:", ["Ano", "Mês (dentro do ano)"], index=0)

df_time = df_f.copy()
df_time = df_time.dropna(subset=["data_internacao"]) if "data_internacao" in df_time.columns else df_time

if "data_internacao" not in df_time.columns or df_time.empty:
    st.info("Sem dados suficientes para comparativo temporal.")
else:
    df_time["ano"] = pd.to_datetime(df_time["data_internacao"], errors="coerce").dt.year
    df_time["mes"] = pd.to_datetime(df_time["data_internacao"], errors="coerce").dt.month

    # base pacientes por ano (para indicador "Quantidade de pacientes")
    df_pac_time = df_time.copy()
    if "prontuario_anonimo" in df_pac_time.columns:
        # um registro por paciente por ano (mantém última internação no ano)
        df_pac_time = (
            df_pac_time.sort_values(["prontuario_anonimo", "data_internacao"])
            .groupby(["prontuario_anonimo", "ano"], as_index=False)
            .tail(1)
        )

    if tipo_comp == "Ano":
        anos = sorted(df_time["ano"].dropna().unique().tolist())
        if not anos:
            st.info("Sem anos para comparar.")
        else:
            rows = []
            for a in anos:
                df_a = df_time[df_time["ano"] == a]
                df_p_a = df_pac_time[df_pac_time["ano"] == a]
                val = calcular_indicador_ano(indicador_selecionado, df_a, df_p_a)
                rows.append({"Ano": int(a), "Valor": val})

            df_comp = pd.DataFrame(rows)

            # Formatação eixo Y
            if indicador_selecionado in indicadores_percentual:
                fig = px.line(df_comp, x="Ano", y="Valor", markers=True)
                fig.update_yaxes(title_text="Taxa (%)")
            elif indicador_selecionado in indicadores_media:
                fig = px.line(df_comp, x="Ano", y="Valor", markers=True)
                fig.update_yaxes(title_text="Média (dias)")
            else:
                fig = px.line(df_comp, x="Ano", y="Valor", markers=True)
                fig.update_yaxes(title_text="Quantidade")

            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Tabela auxiliar com formatação
            df_tbl = df_comp.copy()
            if indicador_selecionado in indicadores_percentual:
                # mostra % e numerador do ano
                nums = []
                for a in df_tbl["Ano"].tolist():
                    df_a = df_time[df_time["ano"] == a]
                    df_p_a = df_pac_time[df_pac_time["ano"] == a]
                    num = numerador_indicador_percentual(indicador_selecionado, df_a, df_p_a)
                    nums.append(num)
                df_tbl["Numerador"] = nums
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%".replace(".", ","))
                df_tbl["Numerador"] = df_tbl["Numerador"].map(lambda x: "—" if x is None else f"{int(x):,}".replace(",", "."))
            elif indicador_selecionado in indicadores_media:
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ","))
            else:
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{int(x):,}".replace(",", "."))

            st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    else:
        # Mês dentro do ano
        anos = sorted(df_time["ano"].dropna().unique().tolist())
        if not anos:
            st.info("Sem anos para comparar.")
        else:
            ano_ref = st.selectbox("Selecione o ano:", anos, index=len(anos) - 1)
            df_y = df_time[df_time["ano"] == ano_ref].copy()

            # base pacientes por mês (para indicador "Quantidade de pacientes")
            df_p_y = df_y.copy()
            if "prontuario_anonimo" in df_p_y.columns:
                df_p_y["mes"] = pd.to_datetime(df_p_y["data_internacao"], errors="coerce").dt.month
                df_p_y = (
                    df_p_y.sort_values(["prontuario_anonimo", "data_internacao"])
                    .groupby(["prontuario_anonimo", "mes"], as_index=False)
                    .tail(1)
                )

            meses = list(range(1, 13))
            rows = []
            for m in meses:
                df_m = df_y[df_y["mes"] == m]
                df_p_m = df_p_y[df_p_y["mes"] == m] if "mes" in df_p_y.columns else df_p_y.iloc[0:0]

                if df_m.empty:
                    rows.append({"Mês": m, "Valor": np.nan})
                    continue

                val = calcular_indicador_ano(indicador_selecionado, df_m, df_p_m)
                rows.append({"Mês": m, "Valor": val})

            df_comp = pd.DataFrame(rows)
            df_comp["MêsNome"] = df_comp["Mês"].map(
                {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
            )

            fig = px.line(df_comp, x="MêsNome", y="Valor", markers=True)
            if indicador_selecionado in indicadores_percentual:
                fig.update_yaxes(title_text="Taxa (%)")
            elif indicador_selecionado in indicadores_media:
                fig.update_yaxes(title_text="Média (dias)")
            else:
                fig.update_yaxes(title_text="Quantidade")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # tabela
            df_tbl = df_comp[["MêsNome", "Valor"]].copy()
            if indicador_selecionado in indicadores_percentual:
                nums = []
                for m in meses:
                    df_m = df_y[df_y["mes"] == m]
                    df_p_m = df_p_y[df_p_y["mes"] == m] if "mes" in df_p_y.columns else df_p_y.iloc[0:0]
                    num = numerador_indicador_percentual(indicador_selecionado, df_m, df_p_m) if not df_m.empty else None
                    nums.append(num)
                df_tbl["Numerador"] = nums
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%".replace(".", ","))
                df_tbl["Numerador"] = df_tbl["Numerador"].map(lambda x: "—" if x is None else f"{int(x):,}".replace(",", "."))
            elif indicador_selecionado in indicadores_media:
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ","))
            else:
                df_tbl["Valor"] = df_tbl["Valor"].map(lambda x: "—" if pd.isna(x) else f"{int(x):,}".replace(",", "."))

            st.dataframe(df_tbl, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------
# Rodapé / Debug
# --------------------------------------------------------------------
with st.expander("🔎 Diagnóstico rápido"):
    st.write("Linhas (eventos) no recorte:", len(df_f))
    st.write("Internações únicas:", df_f["codigo_internacao"].nunique() if "codigo_internacao" in df_f.columns else "—")
    st.write("Pacientes únicos (no recorte):", df_pac["prontuario_anonimo"].nunique() if "prontuario_anonimo" in df_pac.columns else "—")
    st.write("Pacientes únicos (caracterização, sem período):", pacientes_base_count)

    st.caption("Se algum indicador ficar diferente do esperado, me diga qual e eu ajusto a regra (numerador/base).")

