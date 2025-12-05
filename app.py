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
        for c in ["codigo_internacao", "prontuario_anonimo", "data_internacao", "data_alta"]
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


@st.cache_data
def load_aux_tables():
    """
    Carrega as tabelas auxiliares de CID-10, SIGTAP e Regiões de Saúde
    a partir de arquivos CSV que já vêm junto com o app.
    """
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

                df_enriched["cid3"] = (
                    df_enriched["cid"].astype(str).str.strip().str.upper().str[:3]
                )
                cid_df["cid3"] = (
                    cid_df[cid_code_col].astype(str).str.strip().str.upper().str[:3]
                )

                keep_cols = ["cid3"]
                keep_cols += [
                    c
                    for c in cid_df.columns
                    if any(k in c for k in ["capítulo", "capitulo", "grupo"])
                ]
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
                keep_cols += [
                    c
                    for c in sig_df.columns
                    if any(k in c for k in ["grupo", "subgrupo", "forma", "nome"])
                ]
                sig_small = sig_df[keep_cols].drop_duplicates(subset=[sig_code_col])
                df_enriched = df_enriched.merge(
                    sig_small,
                    how="left",
                    left_on=proc_col,
                    right_on=sig_code_col,
                )
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
    """
    Marca obito_no_periodo (mortalidade hospitalar).
    Regra: data_obito entre data_internacao e data_alta (mesmo dia conta),
    considerando apenas a DATA (ignora horário). Se data_alta for nula,
    basta data_obito >= data_internacao.
    """
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
    """Marca reinternações em até 30 dias."""
    df = df.copy()
    required = {"prontuario_anonimo", "codigo_internacao", "data_internacao", "data_alta"}
    if not required.issubset(df.columns):
        df["reint_30d_proc"] = False
        df["reint_30d_alta"] = False
        return df

    s = df.sort_values(
        ["prontuario_anonimo", "data_internacao", "data_alta"],
        kind="mergesort"
    ).copy()

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
# AUXILIARES PARA INDICADOR
# --------------------------------------------------------------------


def definir_base_para_indicador(indicador, df_f, df_pac):
    if indicador == "Quantidade de pacientes":
        return df_pac.copy()
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
# FILTROS
# --------------------------------------------------------------------


def build_filters(df: pd.DataFrame):
    if df is None:
        st.error("Dataset não carregado.")
        st.stop()

    st.sidebar.header("Filtros")

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

        # garantir tupla (ini, fim)
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
        "Idade", min_value=0, max_value=max(idade_max, 1), value=(idade_min, idade_max), step=1
    )

    # ---- Estado ----
    estado_col = next(
        (c for c in df.columns if c.lower() in ["estado_residencia", "uf_residencia", "uf", "estado", "sigla_uf"]),
        None,
    )
    estados_sel = []
    if estado_col:
        estados = sorted(df[estado_col].dropna().astype(str).unique().tolist())
        estados_sel = st.sidebar.multiselect("Estado de residência", estados, default=estados)

    # ---- Região de saúde ----
    regiao_col = next(
        (c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()),
        None,
    )
    regioes_sel = []
    if regiao_col:
        regioes = sorted(df[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = st.sidebar.multiselect("Região de saúde", regioes, default=regioes)

    # ---- Município ----
    cidade_col = "cidade_moradia" if "cidade_moradia" in df.columns else None
    cidades_sel = []
    if cidade_col:
        cidade_vals = sorted(df[cidade_col].dropna().astype(str).unique().tolist())
        default_cidades = cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        cidades_sel = st.sidebar.multiselect(
            "Município de residência (amostra)", cidade_vals, default=default_cidades
        )

    # ---- Sexo ----
    sexo_sel = []
    if "sexo" in df.columns:
        sexos = sorted(df["sexo"].dropna().astype(str).unique().tolist())
        sexo_sel = st.sidebar.multiselect("Sexo", sexos, default=sexos)

    return {
        "periodo": periodo_sel,
        "idade": idade_sel,
        "estado": estados_sel,
        "regiao": regioes_sel,
        "cidade": cidades_sel,
        "sexo": sexo_sel,
    }


def apply_filters(df: pd.DataFrame, f):
    # Período
    if "data_internacao" in df.columns and f.get("periodo"):
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
        df = df[df[estado_col].isin(f["estado"])]

    # Região de saúde
    regiao_col = next(
        (c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()),
        None,
    )
    if regiao_col and f["regiao"]:
        df = df[df[regiao_col].isin(f["regiao"])]

    # Município
    if "cidade_moradia" in df.columns and f["cidade"]:
        df = df[df["cidade_moradia"].isin(f["cidade"])]

    # Sexo
    if "sexo" in df.columns and f["sexo"]:
        df = df[df["sexo"].isin(f["sexo"])]

    return df


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

        st.session_state["df"] = df_tmp
        st.success("Arquivos carregados com sucesso! Painel inicializado.")

        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.stop()

# ------------ Depois de carregado, não mostra mais os uploaders ------------
df = st.session_state["df"]

if df is None or df.empty:
    st.error("Dataset vazio ou não carregado corretamente.")
    st.stop()

# Filtros
f = build_filters(df)
df_f = apply_filters(df, f)
df_pac = pacientes_unicos(df_f)

show_active_filters(f)
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

# Estilo dos botões dos indicadores
st.markdown("""
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
""", unsafe_allow_html=True)

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

# tipos de indicador
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
# --------------------------------------------------------


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
if "%" in indicador_selecionado:
    texto_valor = f"{valor_ind:.2f}%" if pd.notna(valor_ind) else "—"
else:
    texto_valor = f"{valor_ind:,.2f}".replace(",", ".") if pd.notna(valor_ind) else "—"


# --------------------------------------------------------------------
# FORMATAÇÃO PARA CARDS
# --------------------------------------------------------------------
def format_val_for_card(indicador: str, v: float) -> str:
    if pd.isna(v):
        return "—"
    if indicador in indicadores_percentual:
        return f"{v:.2f}%"
    if indicador in indicadores_media:
        return f"{v:.1f}"
    # quantidade
    if abs(v) >= 1000:
        return f"{v/1000:.2f} Mil"
    return f"{v:,.0f}".replace(",", ".")


def card_bar_fig(
    df_cat: pd.DataFrame,
    cat_col: str,
    indicador: str,
    colors=None,
    color_map=None,
    height: int = 90,
):
    """Retorna um gráfico Plotly em forma de card (barra única segmentada).

    - colors: lista de cores (sequência padrão)
    - color_map: dict categoria -> cor (fixa por categoria)
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

if indicador_selecionado == "Quantidade de pacientes" and modo_perfil:
    base_charts = df_pac.copy()
else:
    base_charts = definir_base_para_indicador(indicador_selecionado, df_f, df_pac)

base_charts = adicionar_peso_por_indicador(base_charts, indicador_selecionado)

col_esq, col_meio, col_dir = st.columns([1.1, 1.3, 1.1])

# --------------------------------------------------------------------
# PRIMEIRA COLUNA: Sexo; Raça/Cor × Sexo; Pirâmide Etária
# --------------------------------------------------------------------
with col_esq:
    # Sexo (CARD)
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

    # Raça/Cor × Sexo
    st.subheader("Raça/Cor × Sexo")
    if {"etnia", "sexo"}.issubset(base_charts.columns):
        df_etnia = agrega_para_grafico(
            base_charts, ["etnia", "sexo"], indicador_selecionado
        )
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
            tickformat=".2f",
        )
        fig.update_yaxes(title="Raça/Cor")

        fig.update_layout(
            height=350,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Requer colunas 'etnia' e 'sexo'.")

    # Pirâmide Etária
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
        df_pira["sexo"] = df_pira["sexo"].astype(str).str.strip()
        df_pira = df_pira[df_pira["faixa_etaria"].isin(categorias)]

        tabela = agrega_para_grafico(
            df_pira, ["faixa_etaria", "sexo"], indicador_selecionado
        ).rename(columns={"valor": "n"})

        pivot = tabela.pivot(index="faixa_etaria", columns="sexo", values="n").fillna(0)
        pivot = pivot.reindex(categorias).fillna(0)

        fig = go.Figure()

        sexo_color_map = get_sexo_color_map(pivot.columns)

        for idx, sexo_cat in enumerate(pivot.columns):
            values = pivot[sexo_cat]
            x_vals = -values if idx == 0 else values  # lado esquerdo/direito

            cor = sexo_color_map.get(sexo_cat, "#A3A3A3")

            fig.add_bar(
                y=pivot.index,
                x=x_vals,
                name=str(sexo_cat),
                orientation="h",
                marker_color=cor,
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
            showgrid=False,
        )

        fig.update_layout(
            barmode="overlay",
            height=550,
            yaxis=dict(title="Faixa etária", autorange="reversed"),
            margin=dict(l=80, r=80, t=50, b=50),
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Requer colunas 'faixa_etaria' e 'sexo'.")

# --------------------------------------------------------------------
# SEGUNDA COLUNA:
# Caráter do Atendimento; Procedimentos; CID; Treemap
# --------------------------------------------------------------------
with col_meio:
    # Caráter do Atendimento
    st.subheader("Caráter do Atendimento")
    carater_col = None
    for cand in ["carater_atendimento", "caracter_atendimento", "carater", "natureza_agend"]:
        if cand in base_charts.columns:
            carater_col = cand
            break

    if carater_col:
        df_car = agrega_para_grafico(base_charts, [carater_col], indicador_selecionado)
        df_car = df_car.sort_values("valor", ascending=False)

        car_colors = []
        for v in df_car[carater_col]:
            s = str(v).upper()
            if s.startswith("ELE"):
                car_colors.append("#4CAF50")   # eletivo
            elif s.startswith("URG"):
                car_colors.append("#2E7D32")   # urgência
            elif s.startswith("EME") or s.startswith("EMG"):
                car_colors.append("#FBC02D")   # emergência
            else:
                car_colors.append("#7A6FB3")   # padrão

        fig = card_bar_fig(
            df_car,
            cat_col=carater_col,
            indicador=indicador_selecionado,
            colors=car_colors,
            height=90,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Coluna de caráter não encontrada.")

    st.markdown("---")

    # Procedimentos (amostra)
    st.subheader("Procedimentos (amostra)")
    proc_cols = [
        c
        for c in base_charts.columns
        if "proc_nome_prim" in c.lower() or c.lower() == "procedimento"
    ]
    if proc_cols:
        pcol = proc_cols[0]
        top_proc = agrega_para_grafico(base_charts, [pcol], indicador_selecionado)
        top_proc = top_proc.sort_values("valor", ascending=True).tail(10)
        fig = px.bar(
            top_proc,
            y=pcol,
            x="valor",
            orientation="h",
            text="valor",
            color_discrete_sequence=["#4C72B0"],
        )
        fig.update_layout(
            xaxis_title=label_eixo_x(indicador_selecionado),
            yaxis_title="",
            height=260,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não encontrei coluna de procedimento agregada.")

    st.markdown("---")

    # CID (capítulo / grupo) – amostra
    st.subheader("CID (capítulo / grupo) – amostra")

    if "cid_grupo" in base_charts.columns and base_charts["cid_grupo"].notna().any():
        top_cid_grp = agrega_para_grafico(
            base_charts, ["cid_grupo"], indicador_selecionado
        )
        top_cid_grp = top_cid_grp.sort_values("valor", ascending=True).tail(10)
        fig = px.bar(
            top_cid_grp,
            y="cid_grupo",
            x="valor",
            orientation="h",
            text="valor",
            color_discrete_sequence=["#55A868"],
        )
        fig.update_layout(
            xaxis_title=label_eixo_x(indicador_selecionado),
            yaxis_title="",
            height=260,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        cid_candidates = []
        for c in base_charts.columns:
            cl = c.lower()
            if cl.startswith("cid") and "cidade" not in cl:
                cid_candidates.append(c)
            elif "descricao_cid" in cl or "diagnostico" in cl or "diag_princ" in cl:
                cid_candidates.append(c)

        if cid_candidates:
            col_cid = cid_candidates[0]
            top = agrega_para_grafico(base_charts, [col_cid], indicador_selecionado)
            top[col_cid] = top[col_cid].astype(str).str.upper().str[:60]
            top = top.sort_values("valor", ascending=True).tail(10)
            fig = px.bar(
                top,
                y=col_cid,
                x="valor",
                orientation="h",
                text="valor",
                color_discrete_sequence=["#55A868"],
            )
            fig.update_layout(
                xaxis_title=label_eixo_x(indicador_selecionado),
                yaxis_title="",
                height=260,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não encontrei nenhuma coluna de CID ou diagnóstico no dataset.")

    st.markdown("---")

    # Treemap – Estado → Região de Saúde → Município
    st.subheader("Estado → Região de Saúde → Município de residência")

    if {"uf", "regiao_saude", "cidade_moradia"}.issubset(base_charts.columns):
        df_geo_raw = base_charts.dropna(subset=["cidade_moradia"]).copy()
        df_geo_plot = agrega_para_grafico(
            df_geo_raw, ["uf", "regiao_saude", "cidade_moradia"], indicador_selecionado
        )
        df_geo_plot["valor"] = df_geo_plot["valor"].clip(lower=0)
        df_geo_plot["valor_plot"] = np.sqrt(df_geo_plot["valor"])

        fig = px.treemap(
            df_geo_plot,
            path=["uf", "regiao_saude", "cidade_moradia"],
            values="valor_plot",
        )
        fig.update_layout(height=380, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Colunas 'uf', 'regiao_saude' ou 'cidade_moradia' não disponíveis."
        )

# --------------------------------------------------------------------
# TERCEIRA COLUNA:
# Valor do indicador; Boxplot – Idade por sexo
# --------------------------------------------------------------------
with col_dir:
    st.subheader(indicador_selecionado)
    if pd.notna(valor_ind):
        st.markdown(
            f"<h2 style='text-align:center;'>{texto_valor}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<h2 style='text-align:center;'>—</h2>", unsafe_allow_html=True)
    st.caption("Valor do indicador no período filtrado")

    st.markdown("---")

    # Boxplot – Idade por sexo
    st.subheader("Boxplot – Idade por sexo")

    if {"idade", "sexo"}.issubset(base_charts.columns):
        df_box = base_charts.dropna(subset=["idade", "sexo"]).copy()
        df_box["sexo"] = df_box["sexo"].astype(str).str.strip()

        sexo_color_map = get_sexo_color_map(df_box["sexo"].unique())

        fig_box = px.box(
            df_box,
            x="sexo",
            y="idade",
            color="sexo",
            points="all",
            color_discrete_map=sexo_color_map,
        )

        fig_box.update_xaxes(title="Sexo")
        fig_box.update_yaxes(title="Idade (anos)")
        fig_box.update_layout(
            height=400,
            margin=dict(t=40, b=40),
        )

        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Não há colunas 'idade' e 'sexo' disponíveis para o boxplot.")

# --------------------------------------------------------------------
# COMPARATIVO TEMPORAL (ANO ou MÊS)
# --------------------------------------------------------------------

st.divider()
st.markdown("### Comparativo do indicador selecionado")

modo_comp = st.radio(
    "Agrupar por:",
    ["Ano", "Mês"],
    horizontal=True,
)

ano_col = (
    "ano_internacao"
    if "ano_internacao" in df_f.columns
    else ("ano" if "ano" in df_f.columns else None)
)

if modo_comp == "Ano":
    if ano_col:
        df_valid = df_f[~df_f[ano_col].isna()].copy()
        if not df_valid.empty:
            anos_validos = sorted(df_valid[ano_col].dropna().unique())
            linhas = []
            for a in anos_validos:
                df_ano = df_valid[df_valid[ano_col] == a]
                df_pac_ano = pacientes_unicos(df_ano)
                val_ano = calcular_indicador_ano(indicador_selecionado, df_ano, df_pac_ano)
                linhas.append({ano_col: int(a), "valor": val_ano})

            df_plot = pd.DataFrame(linhas).dropna(subset=["valor"]).sort_values(ano_col)

            if not df_plot.empty:
                fig_ano = px.bar(df_plot, x=ano_col, y="valor")
                if "%" in indicador_selecionado:
                    fig_ano.update_traces(
                        texttemplate="%{y:.2f}%",
                        textposition="outside",
                    )
                    fig_ano.update_yaxes(tickformat=".2f")
                else:
                    fig_ano.update_traces(
                        texttemplate="%{y}",
                        textposition="outside",
                    )

                fig_ano.update_layout(
                    xaxis_title="Ano",
                    yaxis_title=indicador_selecionado,
                    height=280,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_ano, use_container_width=True)
            else:
                st.info("Sem valores para o comparativo anual com o indicador selecionado.")
        else:
            st.info("Sem dados para o comparativo anual com os filtros atuais.")
    else:
        st.info("Coluna de ano não encontrada no dataset.")

else:  # modo_comp == "Mês"
    if "data_internacao" in df_f.columns:
        df_valid = df_f[~df_f["data_internacao"].isna()].copy()
        if not df_valid.empty:
            df_valid["mes_ano"] = df_valid["data_internacao"].dt.to_period("M")
            linhas = []
            for m in sorted(df_valid["mes_ano"].unique()):
                df_mes = df_valid[df_valid["mes_ano"] == m]
                df_pac_mes = pacientes_unicos(df_mes)
                val_mes = calcular_indicador_ano(indicador_selecionado, df_mes, df_pac_mes)
                label = f"{m.month:02d}/{m.year}"
                linhas.append({"mes": label, "valor": val_mes})

            df_plot = pd.DataFrame(linhas).dropna(subset=["valor"])

            if not df_plot.empty:
                fig_mes = px.bar(df_plot, x="mes", y="valor")
                if "%" in indicador_selecionado:
                    fig_mes.update_traces(
                        texttemplate="%{y:.2f}%",
                        textposition="outside",
                    )
                    fig_mes.update_yaxes(tickformat=".2f")
                else:
                    fig_mes.update_traces(
                        texttemplate="%{y}",
                        textposition="outside",
                    )

                fig_mes.update_layout(
                    xaxis_title="Mês/Ano",
                    yaxis_title=indicador_selecionado,
                    height=280,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_mes, use_container_width=True)
            else:
                st.info("Sem valores para o comparativo mensal com o indicador selecionado.")
        else:
            st.info("Sem dados para o comparativo mensal com os filtros atuais.")
    else:
        st.info("Coluna 'data_internacao' não encontrada para o comparativo mensal.")
