# app.py — Painel de Indicadores (Pacientes / Internações)
# Requisitos: streamlit, pandas, numpy, plotly, duckdb, python-dateutil, pyarrow

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os
import tempfile

st.set_page_config(page_title="Painel de Pacientes", layout="wide")

# --------------------------------------------------------------------
# FUNÇÕES DE CARGA E PRÉ-PROCESSAMENTO
# --------------------------------------------------------------------


def _post_load(df: pd.DataFrame) -> pd.DataFrame:
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

    # ----------------- ano da internação (derivado) -----------------
    if (
        "data_internacao" in df.columns
        and "ano_internacao" not in df.columns
        and "ano" not in df.columns
    ):
        df["ano_internacao"] = df["data_internacao"].dt.year

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
        for c in ["codigo_internacao", "prontuario_anonimo", "data_internacao", "data_alta"]
        if c in df.columns
    ]
    if keys:
        df = df.drop_duplicates(subset=keys)

    return df


@st.cache_data(show_spinner=False)
def load_parquet(file):
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

    # cria as 3 views brutas
    make_view("evolu", evo)
    make_view("proced", proc)
    make_view("cids", cti)

    # normaliza prontuário e agrega procedimentos
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

        -- agregação por prontuário para não multiplicar linhas
        CREATE VIEW proc_agg AS
        SELECT
            prontuario_anonimo,
            COUNT(*)                        AS n_proced,
            ANY_VALUE(codigo_procedimento)  AS proc_prim,
            ANY_VALUE(procedimento)         AS proc_nome_prim,
            ANY_VALUE(natureza_agend)       AS natureza_agend
        FROM proc_n
        GROUP BY prontuario_anonimo;

        -- dataset final
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
    """
    Enriquecimento opcional com:
    - CID-10 (capítulo / grupo)
    - Procedimentos SIGTAP (grupo / subgrupo / forma org.)
    - Geografia (UF / macro / região de saúde)
    """
    if df is None:
        return df

    df_enriched = df.copy()

    # -------- CID-10 --------
    if cid_file is not None:
        try:
            cid_df = pd.read_csv(cid_file, dtype=str)
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

    # -------- Procedimentos (SIGTAP) --------
    if sigtap_file is not None:
        try:
            sig_df = pd.read_csv(sigtap_file, dtype=str)
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

    # -------- Geografia (UF / Macro / Região de Saúde) --------
    if geo_file is not None:
        try:
            geo_df = pd.read_csv(geo_file, dtype=str)
            geo_df.columns = [c.lower() for c in geo_df.columns]

            if "cidade_moradia" in df_enriched.columns and {"no_municipio", "sg_uf"}.issubset(
                geo_df.columns
            ):
                # CIDADE_MORADIA está no formato "cidade, UF"
                partes = (
                    df_enriched["cidade_moradia"].astype(str).str.split(",", n=1, expand=True)
                )
                df_enriched["cidade_nome_norm"] = partes[0].str.upper().str.strip()
                if partes.shape[1] > 1:
                    df_enriched["uf_from_cidade"] = partes[1].str.upper().str.strip()
                else:
                    df_enriched["uf_from_cidade"] = np.nan

                geo_df["no_municipio_norm"] = (
                    geo_df["no_municipio"].astype(str).str.upper().str.strip()
                )
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
    """
    Internação em UTI (%):
    percentual de internações que tiveram passagem pela UTI.
    """
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
    """
    Tempo médio de internação em UTI (dias).
    """
    if not {"dt_entrada_cti", "dt_saida_cti"}.issubset(df.columns):
        return np.nan

    e = df.copy()
    dias_uti = (e["dt_saida_cti"] - e["dt_entrada_cti"]).dt.days
    dias_uti = dias_uti.replace([np.inf, -np.inf], np.nan).dropna()
    return dias_uti.mean() if len(dias_uti) else np.nan


def mortalidade_30d_pos_proced(df: pd.DataFrame):
    """
    Mortalidade em até 30 dias do procedimento (%).
    Usa como melhor proxy de data do procedimento:
    data_procedimento > data_cirurgia > data_cirurgia_min > data_internacao.
    """
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
    """
    Mortalidade em até 30 dias da alta (%).
    """
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
# FUNÇÕES AUXILIARES PARA O INDICADOR SELECIONADO
# --------------------------------------------------------------------


def definir_base_para_indicador(indicador, df_f, df_pac):
    """
    Define qual base será usada nos gráficos,
    dependendo do indicador.
    """
    if indicador == "Quantidade de pacientes":
        return df_pac.copy()
    # Para todos os outros, trabalhamos em nível de internação/evento
    return df_f.copy()


def adicionar_peso_por_indicador(df: pd.DataFrame, indicador: str) -> pd.DataFrame:
    """
    Cria a coluna 'peso' que será somada nos gráficos.
    A ideia é sempre usar CONTAGEM (soma de eventos) e não média.
    """
    df = df.copy()
    df["peso"] = 0.0

    if indicador in ["Quantidade de pacientes", "Quantidade de internações"]:
        df["peso"] = 1.0

    elif indicador == "Quantidade de procedimentos":
        df["peso"] = df.get("n_proced", 0).fillna(0)

    elif indicador == "Tempo médio de internação (dias)":
        # soma dos dias de permanência; o gráfico mostra total de dias por categoria
        if "dias_permanencia" in df.columns:
            df["peso"] = df["dias_permanencia"].clip(lower=0).fillna(0)
        else:
            df["peso"] = 0.0

    elif indicador == "Internação em UTI (%)":
        df = marcar_uti_flag(df)
        df["peso"] = df["uti_flag"].astype(int)

    elif indicador == "Tempo médio de internação em UTI (dias)":
        if {"dt_entrada_cti", "dt_saida_cti"}.issubset(df.columns):
            dias_uti = (df["dt_saida_cti"] - df["dt_entrada_cti"]).dt.days
            df["peso"] = dias_uti.clip(lower=0).fillna(0)
        else:
            df["peso"] = 0.0

    elif indicador == "Reinternação em até 30 dias do procedimento (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = df["reint_30d_proc"].astype(int)

    elif indicador == "Reinternação em até 30 dias da alta (%)":
        df = marcar_reinternacoes(df)
        df["peso"] = df["reint_30d_alta"].astype(int)

    elif indicador == "Mortalidade hospitalar (%)":
        df = marcar_obito_periodo(df)
        df["peso"] = df["obito_no_periodo"].astype(int)

    elif indicador == "Mortalidade em até 30 dias do procedimento (%)":
        df = marcar_mort_30d_proc(df)
        df["peso"] = df["obito_30d_proc"].astype(int)

    elif indicador == "Mortalidade em até 30 dias da alta (%)":
        df = marcar_mort_30d_alta(df)
        df["peso"] = df["obito_30d_alta"].astype(int)

    return df


# --------------------------------------------------------------------
# FILTROS
# --------------------------------------------------------------------


def build_filters(df: pd.DataFrame):
    if df is None:
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

    return {"ano": ano_sel, "idade": idade_sel, "estado": estados_sel, "regiao": regioes_sel, "cidade": cidades_sel}


def apply_filters(df: pd.DataFrame, f):
    if "ano_internacao" in df.columns and f["ano"]:
        df = df[df["ano_internacao"].isin(f["ano"])]
    elif "ano" in df.columns and f["ano"]:
        df = df[df["ano"].isin(f["ano"])]

    if "idade" in df.columns and f["idade"]:
        df = df[(df["idade"] >= f["idade"][0]) & (df["idade"] <= f["idade"][1])]

    estado_col = next(
        (c for c in df.columns if c.lower() in ["estado_residencia", "uf_residencia", "uf", "estado", "sigla_uf"]),
        None,
    )
    if estado_col and f["estado"]:
        df = df[df[estado_col].isin(f["estado"])]

    regiao_col = next(
        (c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()),
        None,
    )
    if regiao_col and f["regiao"]:
        df = df[df[regiao_col].isin(f["regiao"])]

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
        st.markdown("**Filtros ativos:** " + " | ".join(partes))
    else:
        st.markdown("**Filtros ativos:** nenhum filtro aplicado.")


# --------------------------------------------------------------------
# INTERFACE
# --------------------------------------------------------------------

st.title("Perfil dos Pacientes")

tab_parquet, tab_csv = st.tabs(["Parquet único (recomendado)", "3 CSVs (DuckDB)"])

df = None

with tab_parquet:
    file_parquet = st.file_uploader(
        "Carregue o Parquet único", type=["parquet"], key="pq"
    )
    if file_parquet:
        df = load_parquet(file_parquet)

with tab_csv:
    c1, c2, c3 = st.columns(3)
    evo = c1.file_uploader("EVOLUÇÕES (csv)", type=["csv"], key="evo")
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
        df = df_from_duckdb(con, "SELECT * FROM dataset")
        df = _post_load(df)

if df is None or df.empty:
    st.info("Carregue um Parquet ou os 3 CSVs para iniciar.")
    st.stop()

with st.expander("Carregar tabelas auxiliares (opcional) – CID-10, SIGTAP e Regiões de Saúde"):
    cid_file = st.file_uploader(
        "Tabela de CIDs (LIST_CID_2019_2021_BINDED.csv)", type=["csv"], key="cid_map"
    )
    sigtap_file = st.file_uploader(
        "Tabela de procedimentos SIGTAP (Matriz de Dados do SIGTAP.csv)", type=["csv"], key="sigtap_map"
    )
    geo_file = st.file_uploader(
        "Tabela UF/Macro/Região/Município (UF_Macro_Região_Município.csv)", type=["csv"], key="geo_map"
    )

if cid_file or sigtap_file or geo_file:
    df = enrich_with_aux_tables(df, cid_file, sigtap_file, geo_file)

# --------------------------------------------------------------------
# FILTROS E BASES
# --------------------------------------------------------------------

f = build_filters(df)
df_f = apply_filters(df, f)
df_pac = pacientes_unicos(df_f)

show_active_filters(f)
st.divider()

modo_perfil = st.toggle(
    "Contar por **paciente único** (perfil) nos gráficos quando o indicador for 'Quantidade de pacientes'.",
    value=True,
)

# --------------------------------------------------------------------
# KPIs GLOBAIS
# --------------------------------------------------------------------

pacientes, internacoes, tmi, mort_hosp = kpis(df_f, df_pac)
ri_proc = reinternacao_30d_pos_proced(df_f)
ri_alta = reinternacao_30d_pos_alta(df_f)
uti_pct = internacao_uti_pct(df_f)
tmi_uti = tempo_medio_uti_dias(df_f)
mort_30_proc = mortalidade_30d_pos_proced(df_f)
mort_30_alta = mortalidade_30d_pos_alta(df_f)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "Pacientes (distintos)",
    f"{int(pacientes):,}".replace(",", ".") if pd.notna(pacientes) else "—",
)
k2.metric(
    "Internações",
    f"{int(internacoes):,}".replace(",", ".") if pd.notna(internacoes) else "—",
)
k3.metric(
    "Tempo médio de internação (dias)",
    f"{tmi:.1f}" if pd.notna(tmi) else "—",
)
k4.metric(
    "Reinternação 30d (procedimento)",
    f"{ri_proc:.1f}%" if pd.notna(ri_proc) else "—",
)
k5.metric(
    "Reinternação 30d (alta)",
    f"{ri_alta:.1f}%" if pd.notna(ri_alta) else "—",
)
k6.metric(
    "Mortalidade hospitalar",
    f"{mort_hosp:.1f}%" if pd.notna(mort_hosp) else "—",
)


# --------------------------------------------------------------------
# INDICADOR SELECIONADO
# --------------------------------------------------------------------

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
# --------------------------------------------------------------------

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
    """
    Calcula o mesmo indicador, mas apenas para um subconjunto (ex.: por ano),
    para montar o comparativo anual.
    """
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
    texto_valor = (
        f"{valor_ind:,.2f}".replace(",", ".") if pd.notna(valor_ind) else "—"
    )

st.metric("Valor do indicador selecionado", texto_valor)

st.divider()

# --------------------------------------------------------------------
# COMPARATIVO ANUAL DO INDICADOR SELECIONADO
# --------------------------------------------------------------------

st.markdown("### Comparativo anual do indicador selecionado")

ano_col = (
    "ano_internacao"
    if "ano_internacao" in df_f.columns
    else ("ano" if "ano" in df_f.columns else None)
)

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
            fig_ano = px.bar(df_plot, x=ano_col, y="valor", text_auto=True)
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

st.divider()

# --------------------------------------------------------------------
# BASE PARA OS GRÁFICOS (RESPONDE AO INDICADOR)
# --------------------------------------------------------------------

if indicador_selecionado == "Quantidade de pacientes" and modo_perfil:
    base_charts = df_pac.copy()
else:
    base_charts = definir_base_para_indicador(indicador_selecionado, df_f, df_pac)

base_charts = adicionar_peso_por_indicador(base_charts, indicador_selecionado)

# --------------------------------------------------------------------
# GRID PRINCIPAL (3 COLUNAS)
# --------------------------------------------------------------------

col_esq, col_meio, col_dir = st.columns([1.1, 1.3, 1.1])

# ============================
# COLUNA ESQUERDA
# ============================
with col_esq:
    c1, c2 = st.columns(2)

    # Sexo
    with c1:
        st.subheader("Sexo")
        if "sexo" in base_charts.columns:
            df_sexo = (
                base_charts.groupby("sexo", dropna=False)["peso"]
                .sum()
                .reset_index()
                .rename(columns={"peso": "valor"})
            )
            fig = px.bar(df_sexo, x="sexo", y="valor", text_auto=True)
            fig.update_layout(height=230, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna 'sexo' não encontrada.")

    # Caráter do atendimento
    with c2:
        st.subheader("Caráter do atendimento")

        carater_col = None
        for cand in [
            "carater_atendimento",
            "caracter_atendimento",
            "carater",
            "caráter_atendimento",
            "carater_atend",
            "natureza_agend",
        ]:
            if cand in base_charts.columns:
                carater_col = cand
                break

        if carater_col:
            df_car = (
                base_charts.groupby(carater_col, dropna=False)["peso"]
                .sum()
                .reset_index()
                .rename(columns={"peso": "valor"})
            )
            ordem = df_car.sort_values("valor", ascending=False)[carater_col].tolist()
            fig = px.bar(
                df_car,
                x=carater_col,
                y="valor",
                text_auto=True,
                category_orders={carater_col: ordem},
            )
            fig.update_layout(
                height=230,
                xaxis_title="",
                margin=dict(t=40, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna de 'caráter do atendimento' não encontrada.")

    # PIRÂMIDE ETÁRIA
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
        df_pira["sexo"] = df_pira["sexo"].astype(str).str.strip().str.title()
        df_pira = df_pira[df_pira["faixa_etaria"].isin(categorias)]

        tabela = (
            df_pira.groupby(["faixa_etaria", "sexo"], dropna=False)["peso"]
            .sum()
            .reset_index(name="n")
        )

        pivot = tabela.pivot(index="faixa_etaria", columns="sexo", values="n").fillna(0)
        pivot = pivot.reindex(categorias)

        fig = go.Figure()
        palette = [
            "#DA83A3", "#91A8C5", "#A3D977", "#E6C76A", "#C28CCB",
            "#6CC5C6", "#E39A8D", "#8AA9B7"
        ]
        color_index = 0

        for sexo_cat in pivot.columns:
            values = pivot[sexo_cat]
            if color_index == 0:
                x_vals = -values
            else:
                x_vals = values

            fig.add_bar(
                y=pivot.index,
                x=x_vals,
                name=str(sexo_cat),
                orientation="h",
                marker_color=palette[color_index % len(palette)],
                text=values.astype(int),
                textposition="outside",
            )
            color_index += 1

        fig.update_layout(
            barmode="overlay",
            height=550,
            xaxis=dict(
                title="Quantidade (conforme indicador)",
                showgrid=False,
            ),
            yaxis=dict(title="Faixa etária", autorange="reversed"),
            margin=dict(l=80, r=80, t=50, b=50),
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Requer colunas 'faixa_etaria' e 'sexo'.")


# ============================
# COLUNA DO MEIO
# ============================
with col_meio:
    st.subheader("Estado → Região de Saúde → Município de residência")

    if {"uf", "regiao_saude", "cidade_moradia"}.issubset(base_charts.columns):
        df_geo_plot = base_charts.dropna(subset=["cidade_moradia"]).copy()
        df_geo_plot["valor"] = df_geo_plot["peso"]

        fig = px.treemap(
            df_geo_plot,
            path=["uf", "regiao_saude", "cidade_moradia"],
            values="valor",
        )
        fig.update_layout(height=550, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Hierarquia: Estado → Região de Saúde → Município. Use os filtros para refinar."
        )
    else:
        st.info(
            "Colunas 'uf', 'regiao_saude' ou 'cidade_moradia' não disponíveis. "
            "Carregue a tabela de Regiões de Saúde no painel para habilitar."
        )

    st.subheader("Raça/Cor × Sexo")
    if {"etnia", "sexo"}.issubset(base_charts.columns):
        df_etnia = (
            base_charts.groupby(["etnia", "sexo"], dropna=False)["peso"]
            .sum()
            .reset_index(name="valor")
        )
        fig = px.bar(
            df_etnia,
            x="etnia",
            y="valor",
            color="sexo",
            barmode="group",
            text_auto=True,
        )
        fig.update_layout(
            xaxis_title="Raça/Cor",
            yaxis_title="Quantidade (conforme indicador)",
            height=320,
            margin=dict(t=40, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Requer colunas 'etnia' e 'sexo'.")

# ============================
# COLUNA DIREITA
# ============================
with col_dir:
    st.subheader("Quantidade de pacientes")
    if pd.notna(pacientes):
        st.markdown(
            f"<h2 style='text-align:center;'>{int(pacientes):,}</h2>".replace(",", "."),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<h2 style='text-align:center;'>—</h2>", unsafe_allow_html=True)
    st.caption("Pacientes distintos no período filtrado")

    st.markdown("---")

    st.subheader("Procedimentos (amostra)")
    proc_cols = [
        c
        for c in base_charts.columns
        if "proc_nome_prim" in c.lower() or c.lower() == "procedimento"
    ]
    if proc_cols:
        pcol = proc_cols[0]
        top_proc = (
            base_charts.groupby(pcol, dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        top_proc = top_proc.sort_values("valor", ascending=False).head(10)
        top_proc.columns = ["Procedimento", "Quantidade (conforme indicador)"]
        fig = px.bar(top_proc, x="Procedimento", y="Quantidade (conforme indicador)", text_auto=True)
        fig.update_layout(
            xaxis_tickangle=-35,
            height=260,
            margin=dict(t=40, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não encontrei coluna de procedimento agregada.")

    st.markdown("---")

    st.subheader("CID (capítulo / grupo) – amostra")

    if "cid_grupo" in base_charts.columns:
        top_cid_grp = (
            base_charts.groupby("cid_grupo", dropna=False)["peso"]
            .sum()
            .reset_index()
            .rename(columns={"peso": "valor"})
        )
        top_cid_grp = top_cid_grp.sort_values("valor", ascending=False).head(10)
        top_cid_grp.columns = ["Grupo CID-10 (amostra)", "Quantidade (conforme indicador)"]
        fig = px.bar(
            top_cid_grp,
            x="Grupo CID-10 (amostra)",
            y="Quantidade (conforme indicador)",
            text_auto=True,
        )
        fig.update_layout(
            xaxis_tickangle=-35,
            height=260,
            margin=dict(t=40, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        cid_col = [
            c
            for c in base_charts.columns
            if ("cid" in c.lower() or "descricao" in c.lower())
        ]
        if cid_col:
            col_cid = cid_col[0]
            top = (
                base_charts.groupby(col_cid, dropna=False)["peso"]
                .sum()
                .reset_index()
                .rename(columns={"peso": "valor"})
            )
            top[col_cid] = top[col_cid].astype(str).str.upper().str[:50]
            top = top.sort_values("valor", ascending=False).head(10)
            top.columns = ["CID/Descrição (amostra)", "Quantidade (conforme indicador)"]
            fig = px.bar(
                top,
                x="CID/Descrição (amostra)",
                y="Quantidade (conforme indicador)",
                text_auto=True,
            )
            fig.update_layout(
                xaxis_tickangle=-35,
                height=260,
                margin=dict(t=40, b=120),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não encontrei informações de CID no dataset.")
