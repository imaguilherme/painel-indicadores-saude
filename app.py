import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os

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
            0,
            8,
            17,
            26,
            35,
            44,
            53,
            62,
            71,
            80,
            89,
            200,
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
                header = TRUE,
                delim = ',',
                encoding = 'utf-8',
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

        CREATE VIEW dataset AS
        SELECT
            i.*,
            e.* EXCLUDE (prontuario_anonimo),
            c.* EXCLUDE (prontuario_anonimo)
        FROM internacoes_base i
        LEFT JOIN evolu_n e USING (prontuario_anonimo)
        LEFT JOIN cids_n c USING (prontuario_anonimo);

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
# FILTROS
# --------------------------------------------------------------------


def build_filters(df: pd.DataFrame):
    if df is None:
        st.error("Dataset não carregado.")
        st.stop()

    st.sidebar.header("Filtros")

    def _multiselect_com_todos(titulo: str, opcoes: list, key: str, default=None, help_text: str | None = None):
        all_token = "Selecionar todos"
        prev_key = f"__prev_{key}"

        opcoes = [str(x) for x in opcoes]
        if default is None:
            default = opcoes
        default = [str(x) for x in default if str(x) in opcoes]

        if key not in st.session_state:
            if len(opcoes) > 0 and set(default) == set(opcoes):
                st.session_state[key] = [all_token] + opcoes
            else:
                st.session_state[key] = default

        if prev_key not in st.session_state:
            st.session_state[prev_key] = st.session_state.get(key, []).copy()

        def _on_change():
            prev = st.session_state.get(prev_key, [])
            curr = st.session_state.get(key, [])

            if all_token in curr and all_token not in prev:
                st.session_state[key] = [all_token] + opcoes

            elif all_token in prev and all_token in curr and set(curr) != set([all_token] + opcoes):
                st.session_state[key] = [x for x in curr if x != all_token]

            else:
                curr_wo = [x for x in curr if x != all_token]

                if len(opcoes) > 0 and set(curr_wo) == set(opcoes):
                    st.session_state[key] = [all_token] + opcoes
                else:
                    st.session_state[key] = curr_wo

            st.session_state[prev_key] = st.session_state.get(key, []).copy()

        with st.sidebar.container(border=True):
            st.markdown(f"**{titulo}**")

            valid_options = [all_token] + opcoes
            curr_safe = [x for x in st.session_state.get(key, []) if x in valid_options]
            prev_safe = [x for x in st.session_state.get(prev_key, []) if x in valid_options]
            st.session_state[key] = curr_safe
            st.session_state[prev_key] = prev_safe

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
            df_estado = df[df[estado_col].isin(estados_sel)]

    regiao_col = next(
        (c for c in df.columns if "regiao" in c.lower() and "saud" in c.lower()),
        None,
    )

    regioes_sel = []
    if regiao_col and df_estado is not None and not df_estado.empty:
        regioes = sorted(df_estado[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = _multiselect_com_todos(
            "Região de saúde",
            regioes,
            key="ms_regioes",
            default=regioes,
        )

    cidade_col = "cidade_moradia" if "cidade_moradia" in df.columns else None
    cidades_sel = []

    df_cidade_base = df_estado
    if regiao_col and regioes_sel and df_cidade_base is not None and not df_cidade_base.empty:
        df_cidade_base = df_cidade_base[df_cidade_base[regiao_col].isin(regioes_sel)]

    if cidade_col and df_cidade_base is not None and not df_cidade_base.empty:
        cidade_vals = sorted(df_cidade_base[cidade_col].dropna().astype(str).unique().tolist())
        default_cidades = cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        cidades_sel = _multiselect_com_todos(
            "Município de residência",
            cidade_vals,
            key="ms_cidades",
            default=default_cidades,
        )

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
    if include_period and "data_internacao" in df.columns and f.get("periodo"):
        ini, fim = f["periodo"]
        ini = pd.to_datetime(ini)
        fim = pd.to_datetime(fim) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df["data_internacao"] >= ini) & (df["data_internacao"] <= fim)]

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



def _normalize_chart_selection(selection: dict | None) -> dict:
    if not selection:
        return {}

    out = {}
    for dim, values in selection.items():
        if values is None:
            continue
        if not isinstance(values, (list, tuple, set)):
            values = [values]

        cleaned = []
        for v in values:
            if pd.isna(v):
                continue
            cleaned.append(v)

        unique_vals = []
        seen = set()
        for v in cleaned:
            key = repr(v)
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)

        if unique_vals:
            try:
                unique_vals = sorted(unique_vals, key=lambda x: str(x))
            except Exception:
                pass
            out[dim] = unique_vals
    return out


def _extract_plotly_selection_points(widget_state) -> list:
    if widget_state is None:
        return []

    selection = None
    if hasattr(widget_state, "selection"):
        selection = widget_state.selection
    elif isinstance(widget_state, dict):
        selection = widget_state.get("selection")

    if selection is None:
        return []

    if hasattr(selection, "points"):
        points = selection.points
    elif isinstance(selection, dict):
        points = selection.get("points", [])
    else:
        points = []

    return list(points or [])


def _parse_points_from_customdata(points: list, dim_names: list[str]) -> dict:
    selecionados = {dim: [] for dim in dim_names}

    for pt in points:
        custom = None
        if hasattr(pt, "customdata"):
            custom = pt.customdata
        elif isinstance(pt, dict):
            custom = pt.get("customdata")

        if custom is None:
            custom = []

        if not isinstance(custom, (list, tuple)):
            custom = [custom]

        for idx, dim in enumerate(dim_names):
            if idx < len(custom) and pd.notna(custom[idx]):
                selecionados[dim].append(custom[idx])

    return _normalize_chart_selection(selecionados)


def _parse_treemap_points(points: list) -> dict:
    selecionados = {"uf": [], "regiao_saude": [], "cidade_moradia": []}

    for pt in points:
        custom = None
        if hasattr(pt, "customdata"):
            custom = pt.customdata
        elif isinstance(pt, dict):
            custom = pt.get("customdata")

        if isinstance(custom, (list, tuple)) and len(custom) >= 3:
            if pd.notna(custom[0]):
                selecionados["uf"].append(custom[0])
            if pd.notna(custom[1]):
                selecionados["regiao_saude"].append(custom[1])
            if pd.notna(custom[2]):
                selecionados["cidade_moradia"].append(custom[2])
            continue

        raw_id = None
        if hasattr(pt, "id"):
            raw_id = pt.id
        elif isinstance(pt, dict):
            raw_id = pt.get("id")

        if raw_id:
            partes = [p for p in str(raw_id).split("/") if p]
            for item in partes:
                if item.startswith("UF: "):
                    selecionados["uf"].append(item.replace("UF: ", "", 1).strip())
                elif item.startswith("RS: "):
                    selecionados["regiao_saude"].append(item.replace("RS: ", "", 1).strip())
                elif item.startswith("Mun: "):
                    selecionados["cidade_moradia"].append(item.replace("Mun: ", "", 1).strip())

    return _normalize_chart_selection(selecionados)


def combine_chart_filters(filters_by_chart: dict | None) -> dict:
    if not filters_by_chart:
        return {}

    agrupado = {}
    for _, selection in filters_by_chart.items():
        selection = _normalize_chart_selection(selection)
        for dim, values in selection.items():
            conjunto = set(values)
            if dim not in agrupado:
                agrupado[dim] = conjunto
            else:
                agrupado[dim] = agrupado[dim].intersection(conjunto)

    combinado = {}
    for dim, values in agrupado.items():
        if values:
            combinado[dim] = sorted(values, key=lambda x: str(x))
    return combinado


def sanitize_chart_filter_state(df_ref: pd.DataFrame) -> dict:
    filters_by_chart = dict(st.session_state.get("__chart_filters_by_chart", {}))
    changed = False

    for chart_id, selection in list(filters_by_chart.items()):
        selection = _normalize_chart_selection(selection)
        cleaned = {}

        for dim, values in selection.items():
            if dim not in df_ref.columns:
                cleaned[dim] = values
                continue

            valid_values = set(df_ref.loc[df_ref[dim].notna(), dim].tolist())
            kept = [v for v in values if v in valid_values]
            if kept:
                cleaned[dim] = kept

        if cleaned:
            if cleaned != selection:
                changed = True
            filters_by_chart[chart_id] = cleaned
        else:
            filters_by_chart.pop(chart_id, None)
            changed = True

    if changed:
        st.session_state["__chart_filters_by_chart"] = filters_by_chart

    return filters_by_chart


def apply_chart_filters(df: pd.DataFrame | None, filters_by_chart: dict | None) -> pd.DataFrame | None:
    if df is None:
        return None

    combined = combine_chart_filters(filters_by_chart)
    if not combined:
        return df

    out = df.copy()
    for dim, values in combined.items():
        if dim in out.columns and values:
            out = out[out[dim].isin(values)]
    return out


def show_active_chart_filters(filters_by_chart: dict | None):
    combined = combine_chart_filters(filters_by_chart)
    if not combined:
        st.caption("Seleções dos gráficos: nenhuma")
        return

    partes = []
    for dim, values in combined.items():
        rotulo = dim.replace("_", " ").title()
        partes.append(f"**{rotulo}:** " + ", ".join(str(v) for v in values))
    st.caption("Seleções dos gráficos: " + " | ".join(partes))


def render_interactive_chart(
    fig,
    chart_id: str,
    parser,
    use_container_width: bool = True,
    config: dict | None = None,
):
    fig.update_layout(clickmode="event+select", uirevision="crossfilter")

    widget_key = f"plot_{chart_id}_{st.session_state.get('__chart_nonce', 0)}"
    config = config or {}

    supports_selection = True
    try:
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            key=widget_key,
            config=config,
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )
    except TypeError:
        supports_selection = False
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            key=widget_key,
            config=config,
        )

    if not supports_selection:
        return

    widget_state = st.session_state.get(widget_key)
    points = _extract_plotly_selection_points(widget_state)
    new_selection = _normalize_chart_selection(parser(points))

    filters_by_chart = dict(st.session_state.get("__chart_filters_by_chart", {}))
    old_selection = _normalize_chart_selection(filters_by_chart.get(chart_id, {}))

    if new_selection != old_selection:
        if new_selection:
            filters_by_chart[chart_id] = new_selection
        else:
            filters_by_chart.pop(chart_id, None)

        st.session_state["__chart_filters_by_chart"] = filters_by_chart
        st.rerun()


# --------------------------------------------------------------------
# INTERFACE PRINCIPAL
# --------------------------------------------------------------------

st.title("Perfil dos Pacientes")

# --------------------------------------------------------------------
# CARREGAMENTO DIRETO DOS CSVs DA RAIZ DO PROJETO
# --------------------------------------------------------------------

if "df" not in st.session_state:
    evo_path = "caracterizacao.csv"
    proc_path = "procedimentos.csv"
    cti_path = "cids.csv"

    missing = [p for p in [evo_path, proc_path, cti_path] if not os.path.exists(p)]
    if missing:
        st.error(
            "Os seguintes arquivos não foram encontrados na raiz do projeto: "
            + ", ".join(missing)
        )
        st.stop()

    con = load_duckdb((evo_path, proc_path, cti_path))

    df_tmp = df_from_duckdb(con, "SELECT * FROM dataset")
    df_tmp = _post_load(df_tmp)

    cid_df, sigtap_df, geo_df = load_aux_tables()
    df_tmp = enrich_with_aux_tables(df_tmp, cid_df, sigtap_df, geo_df)

    df_base_tmp = df_from_duckdb(con, "SELECT * FROM pacientes_base")
    df_base_tmp = _post_load(df_base_tmp)
    df_base_tmp = enrich_with_aux_tables(df_base_tmp, cid_df, sigtap_df, geo_df)

    st.session_state["df"] = df_tmp
    st.session_state["df_base"] = df_base_tmp

df = st.session_state["df"]
df_base = st.session_state.get("df_base")

if df is None or df.empty:
    st.error("Dataset vazio ou não carregado corretamente.")
    st.stop()

f = build_filters(df)
df_sidebar_f = apply_filters(df, f, include_period=True)
df_base_sidebar_f = apply_filters(df_base, f, include_period=False) if df_base is not None else None

filters_by_chart = sanitize_chart_filter_state(df_sidebar_f)
df_f = apply_chart_filters(df_sidebar_f, filters_by_chart)
df_base_f = apply_chart_filters(df_base_sidebar_f, filters_by_chart) if df_base_sidebar_f is not None else None
df_pac = pacientes_unicos(df_f)

pacientes_base_count = (
    df_base_f["prontuario_anonimo"].nunique()
    if (df_base_f is not None and "prontuario_anonimo" in df_base_f.columns)
    else np.nan
)

st.divider()

modo_perfil = True

pacientes, internacoes, tmi, mort_hosp = kpis(df_f, df_pac)
ri_proc = reinternacao_30d_pos_proced(df_f)
ri_alta = reinternacao_30d_pos_alta(df_f)
uti_pct = internacao_uti_pct(df_f)
tmi_uti = tempo_medio_uti_dias(df_f)
mort_30_proc = mortalidade_30d_pos_proced(df_f)
mort_30_alta = mortalidade_30d_pos_alta(df_f)

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


def get_sexo_color_map(categories):
    mapa = {}
    for s in categories:
        chave = str(s)
        norm = chave.strip().upper()
        if norm in ["M", "MASCULINO"]:
            mapa[chave] = "#6794DC"
        elif norm in ["F", "FEMININO"]:
            mapa[chave] = "#E86F86"
        else:
            mapa[chave] = "#A3A3A3"
    return mapa


st.markdown("### Indicadores disponíveis")

indicador_selecionado = st.radio(
    "Selecione o indicador para detalhar e para o comparativo temporal:",
    indicadores_icardio,
    horizontal=True,
)


col_cf_a, col_cf_b = st.columns([1.1, 3.0])
with col_cf_a:
    if st.button("Limpar seleções dos gráficos", use_container_width=True):
        st.session_state["__chart_filters_by_chart"] = {}
        st.session_state["__chart_nonce"] = st.session_state.get("__chart_nonce", 0) + 1
        st.rerun()
with col_cf_b:
    st.caption("Clique nos gráficos para aplicar filtros acumulativos. Para remover tudo, use o botão ao lado.")
    show_active_chart_filters(filters_by_chart)


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


def _fmt_int_pt(v) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "—"


def indicador_percentual_info(nome: str, df_eventos: pd.DataFrame):
    if df_eventos is None or df_eventos.empty:
        return np.nan, np.nan, np.nan

    if "codigo_internacao" in df_eventos.columns:
        denom = df_eventos["codigo_internacao"].nunique()
    else:
        denom = len(df_eventos)

    if not denom:
        return np.nan, 0, 0

    if nome == "Internação em UTI (%)":
        e = marcar_uti_flag(df_eventos.copy())
        numer = e.loc[e["uti_flag"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["uti_flag"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    if nome == "Reinternação em até 30 dias do procedimento (%)":
        e = marcar_reinternacoes(df_eventos.copy())
        numer = e.loc[e["reint_30d_proc"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["reint_30d_proc"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    if nome == "Reinternação em até 30 dias da alta (%)":
        e = marcar_reinternacoes(df_eventos.copy())
        numer = e.loc[e["reint_30d_alta"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["reint_30d_alta"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    if nome == "Mortalidade hospitalar (%)":
        e = marcar_obito_periodo(df_eventos.copy())
        numer = e.loc[e["obito_no_periodo"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["obito_no_periodo"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    if nome == "Mortalidade em até 30 dias do procedimento (%)":
        e = marcar_mort_30d_proc(df_eventos.copy())
        numer = e.loc[e["obito_30d_proc"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["obito_30d_proc"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    if nome == "Mortalidade em até 30 dias da alta (%)":
        e = marcar_mort_30d_alta(df_eventos.copy())
        numer = e.loc[e["obito_30d_alta"].fillna(False), "codigo_internacao"].nunique() if "codigo_internacao" in e.columns else int(e["obito_30d_alta"].fillna(False).sum())
        return (numer / denom * 100.0), numer, denom

    return np.nan, np.nan, np.nan


valor_ind = calcular_indicador(indicador_selecionado)

pct_numer = pct_denom = np.nan
if indicador_selecionado in indicadores_percentual:
    _, pct_numer, pct_denom = indicador_percentual_info(indicador_selecionado, df_f)

if pd.isna(valor_ind):
    texto_valor = "—"
elif indicador_selecionado in [
    "Quantidade de pacientes",
    "Quantidade de internações",
    "Quantidade de procedimentos",
]:
    texto_valor = _fmt_int_pt(valor_ind)
elif indicador_selecionado in indicadores_percentual:
    if pd.notna(pct_numer) and pd.notna(pct_denom) and pct_denom:
        texto_valor = f"{valor_ind:.2f}% ({_fmt_int_pt(pct_numer)}/{_fmt_int_pt(pct_denom)})"
    else:
        texto_valor = f"{valor_ind:.2f}%"
else:
    texto_valor = f"{valor_ind:.2f}".replace(".", ",")


def format_val_for_card(indicador: str, v: float) -> str:
    if pd.isna(v):
        return "—"

    if indicador in indicadores_percentual:
        return f"{v:.2f}%"

    if indicador in indicadores_media:
        return f"{v:.1f}"

    if indicador in [
        "Quantidade de pacientes",
        "Quantidade de internações",
        "Quantidade de procedimentos",
    ]:
        v_int = int(round(float(v)))
        if abs(v_int) >= 1000:
            return f"{int(v_int/1000)} Mil"
        return f"{v_int}"

    return f"{v:,.0f}".replace(",", ".")


def card_bar_fig(
    df_cat: pd.DataFrame,
    cat_col: str,
    indicador: str,
    colors=None,
    color_map=None,
    height: int = 90,
    custom_data=None,
):
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
        custom_data=custom_data,
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


st.divider()

if indicador_selecionado == "Quantidade de pacientes" and modo_perfil:
    base_charts = df_pac.copy()
else:
    base_charts = definir_base_para_indicador(indicador_selecionado, df_f, df_pac)

base_charts = adicionar_peso_por_indicador(base_charts, indicador_selecionado)

col_esq, col_meio, col_dir = st.columns([1.1, 1.3, 1.1])

# --------------------------------------------------------------------
# PRIMEIRA COLUNA
# --------------------------------------------------------------------
with col_esq:
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
            custom_data=["sexo"],
        )
        render_interactive_chart(
            fig,
            chart_id="sexo",
            parser=lambda points: _parse_points_from_customdata(points, ["sexo"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Coluna 'sexo' não encontrada.")

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
            custom_data=["etnia", "sexo"],
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
        render_interactive_chart(
            fig,
            chart_id="etnia_sexo",
            parser=lambda points: _parse_points_from_customdata(points, ["etnia", "sexo"]),
            use_container_width=True,
            config={"displayModeBar": True},
        )
    else:
        st.info("Requer colunas 'etnia' e 'sexo'.")

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
            x_vals = -values if idx == 0 else values

            cor = sexo_color_map.get(sexo_cat, "#A3A3A3")

            fig.add_bar(
                y=pivot.index,
                x=x_vals,
                name=str(sexo_cat),
                orientation="h",
                marker_color=cor,
                text=np.round(values, 2),
                textposition="outside",
                customdata=list(zip(pivot.index.tolist(), [sexo_cat] * len(pivot.index))),
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

        render_interactive_chart(
            fig,
            chart_id="faixa_etaria_sexo",
            parser=lambda points: _parse_points_from_customdata(points, ["faixa_etaria", "sexo"]),
            use_container_width=True,
        )
    else:
        st.info("Requer colunas 'faixa_etaria' e 'sexo'.")

# --------------------------------------------------------------------
# SEGUNDA COLUNA
# --------------------------------------------------------------------
with col_meio:
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
                car_colors.append("#4CAF50")
            elif s.startswith("URG"):
                car_colors.append("#2E7D32")
            elif s.startswith("EME") or s.startswith("EMG"):
                car_colors.append("#FBC02D")
            else:
                car_colors.append("#7A6FB3")

        fig = card_bar_fig(
            df_car,
            cat_col=carater_col,
            indicador=indicador_selecionado,
            colors=car_colors,
            height=90,
            custom_data=[carater_col],
        )
        render_interactive_chart(
            fig,
            chart_id="carater_atendimento",
            parser=lambda points, dim=carater_col: _parse_points_from_customdata(points, [dim]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Coluna de caráter não encontrada.")

    st.markdown("---")

    st.subheader("Procedimentos")
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
            custom_data=[pcol],
            color_discrete_sequence=["#4C72B0"],
        )
        fig.update_layout(
            xaxis_title=label_eixo_x(indicador_selecionado),
            yaxis_title="",
            height=260,
            margin=dict(t=40, b=40),
        )
        render_interactive_chart(
            fig,
            chart_id="procedimento",
            parser=lambda points, dim=pcol: _parse_points_from_customdata(points, [dim]),
            use_container_width=True,
        )
    else:
        st.info("Não encontrei coluna de procedimento agregada.")

    st.markdown("---")

    st.subheader("CID (capítulo / grupo)")

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
            custom_data=["cid_grupo"],
            color_discrete_sequence=["#55A868"],
        )
        fig.update_layout(
            xaxis_title=label_eixo_x(indicador_selecionado),
            yaxis_title="",
            height=260,
            margin=dict(t=40, b=40),
        )
        render_interactive_chart(
            fig,
            chart_id="cid_grupo",
            parser=lambda points: _parse_points_from_customdata(points, ["cid_grupo"]),
            use_container_width=True,
        )
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
                custom_data=[col_cid],
                color_discrete_sequence=["#55A868"],
            )
            fig.update_layout(
                xaxis_title=label_eixo_x(indicador_selecionado),
                yaxis_title="",
                height=260,
                margin=dict(t=40, b=40),
            )
            render_interactive_chart(
                fig,
                chart_id="cid_diagnostico",
                parser=lambda points, dim=col_cid: _parse_points_from_customdata(points, [dim]),
                use_container_width=True,
            )
        else:
            st.info("Não encontrei nenhuma coluna de CID ou diagnóstico no dataset.")

    st.markdown("---")

    st.subheader("Estado → Região de Saúde → Município de residência")

    if {"uf", "regiao_saude", "cidade_moradia"}.issubset(base_charts.columns):
        df_geo_raw = base_charts.copy()

        for col in ["uf", "regiao_saude", "cidade_moradia"]:
            df_geo_raw[col] = df_geo_raw[col].astype("string").str.strip()
            df_geo_raw[col] = df_geo_raw[col].replace({"": pd.NA})

        df_geo_raw["uf"] = df_geo_raw["uf"].fillna("Sem UF").str.upper()
        df_geo_raw["regiao_saude"] = df_geo_raw["regiao_saude"].fillna("Sem região de saúde")
        df_geo_raw["cidade_moradia"] = df_geo_raw["cidade_moradia"].fillna("Sem município")

        df_geo_raw["uf_lbl"] = "UF: " + df_geo_raw["uf"].astype(str)
        df_geo_raw["regiao_lbl"] = "RS: " + df_geo_raw["regiao_saude"].astype(str)
        df_geo_raw["cidade_lbl"] = "Mun: " + df_geo_raw["cidade_moradia"].astype(str)

        df_geo_plot = agrega_para_grafico(
            df_geo_raw,
            ["uf", "regiao_saude", "cidade_moradia", "uf_lbl", "regiao_lbl", "cidade_lbl"],
            indicador_selecionado,
        )
        df_geo_plot["valor"] = df_geo_plot["valor"].clip(lower=0)
        df_geo_plot["valor_plot"] = np.sqrt(df_geo_plot["valor"])

        fig = px.treemap(
            df_geo_plot,
            path=["uf_lbl", "regiao_lbl", "cidade_lbl"],
            values="valor_plot",
            custom_data=["uf", "regiao_saude", "cidade_moradia"],
        )
        fig.update_layout(height=380, margin=dict(t=40, l=0, r=0, b=0))
        render_interactive_chart(
            fig,
            chart_id="geografia",
            parser=_parse_treemap_points,
            use_container_width=True,
        )
    else:
        st.info("Colunas 'uf', 'regiao_saude' ou 'cidade_moradia' não disponíveis.")

# --------------------------------------------------------------------
# TERCEIRA COLUNA
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

    if indicador_selecionado == "Quantidade de pacientes" and pd.notna(pacientes_base_count):
        st.caption(
            f"Também na base de caracterização (sem filtro de período): **{int(pacientes_base_count):,}**".replace(",", ".")
        )

    st.markdown("---")

# --------------------------------------------------------------------
# COMPARATIVO TEMPORAL
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

else:
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
