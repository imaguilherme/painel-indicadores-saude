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
    for c in [
        "data_internacao","data_alta","data_obito",
        "dthr_valida","dt_entrada_cti","dt_saida_cti",
        "data_cirurgia_min","data_cirurgia_max"
    ]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # numéricos
    for c in ["idade","ano","ano_internacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sexo
    if "sexo" in df.columns:
        df["sexo"] = (
            df["sexo"].astype(str).str.strip().str.upper()
              .replace({"M":"Masculino","F":"Feminino",
                        "MASCULINO":"Masculino","FEMININO":"Feminino"})
        )

    # derivar ano da internação, se não vier pronto
    if "data_internacao" in df.columns and "ano_internacao" not in df.columns:
        df["ano_internacao"] = df["data_internacao"].dt.year

    # dias permanência
    if {"data_internacao","data_alta"}.issubset(df.columns):
        df["dias_permanencia"] = (df["data_alta"] - df["data_internacao"]).dt.days

    # faixas etárias
    if "idade" in df.columns:
        bins = [-1,0,4,11,17,24,34,44,54,64,74,84,120]
        labels = ["<1","1–4","5–11","12–17","18–24",
                  "25–34","35–44","45–54","55–64",
                  "65–74","75–84","85+"]
        df["faixa_etaria"] = pd.cut(
            pd.to_numeric(df["idade"], errors="coerce"),
            bins=bins, labels=labels, right=True
        )

    # dedup de eventos (AIH)
    keys = [c for c in ["codigo_internacao","prontuario_anonimo",
                         "data_internacao","data_alta"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)

    return df

@st.cache_data(show_spinner=False)
def load_parquet(file):
    return _post_load(pd.read_parquet(file))

@st.cache_resource(show_spinner=False)
def load_duckdb(csv_paths):
    """
    Registra 3 CSVs (EVOLUCOES, PROCED, CIDS/UTI) e cria view dataset unificada (lazy).

    Ajustes:
    - Normaliza PRONTUARIO_ANONIMO e CODIGO_INTERNACAO.
    - Procedimentos agregados por PRONTUARIO_ANONIMO (chave comum entre tabelas).
    """
    con = duckdb.connect(database=":memory:")
    evo, proc, cti = csv_paths

    # Aqui mantive sep=';' porque é o mais comum no SUS/Oracle;
    # se seus CSVs forem separados por vírgula, pode remover o sep.
    con.execute("""
        CREATE VIEW evolu AS
        SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);
    """, [evo])

    con.execute("""
        CREATE VIEW proced AS
        SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);
    """, [proc])

    con.execute("""
        CREATE VIEW cids AS
        SELECT * FROM read_csv_auto(?, sep=';', header=True, SAMPLE_SIZE=-1);
    """, [cti])

    # normaliza chaves para lower/trim
    con.execute("""
        CREATE VIEW evolu_n AS
        SELECT
          lower(trim(CAST(PRONTUARIO_ANONIMO AS VARCHAR))) AS prontuario_anonimo,
          lower(trim(CAST(CODIGO_INTERNACAO AS VARCHAR)))   AS codigo_internacao,
          *
        FROM evolu;
    """)

    con.execute("""
        CREATE VIEW cids_n AS
        SELECT
          lower(trim(CAST(PRONTUARIO_ANONIMO AS VARCHAR))) AS prontuario_anonimo,
          lower(trim(CAST(CODIGO_INTERNACAO AS VARCHAR)))   AS codigo_internacao,
          *
        FROM cids;
    """)

    # Procedimentos agregados POR PRONTUARIO_ANONIMO (chave comum que liga as tabelas)
    con.execute("""
        CREATE VIEW proc_agg AS
        SELECT
          lower(trim(CAST(PRONTUARIO_ANONIMO AS VARCHAR))) AS prontuario_anonimo,
          COUNT(DISTINCT CODIGO_PROCEDIMENTO)              AS n_proced,
          ANY_VALUE(CODIGO_PROCEDIMENTO)                   AS proc_prim,
          ANY_VALUE(PROCEDIMENTO)                          AS proc_nome_prim,
          MIN(COALESCE(DATA_CIRURGIA, DATA_INTERNACAO))    AS data_cirurgia_min,
          MAX(COALESCE(DATA_CIRURGIA, DATA_INTERNACAO))    AS data_cirurgia_max
        FROM proced
        GROUP BY 1;
    """)

    # Dataset final:
    # - EVOLU + CIDS: juntando por (prontuario_anonimo, codigo_internacao)
    # - PROCED: anexado por prontuario_anonimo
    con.execute("""
        CREATE VIEW dataset AS
        SELECT
          e.*,
          c.* EXCLUDE (PRONTUARIO_ANONIMO, CODIGO_INTERNACAO),
          p.*
        FROM evolu_n e
        LEFT JOIN cids_n  c USING (prontuario_anonimo, codigo_internacao)
        LEFT JOIN proc_agg p USING (prontuario_anonimo);
    """)

    return con

def df_from_duckdb(con, sql):
    return con.execute(sql).df()

# -------------------- enriquecimento com tabelas auxiliares --------------------
def enrich_with_aux_tables(df: pd.DataFrame,
                           cid_file=None,
                           sigtap_file=None,
                           geo_file=None) -> pd.DataFrame:
    """
    Enriquecimento opcional com:
    - CID-10 (capítulo / grupo / subcategoria)
    - Procedimentos SIGTAP (grupo / subgrupo / forma organização)
    - Tabela geográfica (UF / macro / região de saúde)

    Todos são opcionais; se o arquivo não for enviado, nada quebra.
    """
    df_enriched = df.copy()

    # -------- CID-10 --------
    if cid_file is not None:
        try:
            cid_df = pd.read_csv(cid_file, dtype=str)
            cid_df.columns = [c.lower() for c in cid_df.columns]

            # tenta identificar coluna de código CID (ex: 'cid', 'codigo_cid', etc.)
            cid_code_col = next(
                (c for c in cid_df.columns if "cid" in c and "descricao" not in c),
                None
            )
            if cid_code_col and "cid" in df_enriched.columns:
                df_enriched["cid"] = (
                    df_enriched["cid"].astype(str).str.strip().str.upper().str[:3]
                )
                cid_df[cid_code_col] = (
                    cid_df[cid_code_col].astype(str).str.strip().str.upper().str[:3]
                )

                # tenta pegar colunas de capítulo / grupo / subcategoria / descrição
                keep_cols = [cid_code_col]
                keep_cols += [
                    c for c in cid_df.columns
                    if any(k in c for k in ["capit", "grupo", "subcat", "subcategoria", "desc"])
                ]
                cid_small = cid_df[keep_cols].drop_duplicates(subset=[cid_code_col])

                df_enriched = df_enriched.merge(
                    cid_small,
                    how="left",
                    left_on="cid",
                    right_on=cid_code_col
                )
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com CID-10: {e}")

    # -------- Procedimentos (SIGTAP) --------
    if sigtap_file is not None:
        try:
            sig_df = pd.read_csv(sigtap_file, dtype=str)
            sig_df.columns = [c.lower() for c in sig_df.columns]

            # identifica código de procedimento no SIGTAP
            sig_code_col = next(
                (c for c in sig_df.columns
                 if "proced" in c and ("cod" in c or "codigo" in c)),
                None
            )

            # coluna de proc no dataset principal
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
                    c for c in sig_df.columns
                    if any(k in c for k in ["grupo", "subgrupo", "forma", "nome"])
                ]
                sig_small = sig_df[keep_cols].drop_duplicates(subset=[sig_code_col])

                df_enriched = df_enriched.merge(
                    sig_small,
                    how="left",
                    left_on=proc_col,
                    right_on=sig_code_col
                )
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com SIGTAP: {e}")

    # -------- Geografia (UF / Macro / Região de Saúde) --------
    if geo_file is not None:
        try:
            geo_df = pd.read_csv(geo_file, dtype=str)
            geo_df.columns = [c.lower() for c in geo_df.columns]

            if "cidade_moradia" in df_enriched.columns:
                df_enriched["cidade_moradia"] = (
                    df_enriched["cidade_moradia"]
                    .astype(str).str.upper().str.strip()
                )

                mun_col = next(
                    (c for c in geo_df.columns
                     if "municip" in c and ("nome" in c or "município" in c)),
                    None
                )
                if mun_col:
                    geo_df[mun_col] = (
                        geo_df[mun_col].astype(str).str.upper().str.strip()
                    )

                    df_enriched = df_enriched.merge(
                        geo_df,
                        how="left",
                        left_on="cidade_moradia",
                        right_on=mun_col
                    )
        except Exception as e:
            st.warning(f"Não foi possível enriquecer com regiões de saúde: {e}")

    return df_enriched

# -------------------- funções de base --------------------
def pacientes_unicos(df: pd.DataFrame) -> pd.DataFrame:
    if {"prontuario_anonimo","data_internacao"}.issubset(df.columns):
        return (
            df.sort_values(["prontuario_anonimo","data_internacao"])
              .groupby("prontuario_anonimo", as_index=False)
              .tail(1)
        )
    if "prontuario_anonimo" in df.columns:
        return df.drop_duplicates(subset=["prontuario_anonimo"])
    return df

def kpis(df_eventos: pd.DataFrame, df_pacientes: pd.DataFrame):
    pacientes = (
        df_pacientes["prontuario_anonimo"].nunique()
        if "prontuario_anonimo" in df_pacientes else np.nan
    )
    internacoes = (
        df_eventos["codigo_internacao"].nunique()
        if "codigo_internacao" in df_eventos else len(df_eventos)
    )
    tmi = (
        df_eventos["dias_permanencia"]
          .replace([np.inf,-np.inf], np.nan)
          .dropna().mean()
        if "dias_permanencia" in df_eventos else np.nan
    )

    mort_hosp = np.nan
    if {"data_internacao","data_alta"}.issubset(df_eventos.columns):
        e = df_eventos.copy()
        if "data_obito" in e.columns:
            e["obito_no_periodo"] = (
                (e["data_obito"].notna()) &
                (e["data_obito"] >= e["data_internacao"]) &
                (e["data_obito"] <= (e["data_alta"] - pd.Timedelta(days=1)))
            )
        elif "evolucao" in e.columns:
            e["obito_no_periodo"] = e["evolucao"].astype(str).str.contains(
                "ÓBITO", case=False, na=False
            )
        else:
            e["obito_no_periodo"] = False

        denom = (
            e["codigo_internacao"].nunique()
            if "codigo_internacao" in e else len(e)
        )
        numer = e.loc[e["obito_no_periodo"]]
        numer = (
            numer["codigo_internacao"].nunique()
            if "codigo_internacao" in e else len(numer)
        )
        mort_hosp = (numer/denom*100) if denom else np.nan

    return pacientes, internacoes, tmi, mort_hosp

def reinternacao_30d_pos_proced(df: pd.DataFrame):
    ok = {"prontuario_anonimo","codigo_internacao",
          "data_internacao","data_alta"}.issubset(df.columns)
    if not ok: return np.nan
    s = df.sort_values(
        ["prontuario_anonimo","data_internacao","data_alta"]
    ).copy()
    s["next_dt_internacao"] = (
        s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    )
    s["delta_proc"] = (
        s["next_dt_internacao"] - s["data_internacao"]
    ).dt.days
    s["delta_pos_alta"] = (
        s["next_dt_internacao"] - s["data_alta"]
    ).dt.days
    s["transfer"] = s["delta_pos_alta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[
        s["delta_proc"].between(0,30, inclusive="both") & (~s["transfer"])
    ]["codigo_internacao"].nunique()
    return (numer/base*100) if base else np.nan

def reinternacao_30d_pos_alta(df: pd.DataFrame):
    ok = {"prontuario_anonimo","codigo_internacao",
          "data_internacao","data_alta"}.issubset(df.columns)
    if not ok: return np.nan
    s = df.sort_values(
        ["prontuario_anonimo","data_internacao","data_alta"]
    ).copy()
    s["next_dt_internacao"] = (
        s.groupby("prontuario_anonimo")["data_internacao"].shift(-1)
    )
    s["delta"] = (s["next_dt_internacao"] - s["data_alta"]).dt.days
    s["transfer"] = s["delta"] <= 1
    base = s["codigo_internacao"].nunique()
    numer = s[
        s["delta"].between(0,30, inclusive="both") & (~s["transfer"])
    ]["codigo_internacao"].nunique()
    return (numer/base*100) if base else np.nan

# -------------------- filtros e indicadores --------------------
def build_filters(df: pd.DataFrame):
    st.sidebar.header("Filtros")

    # ano
    anos_col = (
        "ano_internacao" if "ano_internacao" in df.columns
        else ("ano" if "ano" in df.columns else None)
    )
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
    idade_sel = st.sidebar.slider(
        "Idade", min_value=0, max_value=max(idade_max,1),
        value=(idade_min, idade_max), step=1
    )

    # estado (se existir)
    estado_col = next(
        (c for c in df.columns if c.lower() in
         ["estado_residencia","uf_residencia","uf","estado","sigla_uf"]),
        None
    )
    estados_sel = []
    if estado_col:
        estados = sorted(df[estado_col].dropna().astype(str).unique().tolist())
        estados_sel = st.sidebar.multiselect(
            "Estado de residência", estados, default=estados
        )

    # região de saúde (se existir)
    regiao_col = next(
        (c for c in df.columns
         if "regiao" in c.lower() and "saude" in c.lower()),
        None
    )
    regioes_sel = []
    if regiao_col:
        regioes = sorted(df[regiao_col].dropna().astype(str).unique().tolist())
        regioes_sel = st.sidebar.multiselect(
            "Região de saúde", regioes, default=regioes
        )

    # município de residência (lista baseada em pacientes únicos)
    df_pac_ref = pacientes_unicos(df)
    cidade_col = "cidade_moradia" if "cidade_moradia" in df_pac_ref.columns else None
    cidades_sel = []
    if cidade_col:
        cidade_vals = sorted(
            df_pac_ref[cidade_col].dropna().astype(str).unique().tolist()
        )
        default_cidades = (
            cidade_vals if len(cidade_vals) <= 25 else cidade_vals[:25]
        )
        cidades_sel = st.sidebar.multiselect(
            "Município de residência", cidade_vals, default=default_cidades
        )

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
        df = df[
            (df["idade"] >= f["idade"][0]) &
            (df["idade"] <= f["idade"][1])
        ]

    # estado
    estado_col = next(
        (c for c in df.columns if c.lower() in
         ["estado_residencia","uf_residencia","uf","estado","sigla_uf"]),
        None
    )
    if estado_col and f["estado"]:
        df = df[df[estado_col].isin(f["estado"])]

    # região de saúde
    regiao_col = next(
        (c for c in df.columns
         if "regiao" in c.lower() and "saude" in c.lower()),
        None
    )
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
        st.markdown("**Filtros ativos:** " + " | ".join(partes))
    else:
        st.markdown("**Filtros ativos:** nenhum filtro aplicado.")

# -------------------- UI: carregamento --------------------
st.title("Perfil dos Pacientes")

tab_parquet, tab_csv = st.tabs(["Parquet único (recomendado)", "3 CSVs grandes (DuckDB)"])

df = None
with tab_parquet:
    file_parquet = st.file_uploader(
        "Carregue o Parquet único (dataset_unico_2019_2025.parquet)",
        type=["parquet"], key="pq"
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
        p_evo = os.path.join(tmpdir, "evo.csv");  open(p_evo,  "wb").write(evo.getbuffer())
        p_proc = os.path.join(tmpdir, "proc.csv"); open(p_proc, "wb").write(proc.getbuffer())
        p_cti = os.path.join(tmpdir, "cti.csv");  open(p_cti,  "wb").write(cti.getbuffer())
        con = load_duckdb((p_evo, p_proc, p_cti))
        df = df_from_duckdb(con, "SELECT * FROM dataset")
        df = _post_load(df)

if df is None or df.empty:
    st.info("Carregue um Parquet único **ou** os 3 CSVs.")
    st.stop()

# -------- uploads das tabelas auxiliares --------
with st.expander
