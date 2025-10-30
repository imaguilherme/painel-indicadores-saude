
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from unidecode import unidecode

DATE_COLS = [
    "DATA_INTERNACAO","DATA_ALTA","DATA_OBITO",
    "DT_ENTRADA_CTI","DT_SAIDA_CTI","DTHR_ADMISSAO","DTHR_ALTA","DTHR_OBITO"
]

def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True, utc=False)
    return df

def read_csv_smart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df = _to_datetime(df)
    # Try to coerce numeric-like strings with thousands separators
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].str.replace(".","", regex=False).str.replace(",","", regex=False).str.match(r"^-?\d+$", na=False).mean() > 0.8:
            df[c] = pd.to_numeric(df[c].str.replace(".","", regex=False).str.replace(",","", regex=False), errors="ignore")
    return df

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "SEXO": "SEXO",
        "ETNIA": "RACA_COR",
        "CIDADE_MORADIA": "MUNICIPIO_RESIDENCIA",
        "ESTADO_CIVIL": "ESTADO_CIVIL",
        "ESCOLARIDADE": "ESCOLARIDADE",
        "CODIGO_INTERNACAO":"CODIGO_INTERNACAO",
        "DATA_INTERNACAO":"DATA_INTERNACAO",
        "DATA_ALTA":"DATA_ALTA",
        "DATA_OBITO":"DATA_OBITO",
        "UNIDADE_ADMISSAO":"UNIDADE_ADMISSAO",
        "UNIDADE_SAIDA_CTI":"UNIDADE_SAIDA_CTI",
        "CODIGO_PROCEDIMENTO": "COD_PROCEDIMENTO",
        "PROCEDIMENTO": "PROCEDIMENTO",
        "NATUREZA_AGEND":"CARATER_ATENDIMENTO",
        "TIPO_EVOLUCAO":"TIPO_EVOLUCAO",
        "TO_CHAR(SUBSTR(IEV.DESCRICAO,1,4000))":"EVOLUCAO_TEXTO",
        "IDADE":"IDADE",
        "PRONTUARIO_ANONIMO":"ID_PACIENTE"
    }
    # Keep only columns that exist
    keep = {k:v for k,v in mapping.items() if k in df.columns}
    df = df.rename(columns=keep)
    return df

def compute_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    if "DATA_INTERNACAO" in df.columns:
        df["LOS_dias"] = (df["DATA_ALTA"].fillna(df["DATA_OBITO"]) - df["DATA_INTERNACAO"]).dt.days
    if "DT_ENTRADA_CTI" in df.columns:
        df["LOS_UTI_dias"] = (df["DT_SAIDA_CTI"] - df["DT_ENTRADA_CTI"]).dt.days
    return df

def age_bins(df: pd.DataFrame, col="IDADE"):
    bins = [0,1,4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,120]
    labels = ["<1a","1-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
              "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
    if col in df.columns:
        return pd.cut(df[col], bins=bins, labels=labels, right=True, include_lowest=True)
    return pd.Series(pd.Categorical([]), index=df.index)

def year_month(s):
    if pd.isna(s): return None
    return pd.Timestamp(s).strftime("%Y-%m")

def percent(n, d):
    return (n / d * 100) if d else 0.0

def safe_len(x): 
    try: return len(x)
    except: return 0

def derive_common_fields(pacientes: pd.DataFrame) -> pd.DataFrame:
    df = pacientes.copy()
    df = standardize_cols(df)
    df = compute_length_of_stay(df)
    if "IDADE" not in df.columns and "DATA_NASC" in df.columns and "DATA_INTERNACAO" in df.columns:
        df["IDADE"] = (df["DATA_INTERNACAO"] - pd.to_datetime(df["DATA_NASC"], errors="coerce")).dt.days // 365.25
    df["FAIXA_ETARIA"] = age_bins(df, "IDADE")
    if "DATA_INTERNACAO" in df.columns:
        df["ANO"] = df["DATA_INTERNACAO"].dt.year
        df["YM"] = df["DATA_INTERNACAO"].dt.to_period("M").astype(str)
    return df
