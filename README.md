# 🩺 Painel de Indicadores de Saúde – Streamlit

**Como usar**

1. Crie um ambiente (ou use `pipx runpip streamlit`) e instale dependências:  
   `pip install -r requirements.txt`
2. Inicie: `streamlit run app.py`
3. Na barra lateral, envie seus CSVs exportados do banco (Internações, Procedimentos, UTI).

**Mapeamento de colunas esperadas (flexível)**

- Obrigatórias/Ideais na base de internações (nomes aproximados são reconhecidos):
  - `PRONTUARIO_ANONIMO` → `ID_PACIENTE`
  - `DATA_INTERNACAO`, `DATA_ALTA`, `DATA_OBITO`
  - `SEXO`, `ETNIA` (→ `RACA_COR`), `IDADE` (ou `DATA_NASC`)
  - `NATUREZA_AGEND` (→ `CARATER_ATENDIMENTO`), `UNIDADE_ADMISSAO`
  - `CID10_PRINCIPAL` (opcional)

- Procedimentos:
  - `CODIGO_PROCEDIMENTO`, `PROCEDIMENTO`, `DATA_CIRURGIA` (ou data equivalente)

- UTI/CTI:
  - `DT_ENTRADA_CTI`, `DT_SAIDA_CTI`, unidade

O app tenta converter datas automaticamente e normaliza alguns nomes.

**Customização rápida**

- Cores: edite `.streamlit/config.toml`
- Novos gráficos: crie páginas em `pages/` (são carregadas automaticamente pelo Streamlit).

**Privacidade**: use `PRONTUARIO_ANONIMO`/IDs não identificáveis.