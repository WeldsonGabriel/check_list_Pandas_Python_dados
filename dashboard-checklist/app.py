import streamlit as st
import pandas as pd

from core.checklist_rules import Thresholds, process_transactions_csv
from core.baas_rules import read_baas_file, build_baas_index, extract_accounts_list

st.set_page_config(page_title="Alertas - Checklist", layout="wide")

st.title("Alertas (CSV → Checklist + Cards)")
st.caption("Envie o CSV de transações e informe as empresas. Opcionalmente, envie o CSV BaaS para enriquecer com bloqueio cautelar.")

# ---------------- Sidebar / Inputs
st.sidebar.header("Entrada")

companies = st.sidebar.text_area(
    "Empresas (vírgula ou uma por linha)",
    height=160,
    placeholder="Ex: Treeal\nEmpresa X\nEmpresa Y"
)

tx_file = st.sidebar.file_uploader("CSV Transações", type=["csv"], accept_multiple_files=False)
baas_file = st.sidebar.file_uploader("CSV BaaS (opcional)", type=["csv"], accept_multiple_files=False)

st.sidebar.subheader("Parâmetros")
queda_critica = st.sidebar.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
aumento_relevante = st.sidebar.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
investigar_abs = st.sidebar.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")

run = st.sidebar.button("Processar", type="primary", use_container_width=True)

# ---------------- Helpers
def _normalize_companies_input(text: str) -> str:
    raw = []
    for part in (text or "").replace("\n", ",").split(","):
        p = part.strip()
        if p:
            raw.append(p)
    # remove duplicados case-insensitive
    seen = set()
    out = []
    for c in raw:
        k = c.upper()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return ",".join(out)

def paginate_df(df: pd.DataFrame, page_size: int, key_prefix: str):
    if df is None or df.empty:
        st.info("Sem dados para exibir.")
        return

    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        page = st.number_input(
            "Página",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key=f"{key_prefix}_page"
        )
    with col2:
        st.caption(f"{total} linhas — {pages} páginas — {page_size}/página")
    with col3:
        st.empty()

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    st.dataframe(df.iloc[start:end], use_container_width=True, hide_index=True)

# ---------------- State
if "result" not in st.session_state:
    st.session_state.result = None

if run:
    if not companies.strip():
        st.error("Informe pelo menos 1 empresa.")
    elif not tx_file:
        st.error("Envie o CSV de transações.")
    else:
        thresholds = Thresholds(
            queda_critica=float(queda_critica),
            aumento_relevante=float(aumento_relevante),
            investigar_abs=float(investigar_abs),
        )

        companies_param = _normalize_companies_input(companies)

        with st.spinner("Processando transações..."):
            day_ref, checklist_rows, cards_rows = process_transactions_csv(
                file_bytes=tx_file.getvalue(),
                companies_param=companies_param,
                thresholds=thresholds,
            )

        df_check = pd.DataFrame(checklist_rows or [])
        df_cards = pd.DataFrame(cards_rows or [])

        # BaaS opcional
        df_baas = pd.DataFrame()
        idx_baas = {}
        if baas_file:
            with st.spinner("Lendo CSV BaaS..."):
                df_baas = read_baas_file(baas_file.getvalue())
                idx_baas = build_baas_index(df_baas)

        st.session_state.result = {
            "day_ref": day_ref,
            "df_check": df_check,
            "df_cards": df_cards,
            "df_baas": df_baas,
            "idx_baas": idx_baas,
        }

# ---------------- Render (se tiver resultado)
res = st.session_state.result

# KPIs (topo)
k1, k2, k3, k4, k5 = st.columns(5)
day_ref = res["day_ref"] if res else "—"
df_check = res["df_check"] if res else pd.DataFrame()
df_cards = res["df_cards"] if res else pd.DataFrame()
idx_baas = res["idx_baas"] if res else {}

alerts_count = int((df_check["status"] != "Normal").sum()) if (not df_check.empty and "status" in df_check) else 0
critical_count = int((df_check["status"] == "Escalar (queda)").sum()) if (not df_check.empty and "status" in df_check) else 0
companies_count = int(len(df_check)) if not df_check.empty else 0

# contas com bloqueio (contando por conta do índice BaaS)
blocked_accounts_count = sum(1 for v in (idx_baas or {}).values() if v.get("blocked"))

k1.metric("Dia de referência", day_ref)
k2.metric("Empresas no checklist", companies_count)
k3.metric("Alertas", alerts_count)
k4.metric("Críticos", critical_count)
k5.metric("Contas com bloqueio", blocked_accounts_count)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Checklist Empresas", "BaaS (Bloqueio)", "Cards"])

with tab1:
    st.subheader("Visão Geral")
    st.caption("Aqui entram os gráficos Plotly (status, ranking, distribuições).")
    st.info("Quando você colar seu service, eu já coloco os gráficos automaticamente com base no df_check.")

with tab2:
    st.subheader("Checklist Empresas")
    if df_check is None or df_check.empty:
        st.info("Processa um CSV de transações para gerar o checklist.")
    else:
        # filtros
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            q = st.text_input("Buscar empresa", "")
        with c2:
            status_opt = st.selectbox("Status", ["Todos", "Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"])
        with c3:
            only_alerts = st.checkbox("Somente alertas", value=False)

        df = df_check.copy()
        if q:
            df = df[df["company_name"].astype(str).str.upper().str.contains(q.upper(), na=False)]
        if status_opt != "Todos":
            df = df[df["status"].astype(str) == status_opt]
        if only_alerts and "status" in df:
            df = df[df["status"].astype(str) != "Normal"]

        paginate_df(df, page_size=15, key_prefix="check")

with tab3:
    st.subheader("Checklist Bloqueio Cautelar (BaaS)")
    df_baas = res["df_baas"] if res else pd.DataFrame()
    if df_baas is None or df_baas.empty:
        st.info("Envie o CSV BaaS (opcional) e processe para ver esta tabela.")
    else:
        # tabela simples (paginada)
        show = df_baas[["posicao", "conta", "agencia", "nome", "saldo_bloqueado"]].copy()
        show["status"] = show["saldo_bloqueado"].apply(lambda x: "BLOQUEIO" if float(x or 0) > 0 else "SEM BLOQUEIO")

        paginate_df(show, page_size=15, key_prefix="baas")

with tab4:
    st.subheader("Cards")
    if df_cards is None or df_cards.empty:
        st.info("Sem cards (ou ainda não processado).")
    else:
        # mostra cards como tabela por enquanto
        st.dataframe(df_cards, use_container_width=True, hide_index=True)
