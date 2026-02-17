# app.py
from __future__ import annotations

import re
import math
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

from core import (
    Thresholds,
    AlertConfig,
    STATUS_ORDER,
    STATUS_DOT,
    LOCK_ICON,
    UNLOCK_ICON,
    fmt_int_pt,
    fmt_money_pt,
    clamp_int,
    split_companies_input,
    assistant_explain_row,
)
from services import (
    parse_transactions_csv,
    parse_baas_csv,
    build_checklist_weeks,
    enrich_baas,
    build_daily_matrix,
    build_alerts,
    daily_totals_from_facts,
    top_companies_from_facts,
    zero_and_down_counts_by_day,
)


# =========================
# Page config
# =========================
st.set_page_config(page_title="Checklist Semanas + BaaS", layout="wide")


# =========================
# CSS
# =========================
st.markdown(
    """
<style>
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 14px;
}
.mini-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}
.mini-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px 14px;
}
.mini-title { font-size: 13px; opacity: .75; margin-bottom: 6px; }
.mini-value { font-size: 22px; font-weight: 800; line-height: 1.1; }

.status-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.status-box {
  border-radius: 14px;
  padding: 14px;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.status-num { font-size: 28px; font-weight: 800; line-height: 1.1; }
.status-lab { margin-top: 6px; font-size: 13px; opacity: .85; }

.sb-red { border-color: rgba(231,76,60,0.45); background: linear-gradient(180deg, rgba(231,76,60,0.12), rgba(255,255,255,0.02)); }
.sb-yellow { border-color: rgba(241,196,15,0.45); background: linear-gradient(180deg, rgba(241,196,15,0.12), rgba(255,255,255,0.02)); }
.sb-blue { border-color: rgba(52,152,219,0.45); background: linear-gradient(180deg, rgba(52,152,219,0.12), rgba(255,255,255,0.02)); }
.sb-green { border-color: rgba(46,204,113,0.45); background: linear-gradient(180deg, rgba(46,204,113,0.10), rgba(255,255,255,0.02)); }

.badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.badge {
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  opacity: .95;
}
.badge strong { font-weight: 800; }
.b-red { border-color: rgba(231,76,60,0.45); background: rgba(231,76,60,0.10); }
.b-orange { border-color: rgba(230,126,34,0.50); background: rgba(230,126,34,0.12); }
.b-yellow { border-color: rgba(241,196,15,0.45); background: rgba(241,196,15,0.12); }
.b-green { border-color: rgba(46,204,113,0.45); background: rgba(46,204,113,0.10); }
.b-gray { border-color: rgba(170,180,200,0.35); background: rgba(170,180,200,0.08); }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# UI helpers
# =========================
def render_status_boxes(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Sem dados.")
        return
    counts = {s: int((df["status"] == s).sum()) for s in STATUS_ORDER}
    st.markdown(
        f"""
<div class="status-row">
  <div class="status-box sb-red">
    <div class="status-num">{counts["Alerta (queda/ zerada)"]}</div>
    <div class="status-lab">Alerta</div>
  </div>
  <div class="status-box sb-yellow">
    <div class="status-num">{counts["Investigar"]}</div>
    <div class="status-lab">Investigar</div>
  </div>
  <div class="status-box sb-blue">
    <div class="status-num">{counts["Gerenciar (aumento)"]}</div>
    <div class="status-lab">Gerenciar</div>
  </div>
  <div class="status-box sb-green">
    <div class="status-num">{counts["Normal"]}</div>
    <div class="status-lab">Normal</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_badge(label: str, value: str, tone: str):
    cls = {
        "red": "b-red",
        "orange": "b-orange",
        "yellow": "b-yellow",
        "green": "b-green",
        "gray": "b-gray",
    }.get(tone, "b-gray")
    st.markdown(
        f"""<span class="badge {cls}">{label}: <strong>{value}</strong></span>""",
        unsafe_allow_html=True,
    )


def render_ground_panel(
    title: str,
    w1: int, w2: int, w3: int, w4: int,
    credit: int, debit: int, total_4w: int,
    var: float,
    height: int,
    labels: dict,
):
    if go is None:
        st.warning("plotly não está disponível (instale plotly).")
        return

    max_ref = max(total_4w, w1, w2, w3, w4, 1)
    gauge_max = max_ref * 1.25

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(total_4w),
            number={"valueformat": ".0f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "rgba(255,255,255,0.85)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, gauge_max], "color": "rgba(255,255,255,0.08)"}],
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    var_pct = int(math.ceil(var * 100.0)) if np.isfinite(var) else 0
    st.markdown(
        f"""
<div class="mini-grid">
  <div class="mini-card">
    <div class="mini-title">Semana 1 ({labels.get("w1","")})</div>
    <div class="mini-value">{fmt_int_pt(w1)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 2 ({labels.get("w2","")})</div>
    <div class="mini-value">{fmt_int_pt(w2)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Semana 3 ({labels.get("w3","")})</div>
    <div class="mini-value">{fmt_int_pt(w3)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 4 ({labels.get("w4","")})</div>
    <div class="mini-value">{fmt_int_pt(w4)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Crédito (Qtd)</div>
    <div class="mini-value">{fmt_int_pt(credit)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Débito (Qtd)</div>
    <div class="mini-value">{fmt_int_pt(debit)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Variação (S4 vs média S1–S3)</div>
    <div class="mini-value">{var_pct}%</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Total 4 Semanas</div>
    <div class="mini-value">{fmt_int_pt(total_4w)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# App
# =========================
st.title("Checklist (Semanas 1–4) + Saldos/Bloqueio (BaaS)")

with st.sidebar:
    st.header("Parâmetros (Status semanal)")
    queda_critica = st.number_input("Queda crítica (S4 vs média S1–S3)", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante (não essencial)", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Faixa normal (±)", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.header("Volume muito baixo")
    low_volume_threshold = st.number_input("Limite (Total 4 semanas)", value=5, step=1, min_value=0)

    st.divider()
    st.header("Ground (tamanho)")
    ground_height = st.slider("Altura do Ground", min_value=220, max_value=480, value=340, step=20)

    st.divider()
    st.header("Paginação")
    page_size = st.selectbox("Checklist (por página)", [15, 20, 30, 50], index=0)

    st.divider()
    st.header("Alertas (farol / diário)")

    st.caption("Streak zerado (dias)")
    zero_y = st.number_input("Zerado ≥ (amarelo)", value=5, min_value=1, step=1)
    zero_o = st.number_input("Zerado ≥ (laranja)", value=10, min_value=2, step=1)
    zero_r = st.number_input("Zerado ≥ (vermelho)", value=15, min_value=3, step=1)

    st.caption("Streak queda (dias)")
    down_y = st.number_input("Queda ≥ (amarelo)", value=5, min_value=1, step=1)
    down_o = st.number_input("Queda ≥ (laranja)", value=10, min_value=2, step=1)
    down_r = st.number_input("Queda ≥ (vermelho)", value=15, min_value=3, step=1)

    st.caption("Bloqueio (R$)")
    blk_y = st.number_input("Bloqueio ≥ (amarelo)", value=10_000.0, step=1000.0, format="%.0f")
    blk_o = st.number_input("Bloqueio ≥ (laranja)", value=50_000.0, step=1000.0, format="%.0f")
    blk_r = st.number_input("Bloqueio ≥ (vermelho)", value=100_000.0, step=1000.0, format="%.0f")

    st.caption("Frequência (%, menor = pior)")
    fr_y = st.number_input("Freq ≤ (amarelo)", value=80.0, step=1.0, format="%.0f")
    fr_o = st.number_input("Freq ≤ (laranja)", value=60.0, step=1.0, format="%.0f")
    fr_r = st.number_input("Freq ≤ (vermelho)", value=40.0, step=1.0, format="%.0f")

    st.caption("Baseline / queda")
    baseline_days = st.number_input("Janela baseline (dias)", value=7, min_value=2, step=1)
    drop_ratio = st.slider("Queda = < X% da baseline", min_value=0.30, max_value=0.95, value=0.70, step=0.05)

thresholds = Thresholds(
    queda_critica=float(queda_critica),
    aumento_relevante=float(aumento_relevante),
    investigar_abs=float(investigar_abs),
)

alert_cfg = AlertConfig(
    zero_yellow=int(zero_y),
    zero_orange=int(zero_o),
    zero_red=int(zero_r),
    down_yellow=int(down_y),
    down_orange=int(down_o),
    down_red=int(down_r),
    block_yellow=float(blk_y),
    block_orange=float(blk_o),
    block_red=float(blk_r),
    freq_yellow=float(fr_y),
    freq_orange=float(fr_o),
    freq_red=float(fr_r),
    baseline_days=int(baseline_days),
    drop_ratio=float(drop_ratio),
)

tab_main, tab_alerts, tab_daily, tab_graphs = st.tabs(["Principal", "Alertas", "Análise (Diário)", "Gráficos (Diário)"])


# =========================
# PRINCIPAL
# =========================
with tab_main:
    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    with col_left:
        companies_text = st.text_area(
            "Empresas (obrigatório) — vírgula ou uma por linha",
            height=140,
            placeholder="Ex:\nDOM DIGITAL\nPIX NA HORA LTDA",
            key="companies_text",
        )

    with col_right:
        trans_file = st.file_uploader("CSV Transações (obrigatório)", type=["csv"], key="trans_csv")
        baas_file = st.file_uploader("CSV BaaS (opcional) — com Saldo + Saldo Bloqueado", type=["csv"], key="baas_csv")

    process = st.button("Processar", type="primary", use_container_width=True)

    if process:
        companies_keys = split_companies_input(companies_text)
        if not companies_keys:
            st.error("Você precisa informar pelo menos 1 empresa no input antes de processar.")
            st.stop()
        if trans_file is None:
            st.error("Você precisa subir o CSV de transações antes de processar.")
            st.stop()

        with st.spinner("Processando..."):
            facts = parse_transactions_csv(trans_file)
            if facts.empty:
                st.error("CSV de transações ficou vazio após leitura/normalização.")
                st.stop()

            day_ref, df_checklist, week_labels = build_checklist_weeks(
                facts=facts,
                companies_keys=companies_keys,
                thresholds=thresholds,
                low_volume_threshold=int(low_volume_threshold),
            )

            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_final, kpi_blocked_accounts, kpi_blocked_sum = enrich_baas(df_checklist, df_baas)

            start_day, end_day, daily_table = build_daily_matrix(facts=facts, companies_keys=companies_keys)
            alerts_df = build_alerts(
                daily_table=daily_table,
                start_day=start_day,
                end_day=end_day,
                df_final=df_final,
                cfg=alert_cfg,
            )

            st.session_state["facts"] = facts
            st.session_state["companies_keys"] = companies_keys
            st.session_state["day_ref"] = day_ref
            st.session_state["week_labels"] = week_labels
            st.session_state["df_final"] = df_final
            st.session_state["kpi_blocked_accounts"] = int(kpi_blocked_accounts)
            st.session_state["kpi_blocked_sum"] = float(kpi_blocked_sum)
            st.session_state["page"] = 1

            st.session_state["daily_start"] = start_day
            st.session_state["daily_end"] = end_day
            st.session_state["daily_table"] = daily_table
            st.session_state["alerts_df"] = alerts_df

    if "df_final" not in st.session_state:
        st.info("Preencha empresas + suba CSV + clique em **Processar**.")
        st.stop()

    df_final = st.session_state.get("df_final", pd.DataFrame())
    week_labels = st.session_state.get("week_labels", {})
    day_ref = st.session_state.get("day_ref", "—")
    kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))
    kpi_blocked_sum = float(st.session_state.get("kpi_blocked_sum", 0.0))

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        st.metric("Dia de referência", day_ref)
    with k2:
        st.metric("Empresas filtradas", fmt_int_pt(int(df_final["company_key"].nunique()) if not df_final.empty else 0))
    with k3:
        st.metric("Alertas (semanal)", fmt_int_pt(int((df_final["status"] != "Normal").sum()) if not df_final.empty else 0))
    with k4:
        st.metric("Alertas críticos", fmt_int_pt(int((df_final["status"] == "Alerta (queda/ zerada)").sum()) if not df_final.empty else 0))
    with k5:
        st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))
    with k6:
        st.metric("Soma bloqueada (R$)", fmt_money_pt(kpi_blocked_sum))

    st.divider()
    st.subheader("Visão Geral (semanal)")
    render_status_boxes(df_final)
    st.divider()

    if px is None or go is None:
        st.warning("Instale plotly para ver o Ground.")
    elif df_final.empty:
        st.info("Sem dados.")
    else:
        left, _right = st.columns([1.1, 1.0], gap="large")

        df_final["select_key"] = df_final["company_name"].astype(str) + " | " + df_final["account_id"].astype(str)
        options = ["TOTAL GERAL"] + sorted(df_final["select_key"].unique().tolist())

        with left:
            st.caption("Ground (Total geral ou por Conta)")
            selected = st.selectbox("Selecionar", options, index=0, key="ground_select")

            if selected == "TOTAL GERAL":
                scope = df_final.copy()
                title = "TOTAL 4 SEMANAS (GERAL)"
                w1 = int(scope["week1"].sum())
                w2 = int(scope["week2"].sum())
                w3 = int(scope["week3"].sum())
                w4 = int(scope["week4"].sum())
                credit = int(scope["credit"].sum())
                debit = int(scope["debit"].sum())
                total_4w = int(scope["total_4w"].sum())
                avg_prev = (w1 + w2 + w3) / 3.0 if (w1 + w2 + w3) > 0 else 0.0
                var = (w4 - avg_prev) / avg_prev if avg_prev > 0 else 0.0
            else:
                scope = df_final[df_final["select_key"] == selected].copy()
                r = scope.iloc[0]
                title = selected
                w1, w2, w3, w4 = int(r["week1"]), int(r["week2"]), int(r["week3"]), int(r["week4"])
                credit, debit = int(r["credit"]), int(r["debit"])
                total_4w = int(r["total_4w"])
                var = float(r.get("var_week4_vs_prev", 0.0))

            render_ground_panel(
                title=title,
                w1=w1, w2=w2, w3=w3, w4=w4,
                credit=credit, debit=debit, total_4w=total_4w,
                var=var,
                height=int(ground_height),
                labels=week_labels,
            )

    st.divider()
    st.subheader("Checklist (por conta) — semanal")

    if df_final.empty:
        st.info("Sem dados para mostrar.")
        st.stop()

    f1, f2, f3 = st.columns([1.2, 1.0, 0.8], gap="small")
    with f1:
        q_empresa = st.text_input("Buscar empresa", value="", key="q_empresa")
    with f2:
        q_conta = st.text_input("Buscar conta", value="", key="q_conta")
    with f3:
        status_filter = st.selectbox("Status", ["Todos"] + STATUS_ORDER, index=0, key="status_filter")

    view = df_final.copy()
    if q_empresa.strip():
        q = q_empresa.strip().lower()
        view = view[view["company_name"].astype(str).str.lower().str.contains(q, na=False)]

    if q_conta.strip():
        q = re.sub(r"\D+", "", q_conta.strip())
        if q:
            view = view[view["account_id"].astype(str).str.contains(q, na=False)]

    if status_filter != "Todos":
        view = view[view["status"] == status_filter].copy()

    view["Status"] = view["status"].map(lambda s: STATUS_DOT.get(str(s), "⚪")) + " " + view["status"].astype(str) + " " + view["lock_icon"].astype(str)

    show = pd.DataFrame({
        "Status": view["Status"],
        "Conta": view["account_id"].astype(str),
        "Empresa": view["company_name"].astype(str),
        "Semana 1": view["week1"].astype(int).map(fmt_int_pt),
        "Semana 2": view["week2"].astype(int).map(fmt_int_pt),
        "Semana 3": view["week3"].astype(int).map(fmt_int_pt),
        "Semana 4": view["week4"].astype(int).map(fmt_int_pt),
        "Crédito (Qtd)": view["credit"].astype(int).map(fmt_int_pt),
        "Débito (Qtd)": view["debit"].astype(int).map(fmt_int_pt),
        "Total 4 Semanas": view["total_4w"].astype(int).map(fmt_int_pt),
        "Saldo (R$)": view["saldo_total"].map(fmt_money_pt) if "saldo_total" in view.columns else "0,00",
        "Saldo Bloqueado (R$)": view["saldo_bloqueado_total"].map(fmt_money_pt),
        "Motivo": view["motivo"].astype(str),
    })

    st.caption(
        f"Semana 1: {week_labels.get('w1','')} | Semana 2: {week_labels.get('w2','')} | "
        f"Semana 3: {week_labels.get('w3','')} | Semana 4: {week_labels.get('w4','')}"
    )

    total_rows = int(len(show))
    pages = max(1, int(math.ceil(total_rows / max(1, int(page_size)))))

    st.session_state.setdefault("page", 1)
    st.session_state["page"] = clamp_int(int(st.session_state["page"]), 1, pages)

    st.number_input(
        f"Página Checklist (1–{pages})",
        min_value=1,
        max_value=pages,
        value=int(st.session_state["page"]),
        step=1,
        key="page",
    )

    page_now = clamp_int(int(st.session_state["page"]), 1, pages)
    start = (page_now - 1) * int(page_size)
    end = start + int(page_size)
    st.dataframe(show.iloc[start:end].copy(), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Cards (para colar no Discord) — semanal")

    alerts_week = df_final[df_final["status"] != "Normal"].copy()
    if alerts_week.empty:
        st.caption("Nenhum alerta (status != Normal).")
    else:
        for _, r in alerts_week.iterrows():
            title = f"{r['status']} {r['lock_icon']} — {r['company_name']} | Conta {r['account_id']}"
            with st.expander(title, expanded=False):
                msg = (
                    f"ALERTA (SEMANAL): {r['status']}\n"
                    f"Empresa: {r['company_name']}\n"
                    f"Conta: {r['account_id']}\n"
                    f"Data (day_ref): {r['day_ref']}\n"
                    f"Motivo: {r['motivo']}\n"
                    f"Semanas: S1={r['week1']} | S2={r['week2']} | S3={r['week3']} | S4={r['week4']}\n"
                    f"Crédito(Qtd): {r['credit']} | Débito(Qtd): {r['debit']}\n"
                    f"Total 4 semanas: {r['total_4w']}\n"
                    f"Saldo: R$ {fmt_money_pt(r.get('saldo_total', 0.0))}\n"
                    f"Bloqueio: R$ {fmt_money_pt(r.get('saldo_bloqueado_total', 0.0))} {r['lock_icon']}"
                )
                st.code(msg, language="text")


# =========================
# ALERTAS (HÁBITOS / ZABBIX FINANCEIRO)
# =========================
with tab_alerts:
    if "alerts_df" not in st.session_state or "daily_table" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    alerts_df = st.session_state["alerts_df"]
    daily_table = st.session_state["daily_table"]

    st.subheader("Hábitos — Monitoramento Financeiro (estilo Zabbix)")

    # KPIs (inclui freq e média, como pedido)
    total_transacoes = int(alerts_df["Total_Periodo"].sum()) if not alerts_df.empty else 0
    total_bloqueado = float(alerts_df["Saldo_Bloqueado"].sum()) if not alerts_df.empty else 0.0
    contas_zeradas = int((alerts_df["Streak_zero"] > 0).sum()) if not alerts_df.empty else 0
    contas_queda = int((alerts_df["Streak_down"] > 0).sum()) if not alerts_df.empty else 0
    freq_media = float(alerts_df["Freq_pct"].mean()) if not alerts_df.empty else 0.0

    day_cols = [c for c in daily_table.columns if isinstance(c, pd.Timestamp)]
    media_diaria = int(daily_table[day_cols].sum(axis=0).mean()) if day_cols else 0

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1: st.metric("Total Transações", fmt_int_pt(total_transacoes))
    with g2: st.metric("Bloqueado (R$)", fmt_money_pt(total_bloqueado))
    with g3: st.metric("Contas Zeradas", fmt_int_pt(contas_zeradas))
    with g4: st.metric("Contas em Queda", fmt_int_pt(contas_queda))
    with g5: st.metric("Frequência Média", f"{freq_media:.0f}%")
    with g6: st.metric("Média Diária", fmt_int_pt(media_diaria))

    st.divider()

    empresas = ["TOTAL GERAL"] + sorted(alerts_df["Empresa"].unique().tolist()) if not alerts_df.empty else ["TOTAL GERAL"]

    f1, f2, f3 = st.columns([1.2, 1.0, 0.8])
    with f1:
        sel_emp = st.selectbox("Empresa", empresas, 0)
    with f2:
        busca_nome = st.text_input("Buscar empresa")
    with f3:
        busca_conta = st.text_input("Buscar conta")

    view = alerts_df.copy()
    if not view.empty:
        if sel_emp != "TOTAL GERAL":
            view = view[view["Empresa"] == sel_emp]
        if busca_nome:
            view = view[view["Empresa"].str.lower().str.contains(busca_nome.lower())]
        if busca_conta:
            view = view[view["Conta"].astype(str).str.contains(busca_conta)]
        view = view.sort_values("score", ascending=False)

    def bg_color(tone: str) -> str:
        return {
            "red": "rgba(231,76,60,0.18)",
            "orange": "rgba(230,126,34,0.18)",
            "yellow": "rgba(241,196,15,0.18)",
            "green": "rgba(46,204,113,0.18)",
            "gray": "rgba(170,180,200,0.10)",
        }.get(tone, "rgba(170,180,200,0.08)")

    cols = st.columns(4)

    if view.empty:
        st.info("Sem alertas para o escopo atual.")
    else:
        tone_rank = {"red": 4, "orange": 3, "yellow": 2, "gray": 1, "green": 0}
        for i, (_, r) in enumerate(view.iterrows()):
            with cols[i % 4]:
                tone = max(
                    [r["tone_zero"], r["tone_down"], r["tone_block"], r["tone_freq"]],
                    key=lambda x: tone_rank.get(str(x), 0)
                )
                st.markdown(
                    f"""
                    <div style="
                        background:{bg_color(str(tone))};
                        padding:14px;
                        border-radius:14px;
                        border:1px solid rgba(255,255,255,0.15);
                        margin-bottom:12px;
                    ">
                    <strong>{r['Empresa']}</strong><br>
                    Conta {r['Conta']}<br><br>

                    Total: {fmt_int_pt(int(r['Total_Periodo']))}<br>
                    Último dia: {fmt_int_pt(int(r['Ultimo_Dia']))}<br>
                    Frequência: {float(r['Freq_pct']):.0f}%<br>
                    Média diária: {int(r['Total_Periodo']/max(1,len(day_cols)))}<br>
                    Saldo: R$ {fmt_money_pt(float(r['Saldo']))}<br>
                    Bloqueio: R$ {fmt_money_pt(float(r['Saldo_Bloqueado']))}<br>

                    Zerado: {int(r['Streak_zero'])}d | Queda: {int(r['Streak_down'])}d
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================
# ANÁLISE (DIÁRIO)
# =========================
with tab_daily:
    if "daily_table" not in st.session_state or "df_final" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    daily_table = st.session_state.get("daily_table", pd.DataFrame())
    start_day = st.session_state.get("daily_start", pd.NaT)
    end_day = st.session_state.get("daily_end", pd.NaT)
    df_final = st.session_state.get("df_final", pd.DataFrame())

    if daily_table is None or daily_table.empty or pd.isna(start_day) or pd.isna(end_day):
        st.info("Sem dados suficientes para montar a análise diária.")
        st.stop()

    st.subheader("Análise de Transações por Dia")
    st.caption("Tabela por dia (do 1º ao último dia do CSV) — por empresa e conta. Total no fim da linha.")

    companies_names = ["TOTAL GERAL"] + sorted(daily_table["Empresa"].dropna().astype(str).unique().tolist())

    cA, cB, cC = st.columns([1.2, 1.0, 1.0], gap="small")
    with cA:
        selected_company = st.selectbox("Empresa (filtro)", options=companies_names, index=0, key="daily_company_select")
    with cB:
        q_nome = st.text_input("Buscar por nome", value="", key="daily_q_nome")
    with cC:
        q_conta = st.text_input("Buscar por conta", value="", key="daily_q_conta")

    view = daily_table.copy()
    if selected_company != "TOTAL GERAL":
        view = view[view["Empresa"] == selected_company].copy()
    if q_nome.strip():
        q = q_nome.strip().lower()
        view = view[view["Empresa"].astype(str).str.lower().str.contains(q, na=False)]
    if q_conta.strip():
        q = re.sub(r"\D+", "", q_conta.strip())
        if q:
            view = view[view["Conta"].astype(str).str.contains(q, na=False)]

    day_cols = [c for c in view.columns if isinstance(c, pd.Timestamp)]

    empresas_filtradas = int(view["Empresa"].nunique()) if not view.empty else 0
    total_periodo = int(view["Total"].sum()) if not view.empty else 0
    last_day_total = int(view[end_day].sum()) if (not view.empty and end_day in view.columns) else 0

    freq_pct = 0.0
    if not view.empty and len(day_cols) > 0:
        day_sums = view[day_cols].sum(axis=0)
        days_with_tx = int((day_sums > 0).sum())
        freq_pct = (days_with_tx / max(1, len(day_cols))) * 100.0

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Empresas filtradas (escopo)", fmt_int_pt(empresas_filtradas))
    with k2:
        st.metric("Total transações (período)", fmt_int_pt(total_periodo))
    with k3:
        st.metric("Total no último dia do CSV", fmt_int_pt(last_day_total))
    with k4:
        st.metric("Frequência (dias com TX)", f"{freq_pct:.0f}%")

    st.divider()
    st.subheader("Assistente (sem LLM) — explicar uma conta")
    st.caption("Escolha uma conta (empresa+conta) para ver métricas e explicação determinística.")

    if view.empty:
        st.info("Sem linhas no escopo atual.")
    else:
        view_keys = view[["Empresa", "Conta"]].astype(str)
        options = (view_keys["Empresa"] + " | " + view_keys["Conta"]).tolist()
        sel = st.selectbox("Conta (escopo atual)", options=options, index=0, key="daily_assistant_sel")

        emp_sel, acc_sel = sel.split(" | ", 1)
        row_daily = view[(view["Empresa"] == emp_sel) & (view["Conta"] == acc_sel)].iloc[0].copy()

        tmp_daily = pd.DataFrame([row_daily])
        tmp_alerts = build_alerts(tmp_daily, start_day, end_day, df_final, alert_cfg)
        if tmp_alerts is not None and not tmp_alerts.empty:
            ar = tmp_alerts.iloc[0]

            st.markdown('<div class="badges">', unsafe_allow_html=True)
            render_badge("Zerado (tail)", f"{int(ar['Streak_zero'])}d", str(ar["tone_zero"]))
            render_badge("Queda (tail)", f"{int(ar['Streak_down'])}d", str(ar["tone_down"]))
            render_badge("Bloqueio", f"R$ {fmt_money_pt(float(ar['Saldo_Bloqueado']))}", str(ar["tone_block"]))
            render_badge("Frequência", f"{float(ar['Freq_pct']):.0f}%", str(ar["tone_freq"]))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(assistant_explain_row(ar, alert_cfg))
        else:
            st.info("Não foi possível calcular explicação para esta conta (dados insuficientes).")

    st.divider()

    pretty = view.copy()
    rename_map = {c: c.strftime("%d") for c in pretty.columns if isinstance(c, pd.Timestamp)}
    pretty = pretty.rename(columns=rename_map)

    for col in list(rename_map.values()) + ["Total"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].astype(int).map(fmt_int_pt)

    st.caption(f"Período do CSV: {start_day.date().isoformat()} → {end_day.date().isoformat()}")
    st.dataframe(pretty, use_container_width=True, hide_index=True)


# =========================
# GRÁFICOS (DIÁRIO)
# =========================
with tab_graphs:
    if "facts" not in st.session_state or "daily_table" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    if px is None:
        st.warning("Instale plotly para ver os gráficos (plotly.express).")
        st.stop()

    facts = st.session_state.get("facts", pd.DataFrame())
    companies_keys = st.session_state.get("companies_keys", [])
    daily_table_all = st.session_state.get("daily_table", pd.DataFrame())

    if facts is None or facts.empty or daily_table_all is None or daily_table_all.empty:
        st.info("Sem dados suficientes para gráficos.")
        st.stop()

    st.subheader("Gráficos (Diário)")
    st.caption("Visual rápido do período diário (1..N do CSV). Tudo calculado localmente (sem LLM).")

    companies_names = ["TOTAL GERAL"] + sorted(daily_table_all["Empresa"].dropna().astype(str).unique().tolist())

    g1, g2 = st.columns([1.2, 0.8], gap="small")
    with g1:
        g_company = st.selectbox("Empresa (escopo dos gráficos)", options=companies_names, index=0, key="graphs_company")
    with g2:
        topn_emp = st.selectbox("Top empresas", options=[8, 10, 15, 20, 30], index=2, key="graphs_topn")

    daily_scope = daily_table_all.copy()
    if g_company != "TOTAL GERAL":
        daily_scope = daily_scope[daily_scope["Empresa"].astype(str) == str(g_company)].copy()

    ts = daily_totals_from_facts(facts, companies_keys, company_name_filter=g_company)
    if ts.empty:
        st.info("Sem dados para série diária no escopo atual.")
        st.stop()

    top_emp_df = top_companies_from_facts(facts, companies_keys, topn=int(topn_emp))
    zero_down = zero_and_down_counts_by_day(daily_scope, alert_cfg)

    kA, kB, kC, kD = st.columns(4, gap="small")
    with kA:
        st.metric("Período", f"{pd.to_datetime(ts['date_dt'].min()).date().isoformat()} → {pd.to_datetime(ts['date_dt'].max()).date().isoformat()}")
    with kB:
        st.metric("Total (escopo)", fmt_int_pt(int(ts["total"].sum())))
    with kC:
        st.metric("Média por dia", fmt_int_pt(int(round(float(ts["total"].mean())))))
    with kD:
        st.metric("Pico diário", fmt_int_pt(int(ts["total"].max())))

    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.caption("1) Total de transações por dia (escopo)")
        fig = px.line(ts, x="date_dt", y="total", markers=True)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.caption(f"2) Top empresas por volume (período) — Top {int(topn_emp)}")
        if top_emp_df.empty:
            st.info("Sem dados para Top empresas.")
        else:
            fig = px.bar(
                top_emp_df.sort_values("total", ascending=True),
                x="total",
                y="company_name",
                orientation="h",
            )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.caption("3) Contas zeradas por dia (count) — escopo")
        if zero_down.empty:
            st.info("Sem dados para zerados por dia.")
        else:
            fig = px.line(zero_down, x="date_dt", y="zero_accounts", markers=True)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.caption(f"4) Contas em queda por dia (count) — baseline {alert_cfg.baseline_days}d e limiar {int(alert_cfg.drop_ratio*100)}%")
        if zero_down.empty:
            st.info("Sem dados para queda por dia.")
        else:
            fig = px.line(zero_down, x="date_dt", y="down_accounts", markers=True)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
