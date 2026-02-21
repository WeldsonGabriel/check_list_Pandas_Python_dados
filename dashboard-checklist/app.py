# app.py
from __future__ import annotations

import io
import re
import math
import json
from typing import Optional, List, Tuple, Dict

import streamlit as st
import pandas as pd
import numpy as np

# Plotly (mantém para o Ground). Gráficos de snapshot serão Matplotlib (bytes PNG).
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Matplotlib (snapshot + cards PNG)
import matplotlib.pyplot as plt

from core import (
    Thresholds,
    AlertConfig,
    STATUS_ORDER,
    STATUS_DOT,
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
    build_alerts,
    daily_totals_from_facts,
    top_companies_from_facts,
    zero_and_down_counts_by_day,
    resolve_webhooks_from_sources,
    send_discord_webhook,  # texto (urllib) já existente
    # NOVOS (cards PNG + envio 1 por vez)
    build_card_images_from_alerts,
    send_card_images_to_discord,
    discord_send_single_image,
)


# =========================
# Secrets safe loader
# =========================
def _get_streamlit_secrets_safe() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


# =========================
# CSS
# =========================
st.set_page_config(page_title="Checklist Semanas + BaaS", layout="wide")

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
# Snapshot charts (Matplotlib) -> bytes PNG
# =========================
def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_daily_total(ts: pd.DataFrame, title: str) -> bytes:
    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    if ts is None or ts.empty:
        ax.set_title("Sem dados")
        ax.set_xlabel("Dia")
        ax.set_ylabel("Transações")
        fig.tight_layout()
        return _fig_to_png_bytes(fig)

    x = pd.to_datetime(ts["date_dt"]).dt.date.astype(str).tolist()
    y = ts["total"].astype(int).tolist()

    ax.plot(range(len(x)), y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Dia")
    ax.set_ylabel("Transações")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([d[-2:] for d in x], rotation=0)

    max_y = max(y) if y else 0
    for i, v in enumerate(y):
        ax.annotate(str(int(v)), (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(bottom=0, top=max_y * 1.15 + 1)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def chart_top_companies(top_df: pd.DataFrame, title: str) -> bytes:
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    if top_df is None or top_df.empty:
        ax.set_title("Sem dados")
        ax.set_xlabel("Total")
        ax.set_ylabel("Empresa")
        fig.tight_layout()
        return _fig_to_png_bytes(fig)

    d = top_df.copy()
    d["total"] = d["total"].astype(int)
    d = d.sort_values("total", ascending=True)
    ylabels = d["company_name"].astype(str).tolist()
    vals = d["total"].tolist()

    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    ax.set_xlabel("Total no período")

    max_v = max(vals) if vals else 0
    for i, v in enumerate(vals):
        ax.text(v + max(1, int(max_v * 0.01)), i, str(int(v)), va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def chart_counts(zero_down: pd.DataFrame, col: str, title: str) -> bytes:
    fig = plt.figure(figsize=(10, 3.2))
    ax = fig.add_subplot(111)

    if zero_down is None or zero_down.empty or col not in zero_down.columns:
        ax.set_title("Sem dados")
        ax.set_xlabel("Dia")
        ax.set_ylabel("Contas")
        fig.tight_layout()
        return _fig_to_png_bytes(fig)

    x = pd.to_datetime(zero_down["date_dt"]).dt.date.astype(str).tolist()
    y = zero_down[col].astype(int).tolist()

    ax.plot(range(len(x)), y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Dia")
    ax.set_ylabel("Contas")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([d[-2:] for d in x], rotation=0)

    max_y = max(y) if y else 0
    for i, v in enumerate(y):
        ax.annotate(str(int(v)), (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(bottom=0, top=max_y * 1.15 + 1)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


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

    st.divider()
    st.header("Discord (Webhooks)")

    st.caption("Manual (opcional). Se vazio, tenta st.secrets/env.")
    wh_zero = st.text_input("Webhook — Zeradas", value="", type="password", key="wh_zero")
    wh_down = st.text_input("Webhook — Queda", value="", type="password", key="wh_down")
    wh_block = st.text_input("Webhook — Bloqueio", value="", type="password", key="wh_block")

    st.caption("Snapshot (4 gráficos + cards NORMAL).")
    wh_snapshot = st.text_input("Webhook — Snapshot", value="", type="password", key="wh_snapshot")

    top_n_discord = st.selectbox("Top por categoria (Discord)", [20, 40, 60, 100, 150], index=2, key="top_n_discord")

    st.divider()
    st.header("Snapshot (config)")
    snap_topn_emp = st.selectbox("Top empresas (snapshot)", [8, 10, 15, 20, 30], index=2, key="snap_topn_emp")


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

            # n_days (pra média diária dos cards PNG bater com o período do CSV)
            day_cols_all = [c for c in daily_table.columns if isinstance(c, pd.Timestamp)]
            n_days = int(len(day_cols_all))

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
            st.session_state["n_days"] = n_days

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
    st.subheader("Envio para Discord (manual) — Cards PNG (1 por mensagem) + Snapshot")

    manual_hooks = {
        "DISCORD_WEBHOOK_ZERADAS": st.session_state.get("wh_zero", ""),
        "DISCORD_WEBHOOK_QUEDA": st.session_state.get("wh_down", ""),
        "DISCORD_WEBHOOK_BLOQUEIO": st.session_state.get("wh_block", ""),
        "DISCORD_WEBHOOK_SNAPSHOT": st.session_state.get("wh_snapshot", ""),
    }

    hooks = resolve_webhooks_from_sources(
        secrets=_get_streamlit_secrets_safe(),
        env=None,
        manual=manual_hooks,
    )

    snapshot_hook = (hooks.get("SNAPSHOT") or "").strip()
    zeradas_hook = (hooks.get("ZERADAS") or "").strip()
    queda_hook = (hooks.get("QUEDA") or "").strip()
    bloqueio_hook = (hooks.get("BLOQUEIO") or "").strip()

    b1, b2, b3, b4 = st.columns(4, gap="small")
    with b1:
        send_zero_btn = st.button("Enviar Zeradas (cards)", use_container_width=True)
    with b2:
        send_down_btn = st.button("Enviar Queda (cards)", use_container_width=True)
    with b3:
        send_block_btn = st.button("Enviar Bloqueio (cards)", use_container_width=True)
    with b4:
        send_snapshot_btn = st.button("Enviar Snapshot (gráficos + NORMAL)", use_container_width=True)

    # =========
    # Envios (cards)
    # =========
    if send_zero_btn or send_down_btn or send_block_btn:
        if "alerts_df" not in st.session_state or "daily_table" not in st.session_state:
            st.error("Você precisa processar primeiro (alerts_df/daily_table não encontrados).")
        else:
            alerts_df = st.session_state.get("alerts_df", pd.DataFrame())
            period = str(alerts_df["Periodo"].iloc[0]) if (alerts_df is not None and not alerts_df.empty and "Periodo" in alerts_df.columns) else "—"
            n_days = int(st.session_state.get("n_days", 0))

            top_n_each = int(st.session_state.get("top_n_discord", 60))

            if send_zero_btn:
                if not zeradas_hook:
                    st.error("Webhook Zeradas não configurado.")
                else:
                    desc = (
                        f"**ALERTAS — ZERADAS (CARDS PNG)**\n"
                        f"**Período (diário):** {period}\n"
                        f"**Regra:** Streak_zero ≥ {int(alert_cfg.zero_yellow)} dia(s)\n"
                        f"**Envio:** 1 card por mensagem (sempre com cabeçalho)\n"
                    )
                    with st.spinner("Gerando cards e enviando (ZERADAS)..."):
                        imgs = build_card_images_from_alerts(
                            alerts_df=alerts_df,
                            category="ZERADAS",
                            cfg=alert_cfg,
                            n_days=n_days,
                            top_n=top_n_each,
                        )
                        report = send_card_images_to_discord(
                            zeradas_hook,
                            description=desc,
                            images=imgs,
                            username="Checklist Bot",
                        )
                    st.success(f"ZERADAS: enviados {report.get('sent',0)} card(s).") if not report.get("errors") else st.error(report.get("errors"))

            if send_down_btn:
                if not queda_hook:
                    st.error("Webhook Queda não configurado.")
                else:
                    desc = (
                        f"**ALERTAS — QUEDA (CARDS PNG)**\n"
                        f"**Período (diário):** {period}\n"
                        f"**Regra:** Streak_down ≥ {int(alert_cfg.down_yellow)} dia(s) | baseline {int(alert_cfg.baseline_days)}d | limiar {int(alert_cfg.drop_ratio*100)}%\n"
                        f"**Envio:** 1 card por mensagem (sempre com cabeçalho)\n"
                    )
                    with st.spinner("Gerando cards e enviando (QUEDA)..."):
                        imgs = build_card_images_from_alerts(
                            alerts_df=alerts_df,
                            category="QUEDA",
                            cfg=alert_cfg,
                            n_days=n_days,
                            top_n=top_n_each,
                        )
                        report = send_card_images_to_discord(
                            queda_hook,
                            description=desc,
                            images=imgs,
                            username="Checklist Bot",
                        )
                    st.success(f"QUEDA: enviados {report.get('sent',0)} card(s).") if not report.get("errors") else st.error(report.get("errors"))

            if send_block_btn:
                if not bloqueio_hook:
                    st.error("Webhook Bloqueio não configurado.")
                else:
                    desc = (
                        f"**ALERTAS — BLOQUEIO (CARDS PNG)**\n"
                        f"**Período (diário):** {period}\n"
                        f"**Regra:** Bloqueio ≥ R$ {fmt_money_pt(float(alert_cfg.block_yellow))}\n"
                        f"**Envio:** 1 card por mensagem (sempre com cabeçalho)\n"
                    )
                    with st.spinner("Gerando cards e enviando (BLOQUEIO)..."):
                        imgs = build_card_images_from_alerts(
                            alerts_df=alerts_df,
                            category="BLOQUEIO",
                            cfg=alert_cfg,
                            n_days=n_days,
                            top_n=top_n_each,
                        )
                        report = send_card_images_to_discord(
                            bloqueio_hook,
                            description=desc,
                            images=imgs,
                            username="Checklist Bot",
                        )
                    st.success(f"BLOQUEIO: enviados {report.get('sent',0)} card(s).") if not report.get("errors") else st.error(report.get("errors"))

    # =========
    # Snapshot (gráficos + cards NORMAL no mesmo webhook)
    # =========
    if send_snapshot_btn:
        if not snapshot_hook:
            st.error("Webhook Snapshot não configurado.")
        elif "facts" not in st.session_state or "daily_table" not in st.session_state or "alerts_df" not in st.session_state:
            st.error("Você precisa processar primeiro (dados não encontrados).")
        else:
            facts = st.session_state.get("facts", pd.DataFrame())
            companies_keys = st.session_state.get("companies_keys", [])
            daily_table_all = st.session_state.get("daily_table", pd.DataFrame())
            alerts_df = st.session_state.get("alerts_df", pd.DataFrame())
            n_days = int(st.session_state.get("n_days", 0))

            period = str(alerts_df["Periodo"].iloc[0]) if (alerts_df is not None and not alerts_df.empty and "Periodo" in alerts_df.columns) else "—"

            # recalcula (para refletir baseline/ratio do sidebar)
            ts = daily_totals_from_facts(facts, companies_keys, company_name_filter="TOTAL GERAL")
            top_df = top_companies_from_facts(facts, companies_keys, topn=int(st.session_state.get("snap_topn_emp", 15)))
            zero_down = zero_and_down_counts_by_day(daily_table_all, alert_cfg)

            # KPIs
            total_scope = int(ts["total"].sum()) if not ts.empty else 0
            avg_day = int(round(float(ts["total"].mean()))) if not ts.empty else 0
            peak_day = int(ts["total"].max()) if not ts.empty else 0

            last_day_dt = pd.to_datetime(ts["date_dt"].max()).date() if not ts.empty else None
            last_day = str(last_day_dt.isoformat()) if last_day_dt else "—"
            last_day_tx = int(ts.loc[ts["date_dt"] == pd.to_datetime(last_day_dt), "total"].iloc[0]) if (last_day_dt and not ts.empty) else 0

            # NOVO: média dos dias anteriores + % do último dia vs média anterior
            mean_prev = 0
            pct_last_vs_prev = 0.0
            if ts is not None and not ts.empty and len(ts) >= 2:
                ts_sorted = ts.sort_values("date_dt").reset_index(drop=True)
                prev = ts_sorted.iloc[:-1]
                mean_prev = int(round(float(prev["total"].mean()))) if not prev.empty else 0
                if mean_prev > 0:
                    pct_last_vs_prev = ((float(last_day_tx) - float(mean_prev)) / float(mean_prev)) * 100.0

            # contagens semanal (df_final)
            counts_week = {s: int((df_final["status"] == s).sum()) for s in STATUS_ORDER} if not df_final.empty else {s: 0 for s in STATUS_ORDER}

            # descrição (ajustes pedidos)
            desc = (
                f"**CHECKLIST SNAPSHOT**\n"
                f"**Período (diário):** {period}\n"
                f"**Último dia do CSV:** {last_day}\n"
                f"**Último dia (transações):** {fmt_int_pt(last_day_tx)}\n\n"
                f"**Total no período (transações no escopo):** {fmt_int_pt(total_scope)}\n"
                f"**Média/dia:** {fmt_int_pt(avg_day)} | **Pico diário:** {fmt_int_pt(peak_day)}\n"
                f"**Média (dias anteriores):** {fmt_int_pt(mean_prev)} | **Último dia vs média anterior:** {pct_last_vs_prev:+.0f}%\n\n"
                f"**Status semanal (contas):** "
                f"🔴 {counts_week['Alerta (queda/ zerada)']} | "
                f"🟡 {counts_week['Investigar']} | "
                f"🔵 {counts_week['Gerenciar (aumento)']} | "
                f"🟢 {counts_week['Normal']}\n"
                f"**Bloqueios:** {fmt_int_pt(int(kpi_blocked_accounts))} conta(s) | **R$ {fmt_money_pt(float(kpi_blocked_sum))}**\n"
                f"**NORMAL:** cards PNG enviados neste canal (1 por mensagem)."
            )

            with st.spinner("Gerando e enviando Snapshot (1 por vez)..."):
                # 1) envia o texto (log)
                ok_txt, msg_txt = send_discord_webhook(snapshot_hook, desc, username="Checklist Bot")
                if not ok_txt:
                    st.warning(f"Falha ao enviar texto do snapshot: {msg_txt}")

                # 2) gera PNGs dos 4 gráficos
                img1 = chart_daily_total(ts, "1) Total de transações por dia (escopo)")
                img2 = chart_top_companies(top_df, f"2) Top empresas por volume (Top {len(top_df)})")
                img3 = chart_counts(zero_down, "zero_accounts", "3) Contas zeradas por dia (count)")
                img4 = chart_counts(
                    zero_down,
                    "down_accounts",
                    f"4) Contas em queda por dia (baseline {alert_cfg.baseline_days}d / limiar {int(alert_cfg.drop_ratio*100)}%)",
                )

                # 3) envia 1 gráfico por mensagem (sempre com cabeçalho)
                charts = [
                    ("01 - Total por dia.png", img1),
                    ("02 - Top empresas.png", img2),
                    ("03 - Zeradas por dia.png", img3),
                    ("04 - Queda por dia.png", img4),
                ]
                for fname, bts in charts:
                    ok, msg = discord_send_single_image(
                        snapshot_hook,
                        description=desc,
                        filename=fname,
                        png_bytes=bts,
                        title=fname.replace(".png", ""),
                        username="Checklist Bot",
                    )
                    if not ok:
                        st.error(f"Falha ao enviar gráfico {fname}: {msg}")
                        st.stop()

                # 4) NORMAL -> cards PNG no snapshot
                normal_imgs = build_card_images_from_alerts(
                    alerts_df=alerts_df,
                    category="NORMAL",
                    cfg=alert_cfg,
                    n_days=n_days,
                    top_n=int(st.session_state.get("top_n_discord", 60)),
                )
                normal_report = send_card_images_to_discord(
                    snapshot_hook,
                    description=desc,
                    images=normal_imgs,
                    username="Checklist Bot",
                )

            if normal_report.get("errors"):
                st.warning(f"Snapshot OK (gráficos), mas NORMAL teve erro: {normal_report.get('errors')}")
            else:
                st.success(f"Snapshot enviado + NORMAL: {normal_report.get('sent',0)} card(s).")

    st.divider()
    st.subheader("Visão Geral (semanal)")
    render_status_boxes(df_final)
    st.divider()

    if go is None:
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
    st.subheader("Cards (texto, opcional) — semanal")
    st.caption("Somente para visualização no app (o envio para Discord agora é via cards PNG).")

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
# ALERTAS
# =========================
with tab_alerts:
    if "alerts_df" not in st.session_state or "daily_table" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    alerts_df = st.session_state["alerts_df"]
    daily_table = st.session_state["daily_table"]

    st.subheader("Hábitos — Monitoramento Financeiro (estilo Zabbix)")

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
# GRÁFICOS (DIÁRIO) — Matplotlib (idêntico Streamlit/Discord)
# =========================
with tab_graphs:
    if "facts" not in st.session_state or "daily_table" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    facts = st.session_state.get("facts", pd.DataFrame())
    companies_keys = st.session_state.get("companies_keys", [])
    daily_table_all = st.session_state.get("daily_table", pd.DataFrame())

    if facts is None or facts.empty or daily_table_all is None or daily_table_all.empty:
        st.info("Sem dados suficientes para gráficos.")
        st.stop()

    st.subheader("Gráficos (Diário)")
    st.caption("Snapshot = exatamente estes 4 gráficos (PNG) + cards NORMAL (PNG).")

    topn_emp = int(st.session_state.get("snap_topn_emp", 15))

    ts = daily_totals_from_facts(facts, companies_keys, company_name_filter="TOTAL GERAL")
    top_emp_df = top_companies_from_facts(facts, companies_keys, topn=topn_emp)
    zero_down = zero_and_down_counts_by_day(daily_table_all, alert_cfg)

    if ts.empty:
        st.info("Sem dados para série diária.")
        st.stop()

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

    img1 = chart_daily_total(ts, "1) Total de transações por dia (escopo)")
    st.image(img1, use_container_width=True)

    img2 = chart_top_companies(top_emp_df, f"2) Top empresas por volume (Top {len(top_emp_df)})")
    st.image(img2, use_container_width=True)

    img3 = chart_counts(zero_down, "zero_accounts", "3) Contas zeradas por dia (count)")
    st.image(img3, use_container_width=True)

    img4 = chart_counts(
        zero_down,
        "down_accounts",
        f"4) Contas em queda por dia (baseline {alert_cfg.baseline_days}d / limiar {int(alert_cfg.drop_ratio*100)}%)",
    )
    st.image(img4, use_container_width=True)