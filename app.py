import re
from io import BytesIO
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import patheffects

# Opcional: Plotly para gráficos interativos
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

st.set_page_config(page_title="Análise de Suscetibilidade", layout="wide")

# ---------------------------
# Constantes e utilitários
# ---------------------------
ABX_LEGENDA = [
    "GEN", "TOB", "AMI",
    "ATM",
    "CRO", "CAZ", "CTX", "CFO", "CPM",
    "AMC", "AMP", "PPT", "CZA",
    "MER", "IMP",
    "CIP", "LEV",
    "SUT",
    "POLI B",
]

def norm_str(s):
    if pd.isna(s):
        return ""
    return str(s).replace("–", "-").strip()

def extrai_num_codigo(cod):
    s = norm_str(cod).upper().replace(" ", "")
    m = re.search(r"\bMA\s*(\d+)", s)
    return int(m.group(1)) if m else np.nan

def normaliza_status(v):
    """Extrai o primeiro rótulo válido {SSD, R, S, I} em qualquer ponto da célula."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).upper().replace("–", "-")
    s = s.replace("INTERMEDIARIO", "I").replace("INTERMEDIÁRIO", "I")
    s = s.replace("SSD/I", "SSD").replace("SSD –", "SSD").replace("SSD–", "SSD")
    m = re.search(r"\b(SSD|R|S|I)\b", s)
    return m.group(1) if m else None

def achar_col(df, patt):
    for c in df.columns:
        if re.search(patt, norm_str(c), flags=re.IGNORECASE):
            return c
    return None

def perfil_sri_rows(df_in, abx_cols):
    """Calcula métricas por antibiótico.
    Retorna: lista de linhas [Antibiótico, N, Total, %Cobertura, R, S, I, SSD, %R, %S, %I+SSD, Sem Resultado]
    """
    rows = []
    total = len(df_in)
    for c in abx_cols:
        vals = [normaliza_status(v) for v in df_in[c]]
        valid = [v for v in vals if v is not None]
        n = len(valid)
        cnt = Counter(valid)
        S = cnt.get("S", 0)
        R = cnt.get("R", 0)
        I = cnt.get("I", 0)
        SSD = cnt.get("SSD", 0)
        perc_R = round(100 * R / n, 1) if n > 0 else 0.0
        perc_S = round(100 * S / n, 1) if n > 0 else 0.0
        perc_I_SSD = round(100 * (I + SSD) / n, 1) if n > 0 else 0.0
        cobertura = round(100 * n / total, 1) if total > 0 else 0.0
        rows.append([c, n, total, cobertura, R, S, I, SSD, perc_R, perc_S, perc_I_SSD, total - n])
    rows.sort(key=lambda r: (r[8], r[3], r[0]), reverse=True)
    return rows

def rows_to_df(rows):
    cols = [
        "Antibiótico", "N", "Total", "%Cobertura", "R", "S", "I", "SSD",
        "%R", "%S", "%I+SSD", "Sem Resultado",
    ]
    return pd.DataFrame(rows, columns=cols)

def format_df_numeric(df):
    fmt_cols = ["%Cobertura", "%R", "%S", "%I+SSD"]
    for c in fmt_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(1)
    return df

def fig_topR(rows, titulo, top=15):
    dfp = rows_to_df(rows)
    if dfp.empty:
        return None
    dfp = dfp.sort_values("%R", ascending=False).head(top)
    if HAS_PLOTLY:
        fig = px.bar(
            dfp.sort_values("%R"), x="%R", y="Antibiótico", orientation="h",
            title=titulo, labels={"%R": "% Resistência (R)"}, text="%R"
        )
        # Mostrar rótulos de % dentro das barras
        fig.update_traces(texttemplate='%{x:.1f}% ', textposition='inside', textfont_color='white')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, max(3, len(dfp) * 0.35)))
        bars = ax.barh(dfp["Antibiótico"], dfp["%R"])
        ax.set_title(titulo)
        ax.set_xlabel("% Resistência (R)")
        # Rótulos dentro das barras
        try:
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in dfp["%R"]], label_type='center', color='white')
        except Exception:
            pass
        plt.tight_layout()
        return fig

def fig_stacked(rows, titulo, ordenar_por="%R", label_threshold=5.0, label_decimals=1, label_fontsize=11, label_vertical=True, label_vertical_dir="up", show_labels=True):
    dfp = rows_to_df(rows)
    if dfp.empty:
        return None
    if ordenar_por in dfp.columns:
        dfp = dfp.sort_values(ordenar_por, ascending=False)
    if HAS_PLOTLY:
        dfm = dfp.melt(id_vars=["Antibiótico", "N"], value_vars=["%S", "%I+SSD", "%R"],
                       var_name="Classe", value_name="Percentual")
        ordem = {"%S": "Sensível (S)", "%I+SSD": "Intermediário (I/SSD)", "%R": "Resistente (R)"}
        dfm["Classe"] = dfm["Classe"].map(ordem)
        # Monta rótulo condicional por ponto
        fmt = f"{{:.{label_decimals}f}}%"
        if show_labels and label_vertical:
            dfm["Label"] = dfm["Percentual"].apply(lambda v: fmt.format(float(v)) if float(v) >= float(label_threshold) else "")
        else:
            dfm["Label"] = ""
        fig = px.bar(
            dfm, x="Antibiótico", y="Percentual", color="Classe", title=titulo,
            color_discrete_map={
                "Sensível (S)": "#2ca02c",
                "Intermediário (I/SSD)": "#1f77b4",
                "Resistente (R)": "#d62728",
            }, text="Label"
        )
        # Rótulos dentro das barras empilhadas, verticais e com margem para evitar corte no topo
        # 90 = baixo->cima; -90 (ou 270) = cima->baixo
        angle = 0
        if show_labels and label_vertical:
            angle = 90 if str(label_vertical_dir).lower() in ("up", "cima", "baixo->cima", "bottom-up", "bottom_to_top") else -90
        fig.update_traces(
            textposition=('inside' if show_labels and label_vertical else 'none'),
            textangle=angle,
            textfont_color='white',
            textfont_size=label_fontsize,
            cliponaxis=False
        )
        fig.update_layout(
            xaxis_tickangle=-60,
            yaxis_range=[0, 102],
            uniformtext_minsize=10,
            uniformtext_mode='show',
            margin=dict(t=60)
        )
        return fig
    else:
        x = np.arange(len(dfp))
        fig, ax = plt.subplots(figsize=(max(8, len(dfp) * 0.8), 6.2))
        bars_S = ax.bar(x, dfp["%S"], label="Sensível (S)", color="#2ca02c")
        bars_I = ax.bar(x, dfp["%I+SSD"], bottom=dfp["%S"], label="Intermediário (I/SSD)", color="#1f77b4")
        bars_R = ax.bar(x, dfp["%R"], bottom=dfp["%S"] + dfp["%I+SSD"], label="Resistente (R)", color="#d62728")
        ax.set_title(titulo)
        ax.set_ylabel("% de isolados testados")
        ax.set_xticks(x)
        ax.set_xticklabels(dfp["Antibiótico"], rotation=60, ha="right")
        ax.set_ylim(0, 102)  # pequena folga para rótulos
        ax.legend(loc="upper right")
        # Rótulos centralizados em cada segmento
        if show_labels and label_vertical:
            try:
                fmt = f"{{:.{label_decimals}f}}%"
                def labels(vals):
                    return [fmt.format(float(v)) if float(v) >= float(label_threshold) else "" for v in vals]
                # 90 = baixo->cima; -90 = cima->baixo
                rot = 90 if str(label_vertical_dir).lower() in ("up", "cima", "baixo->cima", "bottom-up", "bottom_to_top") else -90
                texts = []
                texts += ax.bar_label(bars_S, labels=labels(dfp["%S"]), label_type='center', color='white', rotation=rot, padding=0, fontsize=label_fontsize)
                texts += ax.bar_label(bars_I, labels=labels(dfp["%I+SSD"]), label_type='center', color='white', rotation=rot, padding=0, fontsize=label_fontsize)
                texts += ax.bar_label(bars_R, labels=labels(dfp["%R"]), label_type='center', color='white', rotation=rot, padding=0, fontsize=label_fontsize)
                # contorno para legibilidade
                for t in texts:
                    if hasattr(t, 'set_path_effects'):
                        t.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])
            except Exception:
                pass
        plt.tight_layout()
        return fig

def to_csv_download(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    buffer.seek(0)
    return buffer

def _valid_species_df(df: pd.DataFrame, col_maldi: str) -> pd.DataFrame:
    """Retorna apenas linhas com espécie válida (não vazia, não '*')."""
    if col_maldi not in df.columns:
        return pd.DataFrame(columns=df.columns)
    s = df[col_maldi].astype(str).str.strip()
    mask = s.notna() & (s != "") & (s != "*")
    return df.loc[mask].copy()


def _style_susc_df(df_rows: pd.DataFrame):
    """Aplica estilo padrão às tabelas de suscetibilidade."""
    return (
        df_rows.style
        .background_gradient(subset=["%R"], cmap="Reds")
        .background_gradient(subset=["%S"], cmap="Greens")
        .format({"%R": "{:.1f}", "%S": "{:.1f}", "%I+SSD": "{:.1f}", "%Cobertura": "{:.1f}"})
    )


def _build_resumo(df_especie: pd.DataFrame, col_maldi: str, sel_abx: list[str]) -> pd.DataFrame:
    """Monta tabela resumo por espécie com médias ponderadas de %R e %S."""
    resumo_data = []
    for especie, df_esp in df_especie.groupby(col_maldi):
        rows_esp = perfil_sri_rows(df_esp, sel_abx)
        df_rows_esp = format_df_numeric(rows_to_df(rows_esp))
        if not df_rows_esp.empty and df_rows_esp["N"].sum() > 0:
            peso = df_rows_esp["N"].sum()
            resistencia_media = (df_rows_esp["%R"] * df_rows_esp["N"]).sum() / peso
            sensibilidade_media = (df_rows_esp["%S"] * df_rows_esp["N"]).sum() / peso
            total_testes = int(peso)
        else:
            resistencia_media = 0.0
            sensibilidade_media = 0.0
            total_testes = 0
        resumo_data.append({
            "Espécie": especie,
            "Isolados": int(len(df_esp)),
            "Total de Testes": total_testes,
            "% Resistência Média": round(float(resistencia_media), 1),
            "% Sensibilidade Média": round(float(sensibilidade_media), 1),
        })
    df_resumo = pd.DataFrame(resumo_data)
    if not df_resumo.empty:
        df_resumo = df_resumo.sort_values("% Resistência Média", ascending=False)
    return df_resumo


def _comparativo_temporal_especie(df_2025: pd.DataFrame, df_anterior: pd.DataFrame, 
                                  col_maldi: str, especie_selecionada: str, 
                                  abx_selecionados: list[str]):
    """Função para comparar a mesma espécie entre períodos temporais diferentes."""
    
    if not especie_selecionada:
        st.warning("Selecione uma espécie para comparação temporal.")
        return
    
    if not abx_selecionados:
        st.warning("Selecione pelo menos 1 antibiótico para comparação.")
        return
    
    # Filtrar dados para a espécie selecionada em cada período
    df_2025_esp = df_2025[df_2025[col_maldi] == especie_selecionada].copy()
    df_anterior_esp = df_anterior[df_anterior[col_maldi] == especie_selecionada].copy()
    
    n_2025 = len(df_2025_esp)
    n_anterior = len(df_anterior_esp)
    
    st.subheader(f"📊 Comparação Temporal: {especie_selecionada}")
    st.caption(f"2025/atual: {n_2025} isolados | Anos anteriores: {n_anterior} isolados")
    
    if n_2025 == 0 and n_anterior == 0:
        st.warning("Nenhum isolado encontrado para esta espécie em nenhum período.")
        return
    elif n_2025 == 0:
        st.warning("Nenhum isolado encontrado para esta espécie em 2025/atual.")
        return
    elif n_anterior == 0:
        st.warning("Nenhum isolado encontrado para esta espécie em anos anteriores.")
        return
    
    # Calcular perfis para cada período
    rows_2025 = perfil_sri_rows(df_2025_esp, abx_selecionados)
    rows_anterior = perfil_sri_rows(df_anterior_esp, abx_selecionados)
    
    # Converter para DataFrames
    df_2025_perfil = format_df_numeric(rows_to_df(rows_2025))
    df_anterior_perfil = format_df_numeric(rows_to_df(rows_anterior))
    
    # 1. Tabela comparativa lado a lado
    st.markdown("### 📋 Comparação Detalhada por Antibiótico")
    
    dados_comparativos = []
    for antibiotico in abx_selecionados:
        # Dados de 2025
        row_2025 = df_2025_perfil[df_2025_perfil["Antibiótico"] == antibiotico]
        if not row_2025.empty:
            data_2025 = row_2025.iloc[0]
        else:
            data_2025 = {"%R": 0, "%S": 0, "%I+SSD": 0, "%Cobertura": 0, "N": 0}
        
        # Dados anteriores
        row_anterior = df_anterior_perfil[df_anterior_perfil["Antibiótico"] == antibiotico]
        if not row_anterior.empty:
            data_anterior = row_anterior.iloc[0]
        else:
            data_anterior = {"%R": 0, "%S": 0, "%I+SSD": 0, "%Cobertura": 0, "N": 0}
        
        # Calcular diferenças
        diff_r = data_2025["%R"] - data_anterior["%R"]
        diff_s = data_2025["%S"] - data_anterior["%S"]
        
        dados_comparativos.append({
            "Antibiótico": antibiotico,
            "2025_N": data_2025["N"],
            "2025_%R": data_2025["%R"],
            "2025_%S": data_2025["%S"],
            "2025_%I+SSD": data_2025["%I+SSD"],
            "2025_%Cobertura": data_2025["%Cobertura"],
            "Anterior_N": data_anterior["N"],
            "Anterior_%R": data_anterior["%R"],
            "Anterior_%S": data_anterior["%S"],
            "Anterior_%I+SSD": data_anterior["%I+SSD"],
            "Anterior_%Cobertura": data_anterior["%Cobertura"],
            "Δ%R": diff_r,
            "Δ%S": diff_s,
            "Tendência_R": "↑" if diff_r > 5 else "↓" if diff_r < -5 else "→",
            "Tendência_S": "↑" if diff_s > 5 else "↓" if diff_s < -5 else "→"
        })
    
    df_comparativo_temporal = pd.DataFrame(dados_comparativos)
    
    # Mostrar tabela com formatação
    if not df_comparativo_temporal.empty:
        # Separar em três partes para melhor visualização
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**2025/Atual**")
            cols_2025 = ["Antibiótico", "2025_N", "2025_%R", "2025_%S", "2025_%I+SSD", "2025_%Cobertura"]
            df_2025_show = df_comparativo_temporal[cols_2025].copy()
            df_2025_show.columns = ["Antibiótico", "N", "%R", "%S", "%I+SSD", "%Cobertura"]
            
            st.dataframe(
                df_2025_show.style
                .background_gradient(subset=["%R"], cmap="Reds")
                .background_gradient(subset=["%S"], cmap="Greens")
                .format({"%R": "{:.1f}", "%S": "{:.1f}", "%I+SSD": "{:.1f}", "%Cobertura": "{:.1f}"}),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Anos Anteriores**")
            cols_anterior = ["Antibiótico", "Anterior_N", "Anterior_%R", "Anterior_%S", "Anterior_%I+SSD", "Anterior_%Cobertura"]
            df_anterior_show = df_comparativo_temporal[cols_anterior].copy()
            df_anterior_show.columns = ["Antibiótico", "N", "%R", "%S", "%I+SSD", "%Cobertura"]
            
            st.dataframe(
                df_anterior_show.style
                .background_gradient(subset=["%R"], cmap="Reds")
                .background_gradient(subset=["%S"], cmap="Greens")
                .format({"%R": "{:.1f}", "%S": "{:.1f}", "%I+SSD": "{:.1f}", "%Cobertura": "{:.1f}"}),
                use_container_width=True
            )
        
        with col3:
            st.markdown("**Variação (2025 - Anterior)**")
            cols_diff = ["Antibiótico", "Δ%R", "Δ%S", "Tendência_R", "Tendência_S"]
            df_diff_show = df_comparativo_temporal[cols_diff].copy()
            
            # Aplicar cores baseadas na variação com melhor contraste
            def color_delta(val):
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    if val > 5:
                        return 'background-color: #ffebee; color: #c62828; font-weight: bold'  # fundo rosa claro, texto vermelho escuro
                    elif val < -5:
                        return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'  # fundo verde claro, texto verde escuro
                    else:
                        return 'color: #424242'  # texto cinza escuro para valores estáveis
                return ''
            
            st.dataframe(
                df_diff_show.style
                .applymap(color_delta, subset=["Δ%R", "Δ%S"])
                .format({"Δ%R": "{:.1f}", "Δ%S": "{:.1f}"}),
                use_container_width=True
            )
        
        # Legenda das tendências
        st.caption("📊 **Legenda de Tendências**: ↑ = Aumento >5 p.p. | ↓ = Diminuição >5 p.p. | → = Estável (±5 p.p.)")
        
        # Download da tabela completa
        st.download_button(
            label="📥 Baixar Comparação Temporal (CSV)",
            data=to_csv_download(df_comparativo_temporal),
            file_name=f"comparacao_temporal_{especie_selecionada.replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"download_temporal_{especie_selecionada}"
        )
    
    # 2. Gráficos comparativos
    st.markdown("### 📈 Visualizações Comparativas")
    
    tipo_grafico_temporal = st.selectbox(
        "Tipo de visualização",
        options=[
            "Resistência: 2025 vs Anterior", 
            "Sensibilidade: 2025 vs Anterior",
            "Variação da Resistência (Δ%R)",
            "Perfil Completo: Lado a Lado"
        ],
        key="tipo_grafico_temporal"
    )
    
    if not df_comparativo_temporal.empty:
        if tipo_grafico_temporal == "Resistência: 2025 vs Anterior":
            fig = _grafico_temporal_barras_duplas(df_comparativo_temporal, "R", especie_selecionada)
        elif tipo_grafico_temporal == "Sensibilidade: 2025 vs Anterior":
            fig = _grafico_temporal_barras_duplas(df_comparativo_temporal, "S", especie_selecionada)
        elif tipo_grafico_temporal == "Variação da Resistência (Δ%R)":
            fig = _grafico_variacao_temporal(df_comparativo_temporal, especie_selecionada)
        elif tipo_grafico_temporal == "Perfil Completo: Lado a Lado":
            fig = _grafico_perfil_temporal_completo(df_2025_perfil, df_anterior_perfil, especie_selecionada)
        
        if fig:
            if HAS_PLOTLY:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig, use_container_width=True)
    
    # 3. Resumo estatístico temporal
    st.markdown("### 📊 Resumo da Evolução Temporal")
    
    if not df_comparativo_temporal.empty:
        # Filtrar apenas antibióticos com dados em ambos os períodos
        df_validos = df_comparativo_temporal[
            (df_comparativo_temporal["2025_N"] > 0) & 
            (df_comparativo_temporal["Anterior_N"] > 0)
        ].copy()
        
        if not df_validos.empty:
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.markdown("**📈 Tendências de Resistência**")
                aumentos_r = len(df_validos[df_validos["Δ%R"] > 5])
                diminuicoes_r = len(df_validos[df_validos["Δ%R"] < -5])
                estaveis_r = len(df_validos) - aumentos_r - diminuicoes_r
                
                st.metric("Antibióticos com ↑ Resistência", aumentos_r, f"{aumentos_r/len(df_validos)*100:.1f}%")
                st.metric("Antibióticos com ↓ Resistência", diminuicoes_r, f"{diminuicoes_r/len(df_validos)*100:.1f}%")
                st.metric("Antibióticos Estáveis", estaveis_r, f"{estaveis_r/len(df_validos)*100:.1f}%")
            
            with col_stats2:
                st.markdown("**📊 Variações Médias**")
                delta_r_medio = df_validos["Δ%R"].mean()
                delta_s_medio = df_validos["Δ%S"].mean()
                
                st.metric("Δ%R Médio", f"{delta_r_medio:.1f} p.p.", 
                         delta=f"{'+' if delta_r_medio > 0 else ''}{delta_r_medio:.1f} p.p.")
                st.metric("Δ%S Médio", f"{delta_s_medio:.1f} p.p.", 
                         delta=f"{'+' if delta_s_medio > 0 else ''}{delta_s_medio:.1f} p.p.")
                
                # Antibiótico com maior variação
                max_delta_idx = df_validos["Δ%R"].abs().idxmax()
                abx_max_variacao = df_validos.loc[max_delta_idx, "Antibiótico"]
                max_variacao = df_validos.loc[max_delta_idx, "Δ%R"]
                
                st.metric("Maior Variação", abx_max_variacao, f"{max_variacao:+.1f} p.p.")


def _grafico_temporal_barras_duplas(df_comp: pd.DataFrame, metric: str, especie: str):
    """Cria gráfico de barras comparando 2025 vs anterior."""
    
    col_2025 = f"2025_%{metric}"
    col_anterior = f"Anterior_%{metric}"
    metric_label = f"% {metric}"
    
    if HAS_PLOTLY:
        # Preparar dados para Plotly
        antibioticos = df_comp["Antibiótico"].tolist()
        valores_2025 = df_comp[col_2025].tolist()
        valores_anterior = df_comp[col_anterior].tolist()
        
        fig = px.bar(
            x=antibioticos + antibioticos,
            y=valores_2025 + valores_anterior,
            color=["2025/Atual"] * len(antibioticos) + ["Anos Anteriores"] * len(antibioticos),
            title=f"{metric_label} - {especie}: Comparação Temporal",
            labels={"x": "Antibiótico", "y": metric_label},
            color_discrete_map={"2025/Atual": "#1f77b4", "Anos Anteriores": "#ff7f0e"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    else:
        import numpy as np
        
        x = np.arange(len(df_comp))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(max(10, len(df_comp) * 0.8), 6))
        
        bars1 = ax.bar(x - width/2, df_comp[col_2025], width, label="2025/Atual", color="#1f77b4", alpha=0.8)
        bars2 = ax.bar(x + width/2, df_comp[col_anterior], width, label="Anos Anteriores", color="#ff7f0e", alpha=0.8)
        
        ax.set_xlabel("Antibiótico")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} - {especie}: Comparação Temporal")
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp["Antibiótico"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars1, df_comp[col_2025]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, df_comp[col_anterior]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


def _grafico_variacao_temporal(df_comp: pd.DataFrame, especie: str):
    """Cria gráfico de variação temporal (delta)."""
    
    if HAS_PLOTLY:
        colors = ["red" if x > 0 else "green" for x in df_comp["Δ%R"]]
        
        fig = px.bar(
            df_comp,
            x="Antibiótico", 
            y="Δ%R",
            title=f"Variação da Resistência - {especie} (2025 vs Anterior)",
            labels={"Δ%R": "Variação %R (pontos percentuais)"},
            color="Δ%R",
            color_continuous_scale=["green", "white", "red"],
            color_continuous_midpoint=0
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_hline(y=5, line_dash="dot", line_color="red", opacity=0.3)
        fig.add_hline(y=-5, line_dash="dot", line_color="green", opacity=0.3)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(max(10, len(df_comp) * 0.6), 6))
        
        colors = ["red" if x > 0 else "green" for x in df_comp["Δ%R"]]
        bars = ax.bar(df_comp["Antibiótico"], df_comp["Δ%R"], color=colors, alpha=0.7)
        
        ax.set_xlabel("Antibiótico")
        ax.set_ylabel("Variação %R (pontos percentuais)")
        ax.set_title(f"Variação da Resistência - {especie} (2025 vs Anterior)")
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=5, color='red', linestyle=':', alpha=0.3)
        ax.axhline(y=-5, color='green', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha="right")
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, df_comp["Δ%R"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.5 if height > 0 else -0.5), 
                   f'{val:+.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        return fig


def _grafico_perfil_temporal_completo(df_2025: pd.DataFrame, df_anterior: pd.DataFrame, especie: str):
    """Cria gráfico empilhado comparando perfis completos."""
    
    if HAS_PLOTLY:
        # Combinar dados para gráfico lado a lado
        data_combined = []
        
        for _, row in df_2025.iterrows():
            abx = row["Antibiótico"]
            data_combined.extend([
                {"Período_Antibiótico": f"2025\n{abx}", "Período": "2025", "Antibiótico": abx,
                 "Categoria": "Sensível (S)", "Valor": row["%S"]},
                {"Período_Antibiótico": f"2025\n{abx}", "Período": "2025", "Antibiótico": abx,
                 "Categoria": "Intermediário (I/SSD)", "Valor": row["%I+SSD"]},
                {"Período_Antibiótico": f"2025\n{abx}", "Período": "2025", "Antibiótico": abx,
                 "Categoria": "Resistente (R)", "Valor": row["%R"]}
            ])
        
        for _, row in df_anterior.iterrows():
            abx = row["Antibiótico"]
            data_combined.extend([
                {"Período_Antibiótico": f"Anterior\n{abx}", "Período": "Anterior", "Antibiótico": abx,
                 "Categoria": "Sensível (S)", "Valor": row["%S"]},
                {"Período_Antibiótico": f"Anterior\n{abx}", "Período": "Anterior", "Antibiótico": abx,
                 "Categoria": "Intermediário (I/SSD)", "Valor": row["%I+SSD"]},
                {"Período_Antibiótico": f"Anterior\n{abx}", "Período": "Anterior", "Antibiótico": abx,
                 "Categoria": "Resistente (R)", "Valor": row["%R"]}
            ])
        
        df_plot = pd.DataFrame(data_combined)
        
        fig = px.bar(
            df_plot,
            x="Período_Antibiótico",
            y="Valor",
            color="Categoria",
            title=f"Perfil de Suscetibilidade Completo - {especie}",
            color_discrete_map={
                "Sensível (S)": "#2ca02c",
                "Intermediário (I/SSD)": "#1f77b4",
                "Resistente (R)": "#d62728"
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    else:
        # Versão matplotlib simplificada
        antibioticos_comuns = set(df_2025["Antibiótico"]) & set(df_anterior["Antibiótico"])
        
        if not antibioticos_comuns:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        # 2025
        df_2025_comum = df_2025[df_2025["Antibiótico"].isin(antibioticos_comuns)]
        x1 = np.arange(len(df_2025_comum))
        
        ax1.bar(x1, df_2025_comum["%S"], label="Sensível (S)", color="#2ca02c")
        ax1.bar(x1, df_2025_comum["%I+SSD"], bottom=df_2025_comum["%S"], 
               label="Intermediário (I/SSD)", color="#1f77b4")
        ax1.bar(x1, df_2025_comum["%R"], 
               bottom=df_2025_comum["%S"] + df_2025_comum["%I+SSD"], 
               label="Resistente (R)", color="#d62728")
        
        ax1.set_title("2025/Atual")
        ax1.set_xticks(x1)
        ax1.set_xticklabels(df_2025_comum["Antibiótico"], rotation=45, ha="right")
        ax1.set_ylabel("% de isolados")
        ax1.legend()
        
        # Anterior
        df_anterior_comum = df_anterior[df_anterior["Antibiótico"].isin(antibioticos_comuns)]
        x2 = np.arange(len(df_anterior_comum))
        
        ax2.bar(x2, df_anterior_comum["%S"], label="Sensível (S)", color="#2ca02c")
        ax2.bar(x2, df_anterior_comum["%I+SSD"], bottom=df_anterior_comum["%S"], 
               label="Intermediário (I/SSD)", color="#1f77b4")
        ax2.bar(x2, df_anterior_comum["%R"], 
               bottom=df_anterior_comum["%S"] + df_anterior_comum["%I+SSD"], 
               label="Resistente (R)", color="#d62728")
        
        ax2.set_title("Anos Anteriores")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(df_anterior_comum["Antibiótico"], rotation=45, ha="right")
        
        plt.suptitle(f"Perfil de Suscetibilidade Completo - {especie}")
        plt.tight_layout()
        return fig


def _grafico_comparativo_barras(df_comp: pd.DataFrame, metric: str, metric_label: str, 
                               colormap: str, especies: list[str]):
    """Cria gráfico de barras agrupadas para comparação entre espécies."""
    
    if HAS_PLOTLY:
        fig = px.bar(
            df_comp, 
            x="Antibiótico", 
            y=metric, 
            color="Espécie",
            title=f"{metric_label} por Antibiótico - Comparação entre Espécies",
            labels={metric: metric_label},
            text=metric
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    else:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Preparar dados para matplotlib
        antibioticos = df_comp["Antibiótico"].unique()
        x = np.arange(len(antibioticos))
        width = 0.8 / len(especies)
        
        fig, ax = plt.subplots(figsize=(max(10, len(antibioticos) * 0.8), 6))
        
        colors = plt.cm.get_cmap(colormap)(np.linspace(0.3, 0.9, len(especies)))
        
        for i, especie in enumerate(especies):
            df_esp = df_comp[df_comp["Espécie"] == especie]
            valores = [df_esp[df_esp["Antibiótico"] == abx][metric].iloc[0] if len(df_esp[df_esp["Antibiótico"] == abx]) > 0 else 0 for abx in antibioticos]
            
            bars = ax.bar(x + i * width - width * (len(especies) - 1) / 2, valores, 
                         width, label=especie, color=colors[i], alpha=0.8)
            
            # Adicionar rótulos nas barras
            for bar, val in zip(bars, valores):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Antibiótico')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} por Antibiótico - Comparação entre Espécies')
        ax.set_xticks(x)
        ax.set_xticklabels(antibioticos, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def _grafico_comparativo_empilhado(df_comp: pd.DataFrame, especies: list[str], antibioticos: list[str]):
    """Cria gráfico empilhado comparativo para múltiplas espécies."""
    
    if HAS_PLOTLY:
        # Preparar dados para gráfico empilhado comparativo
        data_for_plot = []
        
        for _, row in df_comp.iterrows():
            data_for_plot.extend([
                {"Espécie_Antibiótico": f"{row['Espécie']}\n{row['Antibiótico']}", 
                 "Espécie": row['Espécie'], "Antibiótico": row['Antibiótico'],
                 "Categoria": "Sensível (S)", "Valor": row['%S']},
                {"Espécie_Antibiótico": f"{row['Espécie']}\n{row['Antibiótico']}", 
                 "Espécie": row['Espécie'], "Antibiótico": row['Antibiótico'],
                 "Categoria": "Intermediário (I/SSD)", "Valor": row['%I+SSD']},
                {"Espécie_Antibiótico": f"{row['Espécie']}\n{row['Antibiótico']}", 
                 "Espécie": row['Espécie'], "Antibiótico": row['Antibiótico'],
                 "Categoria": "Resistente (R)", "Valor": row['%R']}
            ])
        
        df_plot = pd.DataFrame(data_for_plot)
        
        fig = px.bar(
            df_plot, 
            x="Espécie_Antibiótico", 
            y="Valor", 
            color="Categoria",
            title="Perfil de Suscetibilidade - Comparação Detalhada",
            color_discrete_map={
                "Sensível (S)": "#2ca02c",
                "Intermediário (I/SSD)": "#1f77b4", 
                "Resistente (R)": "#d62728"
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    else:
        # Versão matplotlib simplificada
        fig, axes = plt.subplots(1, len(especies), figsize=(5 * len(especies), 6), sharey=True)
        if len(especies) == 1:
            axes = [axes]
        
        for i, especie in enumerate(especies):
            df_esp = df_comp[df_comp["Espécie"] == especie]
            
            antibioticos_esp = df_esp["Antibiótico"].tolist()
            s_vals = df_esp["%S"].tolist()
            i_vals = df_esp["%I+SSD"].tolist()
            r_vals = df_esp["%R"].tolist()
            
            x = np.arange(len(antibioticos_esp))
            
            axes[i].bar(x, s_vals, label="Sensível (S)", color="#2ca02c")
            axes[i].bar(x, i_vals, bottom=s_vals, label="Intermediário (I/SSD)", color="#1f77b4")
            axes[i].bar(x, r_vals, bottom=np.array(s_vals) + np.array(i_vals), label="Resistente (R)", color="#d62728")
            
            axes[i].set_title(f"{especie}")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(antibioticos_esp, rotation=45, ha='right')
            axes[i].set_ylim(0, 100)
            
            if i == 0:
                axes[i].set_ylabel("% de isolados")
                axes[i].legend()
        
        plt.suptitle("Perfil de Suscetibilidade - Comparação entre Espécies")
        plt.tight_layout()
        return fig


def _render_especie_bloco(especie: str, df_esp: pd.DataFrame, sel_abx: list[str], col_maldi: str, col_codigo: str | None, titulo: str):
    """Renderiza a análise detalhada de uma espécie (tabela, downloads, gráficos e detalhado)."""
    n_isolados = len(df_esp)
    st.caption(f"Isolados desta espécie: {n_isolados}")
    if n_isolados == 0:
        return

    rows_esp = perfil_sri_rows(df_esp, sel_abx)
    df_rows_esp = format_df_numeric(rows_to_df(rows_esp))
    df_rows_esp = df_rows_esp[df_rows_esp["N"] > 0]
    if df_rows_esp.empty:
        st.info("Nenhum resultado de antibiótico para esta espécie.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(_style_susc_df(df_rows_esp), use_container_width=True)
    with col2:
        st.download_button(
            label=f"Baixar CSV - {especie}",
            data=to_csv_download(df_rows_esp),
            file_name=f"resistencia_{especie.replace(' ', '_')}_{titulo.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key=f"download_{especie}_{titulo.replace(' ', '_')}"
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Top %R por antibiótico")
        fig1_esp = fig_topR(df_rows_esp.values.tolist(), f"Resistência - {especie}", top=15)
        if fig1_esp is not None:
            if HAS_PLOTLY:
                st.plotly_chart(fig1_esp, use_container_width=True)
            else:
                st.pyplot(fig1_esp, use_container_width=True)
    with c4:
        st.markdown("#### Perfil de suscetibilidade")
        fig2_esp = fig_stacked(
            df_rows_esp.values.tolist(), f"Suscetibilidade - {especie}",
            ordenar_por="%R", label_threshold=5.0, label_decimals=1,
            label_fontsize=10, label_vertical=True,
            label_vertical_dir="up", show_labels=True
        )
        if fig2_esp is not None:
            if HAS_PLOTLY:
                st.plotly_chart(fig2_esp, use_container_width=True)
            else:
                st.pyplot(fig2_esp, use_container_width=True)

    with st.expander("Tabela detalhada por isolado (resultados brutos)", expanded=False):
        try:
            cols_det = [c for c in [col_codigo, col_maldi] if c and c in df_esp.columns] + [c for c in sel_abx if c in df_esp.columns]
            df_det = df_esp.loc[:, cols_det].copy()
            for c in sel_abx:
                if c in df_det.columns:
                    df_det[c] = df_det[c].apply(normaliza_status)
            st.dataframe(df_det, use_container_width=True)
            st.download_button(
                label=f"Baixar detalhado - {especie}",
                data=to_csv_download(df_det),
                file_name=f"isolados_{especie.replace(' ', '_')}_{titulo.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"download_det_{especie}_{titulo.replace(' ', '_')}"
            )
        except Exception:
            st.info("Não foi possível montar a tabela detalhada para esta espécie.")


def analise_por_especie(grp_df, titulo, col_maldi, sel_abx, col_codigo: str | None = None, min_n_especie: int = 5):
    """Analisa suscetibilidade por espécie dentro de um grupo temporal (refatorado)."""
    st.subheader(f"Análise por Espécie - {titulo}")

    if col_maldi not in grp_df.columns:
        st.warning("Coluna MALDI-TOF não encontrada.")
        return

    df_especie = _valid_species_df(grp_df, col_maldi)
    if df_especie.empty:
        st.info("Sem dados válidos de espécies para este grupo.")
        return

    especies_count = df_especie[col_maldi].value_counts()
    st.caption(f"Total de isolados com espécie identificada: {len(df_especie)}")

    # Controle do N mínimo por espécie
    base_key = titulo.replace(' ', '_')
    min_n_especie = st.slider(
        f"Mínimo de isolados por espécie ({titulo})",
        min_value=1, max_value=50, value=int(min_n_especie),
        key=f"min_especie_{base_key}",
        help=(
            "Padrão = 5. Com N muito baixo, os percentuais oscilam demais: "
            "1 caso muda 100%, N=3 muda ~33 p.p., N=5 muda 20 p.p., N=10 muda 10 p.p. "
            "Usar 5 reduz a instabilidade sem ocultar completamente espécies menos frequentes."
        ),
    )

    especies_validas = especies_count[especies_count >= min_n_especie].index.tolist()
    if not especies_validas:
        st.warning(f"Nenhuma espécie tem pelo menos {min_n_especie} isolados.")
        return
    st.write(f"Espécies com ≥ {min_n_especie} isolados: {len(especies_validas)}")

    # Destaque opcional
    st.markdown("### Destaque uma espécie específica")
    show_sel = st.checkbox(
        f"Mostrar seleção de espécie ({titulo})",
        value=False,
        key=f"toggle_sel_{base_key}"
    )
    if show_sel:
        incluir_fora_min = st.checkbox(
            "Incluir espécies com N < mínimo",
            value=True,
            key=f"toggle_incluir_{base_key}",
            help="Inclui espécies abaixo do limite de isolados mínimos na lista de opções."
        )
        options_base = sorted(df_especie[col_maldi].astype(str).str.strip().unique().tolist()) if incluir_fora_min else especies_validas
        options = sorted(options_base, key=lambda esp: int(especies_count.get(esp, 0)), reverse=True)
        label_map = {esp: f"{esp} — N={int(especies_count.get(esp, 0))}" for esp in options}
        if not options:
            st.info("Nenhuma espécie disponível para seleção com os filtros atuais.")
        else:
            especie_dest = st.selectbox(
                "Selecione a espécie para destacar",
                options=["— selecione —"] + options,
                index=0,
                format_func=lambda x: label_map.get(x, x),
                key=f"toggle_select2_{base_key}"
            )
            st.caption("Dica: comece a digitar para filtrar a lista.")
            if especie_dest and especie_dest != "— selecione —":
                df_esp_b = df_especie[df_especie[col_maldi] == especie_dest].copy()
                st.markdown(f"#### Espécie destacada: {especie_dest}")
                _render_especie_bloco(especie_dest, df_esp_b, sel_abx, col_maldi, col_codigo, titulo)
                st.divider()
            else:
                st.caption("Escolha uma espécie para ver o destaque.")

    # Seletor principal e opção de mostrar todas
    especies_validas_sorted = sorted(especies_validas, key=lambda esp: int(especies_count.get(esp, 0)), reverse=True)
    label_map2 = {esp: f"{esp} — N={int(especies_count.get(esp, 0))}" for esp in especies_validas_sorted}
    mostrar_todas = st.checkbox(
        f"Mostrar todas as espécies válidas ({titulo})",
        value=False,
        key=f"mostrar_todas_{base_key}"
    )
    especie_selecionada = st.selectbox(
        f"Selecione uma espécie para análise detalhada ({titulo})",
        ["— selecione —"] + especies_validas_sorted,
        index=0,
        format_func=lambda x: label_map2.get(x, x),
        key=f"select_especie2_{base_key}"
    )
    st.caption("Dica: digite parte do nome para buscar. Ordenado por frequência (N).")

    especies_para_mostrar = especies_validas_sorted if mostrar_todas else ([] if especie_selecionada == "— selecione —" else [especie_selecionada])

    if len(especies_para_mostrar) == 0:
        st.info("Nenhuma espécie selecionada. Escolha uma para ver a análise.")
    for i, especie in enumerate(especies_para_mostrar):
        st.markdown(f"### {especie}" if len(especies_para_mostrar) > 1 else f"### Espécie selecionada: {especie}")
        df_esp = df_especie[df_especie[col_maldi] == especie].copy()
        _render_especie_bloco(especie, df_esp, sel_abx, col_maldi, col_codigo, titulo)
        if len(especies_para_mostrar) > 1 and i < len(especies_para_mostrar) - 1:
            st.divider()

    # Resumo comparativo
    st.markdown("### Resumo Comparativo entre Espécies")
    df_resumo = _build_resumo(df_especie[df_especie[col_maldi].isin(especies_validas)], col_maldi, sel_abx)
    if df_resumo.empty:
        st.info("Sem dados para montar o resumo comparativo.")
    else:
        st.dataframe(
            df_resumo.style
            .background_gradient(subset=["% Resistência Média"], cmap="Reds")
            .background_gradient(subset=["% Sensibilidade Média"], cmap="Greens"),
            use_container_width=True
        )
        st.download_button(
            label="Baixar Resumo Comparativo (CSV)",
            data=to_csv_download(df_resumo),
            file_name=f"resumo_especies_{titulo.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key=f"download_resumo_{base_key}"
        )

    # Nota metodológica
    st.divider()
    with st.expander(f"Nota metodológica — Análise por espécie ({titulo})", expanded=False):
        st.markdown(
            """
            O que é:
            - Analisa perfis de suscetibilidade (S, I/SSD, R) por antibiótico dentro de cada espécie identificada por MALDI-TOF no grupo selecionado.

            Como foi feito:
            - Considera apenas registros com espécie válida (não vazia e diferente de "*").
            - Aplica um limite mínimo de isolados por espécie (controle no slider) para evitar conclusões com amostras muito pequenas.
            - Para cada antibiótico na espécie: normaliza o resultado da célula (primeiro rótulo em {SSD, R, S, I}), conta N válidos e calcula %R, %S e %I+SSD sobre N.
            - Exibe: tabela por antibiótico; Top %R; gráfico empilhado de %S, %I+SSD e %R; e um resumo comparativo entre espécies.
            - O resumo usa médias ponderadas pelo número de testes (N) para %R e %S de cada antibiótico na espécie.

            Por que o mínimo padrão é 5 isolados?
            - Percentuais com N muito baixo são instáveis (basta 1 amostra alterar muito o resultado):
              • N=1 → 0% ou 100%; N=3 → ~33 p.p.; N=5 → 20 p.p.; N=10 → 10 p.p.
            - 5 é um compromisso: reduz a volatilidade sem esconder espécies menos frequentes.
            - Ajuste conforme o objetivo: use 3 para exploração inicial de espécies raras; use 10+ quando precisar de resultados mais estáveis/robustos.

            Interpretação e limites:
            - Cobertura baixa (poucos testes em um antibiótico) pode distorcer percentuais; use os filtros de N mínimo e cobertura.
            - "I" e "SSD" são combinados no gráfico como I/SSD; as contagens são mostradas separadamente na tabela.
            - Células com anotações variadas (ex.: "R *", "SSD –") são interpretadas pelo primeiro rótulo válido.

            Fórmulas:
            - %R = 100 * R / N; %S = 100 * S / N; %I+SSD = 100 * (I + SSD) / N; Cobertura = 100 * N / Total.
            """
        )

# ---------------------------
# UI
# ---------------------------
st.title("Análise de Suscetibilidade de Isolados")
st.caption("Carregue o arquivo Excel e explore métricas por antibiótico com gráficos e tabelas interativas.")

with st.sidebar:
    st.header("Configurações")
    # Escolha da fonte de dados
    data_source = st.radio("Fonte dos dados", ["Arquivo padrão (Cepas.xlsx)", "Upload de arquivo"], index=0)
    up = None
    if data_source == "Upload de arquivo":
        up = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx"])
    header_row = st.number_input("Linha do cabeçalho (1 = primeira linha)", min_value=1, value=2)
    codigo_limite = st.number_input("Limite numérico p/ 2025 (ex.: MA >= 181)", min_value=0, value=181)
    top_n = st.slider("Top N por %R (gráfico horizontal)", 5, 30, 15)
    ordenar_por = st.selectbox("Ordenar gráfico empilhado por", ["%R", "%S", "%Cobertura", "Antibiótico"])
    min_n = st.slider("N mínimo testado (filtro)", 0, 100, 0)
    min_cob = st.slider("Cobertura mínima %", 0, 100, 0)
    usar_plotly = st.checkbox("Forçar gráficos interativos (Plotly)", value=HAS_PLOTLY)
    st.subheader("Rótulos das barras (empilhado)")
    show_labels = st.checkbox("Exibir rótulos internos (vertical)", value=True)
    label_threshold = st.slider("Mostrar rótulo a partir de (%)", 0, 20, 5, disabled=not show_labels)
    label_decimals = st.select_slider("Casas decimais", options=[0,1,2], value=1, disabled=not show_labels)
    label_fontsize = st.slider("Tamanho do texto", 8, 18, 11, disabled=not show_labels)
    label_vertical_dir = st.selectbox("Direção do rótulo vertical", ["Baixo -> cima", "Cima -> baixo"], index=0, disabled=not show_labels)
    st.divider()
    st.subheader("Espécies (MALDI-TOF)")
    especies_chart_tipo = st.selectbox("Tipo de gráfico", ["Barra", "Pizza"], index=0)
    especies_top_n = st.slider("Top N espécies", 3, 20, 10)
    especies_agrupar_outros = st.checkbox("Agrupar demais em 'Outros'", value=True)

# Define o arquivo a ser lido
DEFAULT_XLSX = Path("Cepas.xlsx")
file_to_read = None
source_label = ""

if data_source == "Arquivo padrão (Cepas.xlsx)":
    if DEFAULT_XLSX.exists():
        file_to_read = DEFAULT_XLSX
        source_label = f"Arquivo padrão: {DEFAULT_XLSX.name}"
    else:
        st.warning("Arquivo padrão 'Cepas.xlsx' não encontrado na pasta do app. Selecione 'Upload de arquivo'.")
        st.stop()
else:
    if up is None:
        st.info("Envie um arquivo Excel (.xlsx). O app espera cabeçalho na 2ª linha por padrão.")
        st.stop()
    file_to_read = up
    source_label = f"Upload: {getattr(up, 'name', 'arquivo')}"

# ---------------------------
# Leitura do Excel
# ---------------------------
try:
    df = pd.read_excel(file_to_read, dtype=object, header=int(header_row - 1))
except Exception as e:
    st.error(f"Erro ao ler a planilha: {e}")
    st.stop()

df.columns = [norm_str(c) for c in df.columns]
st.caption(f"Fonte dos dados: {source_label}")

col_codigo = achar_col(df, r"^CÓDIGO\s+UFPB$")
if not col_codigo:
    st.error("Coluna 'CÓDIGO UFPB' não encontrada. Verifique o cabeçalho correto.")
    st.dataframe(pd.DataFrame({"Colunas encontradas": df.columns}))
    st.stop()

col_maldi = achar_col(df, r"MALDLI-TOF|MALDI-TOF")
abx_cols_existentes = [c for c in ABX_LEGENDA if c in df.columns]
if not abx_cols_existentes:
    st.error("Nenhuma coluna de antibiótico da legenda foi encontrada no arquivo.")
    st.write("Colunas esperadas (exemplos):", ABX_LEGENDA)
    st.write("Colunas do arquivo:", list(df.columns))
    st.stop()

# Seleção dos antibióticos
st.sidebar.subheader("Antibióticos")
sel_abx = st.sidebar.multiselect("Selecionar colunas de antibióticos", options=abx_cols_existentes, default=abx_cols_existentes)
if not sel_abx:
    st.warning("Selecione pelo menos um antibiótico.")
    st.stop()

# Calcula partições
df["_NUM"] = df[col_codigo].apply(extrai_num_codigo)
grp_2025 = df[df["_NUM"] >= codigo_limite].copy()
grp_prev = df[(df["_NUM"] < codigo_limite) & (~df["_NUM"].isna())].copy()

tabs = st.tabs(["2025 / atual", "Anos anteriores", "Espécies (MALDI-TOF)", "Análise por Espécie - 2025/atual", "Análise por Espécie - Anos anteriores", "Comparação Temporal por Espécie", "📖 Guia de Uso"])

# Helper para uma aba de análise
def render_tab(grp_df, titulo):
    st.subheader(titulo)
    total_iso = len(grp_df)
    st.caption(f"Total de isolados no grupo: {total_iso}")
    if total_iso == 0:
        st.info("Sem registros neste grupo.")
        return

    rows = perfil_sri_rows(grp_df, sel_abx)
    df_rows = format_df_numeric(rows_to_df(rows))
    # Filtros
    df_rows = df_rows[(df_rows["N"] >= min_n) & (df_rows["%Cobertura"] >= min_cob)]
    if df_rows.empty:
        st.warning("Nenhum antibiótico atende aos filtros aplicados.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Tabela de suscetibilidade")
        st.dataframe(
            df_rows.style
            .background_gradient(subset=["%R"], cmap="Reds")
            .background_gradient(subset=["%S"], cmap="Greens")
            .format({"%R": "{:.1f}", "%S": "{:.1f}", "%I+SSD": "{:.1f}", "%Cobertura": "{:.1f}"})
        , use_container_width=True)
    with c2:
        st.download_button(
            label="Baixar CSV",
            data=to_csv_download(df_rows),
            file_name=f"perfil_suscetibilidade_{titulo.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Top %R por antibiótico")
        fig1 = fig_topR(df_rows.values.tolist(), f"Top %R - {titulo}", top=top_n)
        if fig1 is not None:
            if HAS_PLOTLY and usar_plotly:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.pyplot(fig1, use_container_width=True)
    with c4:
        st.markdown("### Suscetibilidade dos isolados (empilhado)")
        fig2 = fig_stacked(
            df_rows.values.tolist(), f"Suscetibilidade - {titulo}", ordenar_por=ordenar_por,
            label_threshold=label_threshold, label_decimals=label_decimals,
            label_fontsize=label_fontsize, label_vertical=show_labels,
            label_vertical_dir=("up" if label_vertical_dir.startswith("Baixo") else "down"),
            show_labels=show_labels
        )
        if fig2 is not None:
            if HAS_PLOTLY and usar_plotly:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.pyplot(fig2, use_container_width=True)


with tabs[0]:
    render_tab(grp_2025, "2025 / atual")

with tabs[1]:
    render_tab(grp_prev, "Anos anteriores")

with tabs[2]:
    st.subheader("Distribuição de espécies (MALDI-TOF)")
    if col_maldi and col_maldi in df.columns:
        esp = df[col_maldi].dropna().astype(str).str.strip()
        esp = esp[(esp != "") & (esp != "*")]
        if not esp.empty:
            dist = esp.value_counts().sort_values(ascending=False)
            total = int(dist.sum())
            # Preparar Top N + 'Outros' se habilitado
            if especies_agrupar_outros and len(dist) > especies_top_n:
                top = dist.head(especies_top_n)
                outros_val = int(dist.iloc[especies_top_n:].sum())
                dist_plot = top.append(pd.Series({"Outros": outros_val}))
            else:
                dist_plot = dist
            dist_df = dist_plot.reset_index()
            dist_df.columns = ["Espécie", "Frequência"]
            dist_df["%"] = (dist_df["Frequência"] / dist_df["Frequência"].sum() * 100).round(1)

            st.dataframe(dist_df, use_container_width=True)

            if especies_chart_tipo in ("Pizza"):
                if HAS_PLOTLY and usar_plotly:
                    fig = px.pie(dist_df, values="Frequência", names="Espécie",
                                  hole=0.4, title="Distribuição de espécies (MALDI-TOF)")
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    # Aumenta levemente o tamanho do gráfico (altura); largura segue o container
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Aumenta levemente o tamanho do gráfico em polegadas (largura x altura)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    wedges, texts, autotexts = ax.pie(
                        dist_df["Frequência"], labels=dist_df["Espécie"], autopct='%1.1f%%',
                        startangle=90, wedgeprops=dict(width=0.4)
                    )
                    ax.set_title("Distribuição de espécies (MALDI-TOF)")
                    ax.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            else:
                if HAS_PLOTLY and usar_plotly:
                    fig = px.bar(dist_df, x="Frequência", y="Espécie", orientation="h",
                                 title="Distribuição de espécies (MALDI-TOF)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots()
                    ax.barh(dist_plot.index[::-1], dist_plot.values[::-1])
                    ax.set_title("Distribuição de espécies (MALDI-TOF)")
                    ax.set_xlabel("Frequência")
                    ax.set_ylabel("Espécie")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
        else:
            st.info("Sem valores válidos em MALDI-TOF.")
    else:
        st.warning("Coluna MALDI-TOF não encontrada; análise de espécies omitida.")

with tabs[3]:
    analise_por_especie(grp_2025, "2025/atual", col_maldi, sel_abx, col_codigo)

with tabs[4]:
    analise_por_especie(grp_prev, "Anos anteriores", col_maldi, sel_abx, col_codigo)

with tabs[5]:
    st.subheader("⏰ Comparação Temporal por Espécie")
    st.caption("Compare a mesma espécie entre diferentes períodos temporais (2025/atual vs Anos anteriores)")
    
    if not col_maldi or col_maldi not in df.columns:
        st.warning("Coluna MALDI-TOF não encontrada. Comparação temporal não disponível.")
    else:
        # Verificar se temos dados em ambos os períodos
        df_2025_validas = _valid_species_df(grp_2025, col_maldi)
        df_anterior_validas = _valid_species_df(grp_prev, col_maldi)
        
        if df_2025_validas.empty and df_anterior_validas.empty:
            st.warning("Nenhuma espécie válida encontrada em ambos os períodos.")
        else:
            # Encontrar espécies que existem em ambos os períodos
            especies_2025 = set(df_2025_validas[col_maldi].unique()) if not df_2025_validas.empty else set()
            especies_anterior = set(df_anterior_validas[col_maldi].unique()) if not df_anterior_validas.empty else set()
            especies_comuns = especies_2025 & especies_anterior
            
            # Estatísticas dos períodos
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("📅 2025/Atual", 
                         f"{len(df_2025_validas)} isolados", 
                         f"{len(especies_2025)} espécies")
            
            with col_info2:
                st.metric("📆 Anos Anteriores", 
                         f"{len(df_anterior_validas)} isolados", 
                         f"{len(especies_anterior)} espécies")
            
            with col_info3:
                st.metric("🔗 Espécies em Comum", 
                         f"{len(especies_comuns)} espécies", 
                         "Para comparação")
            
            if not especies_comuns:
                st.warning("Nenhuma espécie encontrada em ambos os períodos para comparação.")
                
                # Mostrar quais espécies existem apenas em cada período
                with st.expander("Ver espécies disponíveis por período", expanded=False):
                    col_only1, col_only2 = st.columns(2)
                    
                    with col_only1:
                        st.markdown("**Apenas em 2025/Atual:**")
                        especies_apenas_2025 = especies_2025 - especies_anterior
                        if especies_apenas_2025:
                            for esp in sorted(especies_apenas_2025):
                                count = len(df_2025_validas[df_2025_validas[col_maldi] == esp])
                                st.write(f"• {esp} (N={count})")
                        else:
                            st.write("Nenhuma")
                    
                    with col_only2:
                        st.markdown("**Apenas em Anos Anteriores:**")
                        especies_apenas_anterior = especies_anterior - especies_2025
                        if especies_apenas_anterior:
                            for esp in sorted(especies_apenas_anterior):
                                count = len(df_anterior_validas[df_anterior_validas[col_maldi] == esp])
                                st.write(f"• {esp} (N={count})")
                        else:
                            st.write("Nenhuma")
            else:
                # Configurações da comparação temporal
                st.markdown("### ⚙️ Configurações da Análise")
                
                col_config1, col_config2 = st.columns([1, 1])
                
                with col_config1:
                    # Filtro de N mínimo
                    min_n_temporal = st.slider(
                        "N mínimo por período",
                        min_value=1, max_value=30, value=3,
                        key="min_n_temporal",
                        help="Espécie deve ter pelo menos este número de isolados em CADA período."
                    )
                    
                    # Filtrar espécies que atendem ao critério em ambos os períodos
                    especies_validas_temporal = []
                    for esp in especies_comuns:
                        n_2025 = len(df_2025_validas[df_2025_validas[col_maldi] == esp])
                        n_anterior = len(df_anterior_validas[df_anterior_validas[col_maldi] == esp])
                        if n_2025 >= min_n_temporal and n_anterior >= min_n_temporal:
                            especies_validas_temporal.append(esp)
                    
                    st.caption(f"Espécies válidas para comparação: {len(especies_validas_temporal)}")
                
                with col_config2:
                    # Seleção de antibióticos
                    abx_temporal = st.multiselect(
                        "Antibióticos para análise temporal",
                        options=sel_abx,
                        default=sel_abx,  # Todos os antibióticos por padrão
                        key="abx_temporal",
                        help="Antibióticos incluídos na comparação temporal. Por padrão, todos estão selecionados - remova os que não deseja analisar."
                    )
                
                if not especies_validas_temporal:
                    st.warning(f"Nenhuma espécie tem pelo menos {min_n_temporal} isolados em ambos os períodos.")
                elif not abx_temporal:
                    st.warning("Selecione pelo menos um antibiótico para a análise.")
                else:
                    # Seleção da espécie para análise detalhada
                    st.markdown("### 🎯 Seleção da Espécie")
                    
                    # Criar labels informativos
                    especies_labels_temporal = {}
                    for esp in especies_validas_temporal:
                        n_2025 = len(df_2025_validas[df_2025_validas[col_maldi] == esp])
                        n_anterior = len(df_anterior_validas[df_anterior_validas[col_maldi] == esp])
                        especies_labels_temporal[esp] = f"{esp} (2025: N={n_2025}, Anterior: N={n_anterior})"
                    
                    # Ordenar por total de isolados
                    especies_ordenadas = sorted(especies_validas_temporal, 
                                              key=lambda x: len(df_2025_validas[df_2025_validas[col_maldi] == x]) + 
                                                           len(df_anterior_validas[df_anterior_validas[col_maldi] == x]), 
                                              reverse=True)
                    
                    especie_temporal_selecionada = st.selectbox(
                        "Selecione a espécie para comparação temporal detalhada",
                        options=["— selecione —"] + especies_ordenadas,
                        index=0,
                        format_func=lambda x: especies_labels_temporal.get(x, x),
                        key="especie_temporal_select",
                        help="Escolha uma espécie para ver a evolução temporal detalhada."
                    )
                    
                    # Opção de análise rápida de todas as espécies
                    mostrar_resumo_geral = st.checkbox(
                        "📊 Mostrar resumo de todas as espécies válidas",
                        value=False,
                        key="resumo_geral_temporal",
                        help="Exibe uma tabela resumo com as principais métricas de todas as espécies válidas."
                    )
                    
                    # Análise detalhada da espécie selecionada
                    if especie_temporal_selecionada and especie_temporal_selecionada != "— selecione —":
                        _comparativo_temporal_especie(
                            df_2025_validas, 
                            df_anterior_validas, 
                            col_maldi, 
                            especie_temporal_selecionada, 
                            abx_temporal
                        )
                    
                    # Resumo geral de todas as espécies
                    if mostrar_resumo_geral:
                        st.markdown("### 📋 Resumo Geral - Todas as Espécies")
                        
                        resumo_geral = []
                        for esp in especies_validas_temporal:
                            df_2025_esp = df_2025_validas[df_2025_validas[col_maldi] == esp]
                            df_anterior_esp = df_anterior_validas[df_anterior_validas[col_maldi] == esp]
                            
                            # Calcular médias de resistência
                            rows_2025 = perfil_sri_rows(df_2025_esp, abx_temporal)
                            rows_anterior = perfil_sri_rows(df_anterior_esp, abx_temporal)
                            
                            if rows_2025 and rows_anterior:
                                df_2025_perfil = format_df_numeric(rows_to_df(rows_2025))
                                df_anterior_perfil = format_df_numeric(rows_to_df(rows_anterior))
                                
                                # Médias ponderadas
                                if not df_2025_perfil.empty and df_2025_perfil["N"].sum() > 0:
                                    resistencia_2025 = (df_2025_perfil["%R"] * df_2025_perfil["N"]).sum() / df_2025_perfil["N"].sum()
                                else:
                                    resistencia_2025 = 0
                                
                                if not df_anterior_perfil.empty and df_anterior_perfil["N"].sum() > 0:
                                    resistencia_anterior = (df_anterior_perfil["%R"] * df_anterior_perfil["N"]).sum() / df_anterior_perfil["N"].sum()
                                else:
                                    resistencia_anterior = 0
                                
                                delta_resistencia = resistencia_2025 - resistencia_anterior
                                
                                resumo_geral.append({
                                    "Espécie": esp,
                                    "N_2025": len(df_2025_esp),
                                    "N_Anterior": len(df_anterior_esp),
                                    "%R_2025": round(resistencia_2025, 1),
                                    "%R_Anterior": round(resistencia_anterior, 1),
                                    "Δ%R": round(delta_resistencia, 1),
                                    "Tendência": "↑" if delta_resistencia > 5 else "↓" if delta_resistencia < -5 else "→"
                                })
                        
                        if resumo_geral:
                            df_resumo_geral = pd.DataFrame(resumo_geral)
                            df_resumo_geral = df_resumo_geral.sort_values("Δ%R", ascending=False)
                            
                            # Aplicar cores baseadas na tendência
                            def color_tendencia(val):
                                if val == "↑":
                                    return 'background-color: #ffcccc'  # vermelho claro
                                elif val == "↓":
                                    return 'background-color: #ccffcc'  # verde claro
                                return ''
                            
                            st.dataframe(
                                df_resumo_geral.style
                                .applymap(color_tendencia, subset=["Tendência"])
                                .background_gradient(subset=["Δ%R"], cmap="RdYlGn_r")
                                .format({"%R_2025": "{:.1f}", "%R_Anterior": "{:.1f}", "Δ%R": "{:.1f}"}),
                                use_container_width=True
                            )
                            
                            st.download_button(
                                label="📥 Baixar Resumo Geral (CSV)",
                                data=to_csv_download(df_resumo_geral),
                                file_name="resumo_temporal_todas_especies.csv",
                                mime="text/csv",
                                key="download_resumo_geral"
                            )
                            
                            st.caption("📊 **Legenda**: ↑ = Aumento >5 p.p. | ↓ = Diminuição >5 p.p. | → = Estável (±5 p.p.)")
    
    # Nota metodológica
    st.divider()
    with st.expander("ℹ️ Metodologia da Comparação Temporal", expanded=False):
        st.markdown("""
        **Objetivo:**
        - Comparar a evolução da suscetibilidade da mesma espécie entre dois períodos temporais
        - Identificar tendências de aumento/diminuição da resistência ao longo do tempo
        - Avaliar quais antibióticos apresentaram maior variação temporal
        
        **Como funciona:**
        - Filtra espécies que existem em ambos os períodos (2025/atual e anos anteriores)
        - Aplica critério de N mínimo para cada período (garantindo representatividade)
        - Calcula %R, %S, %I+SSD para cada antibiótico em cada período
        - Computa variações (Δ) = 2025 - Anterior
        - Classifica tendências: ↑ aumento >5 p.p., ↓ diminuição >5 p.p., → estável ±5 p.p.
        
        **Interpretação:**
        - **Δ%R positivo**: aumento da resistência no período mais recente
        - **Δ%R negativo**: diminuição da resistência (melhora)
        - **Valores próximos de zero**: padrão estável entre períodos
        - **N mínimo**: evita conclusões baseadas em amostras muito pequenas
        
        **Limitações:**
        - Comparação válida apenas para espécies com dados suficientes em ambos os períodos
        - Diferenças metodológicas ou de coleta entre períodos podem influenciar resultados
        - Variações podem refletir mudanças na população estudada, não necessariamente evolução da resistência
        
        **Aplicações:**
        - Vigilância epidemiológica de resistência antimicrobiana
        - Avaliação de efetividade de políticas de controle de infecção
        - Identificação de antibióticos com perda de eficácia ao longo do tempo
        - Suporte à atualização de protocolos terapêuticos institucionais
        """)

with tabs[6]:
    st.header("📖 Guia Completo de Uso da Aplicação")
    st.caption("Manual passo a passo para aproveitar ao máximo todas as funcionalidades")
    
    # Índice navegável
    st.markdown("## 📋 Índice")
    st.caption("Clique nos botões abaixo para navegar rapidamente para cada seção")
    
    # Criando botões de navegação organizados
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        st.markdown("**🚀 Primeiros Passos**")
        if st.button("1️⃣ Carregamento de Dados", key="nav_1", use_container_width=True):
            st.session_state.scroll_to = "section_1"
            st.rerun()
        if st.button("2️⃣ Configurações Iniciais", key="nav_2", use_container_width=True):
            st.session_state.scroll_to = "section_2"
            st.rerun()
        if st.button("3️⃣ Seleção de Antibióticos", key="nav_3", use_container_width=True):
            st.session_state.scroll_to = "section_3"
            st.rerun()
        
        st.markdown("**📊 Análises Básicas**")
        if st.button("4️⃣ Análise 2025/Atual", key="nav_4", use_container_width=True):
            st.session_state.scroll_to = "section_4"
            st.rerun()
        if st.button("5️⃣ Análise Anos Anteriores", key="nav_5", use_container_width=True):
            st.session_state.scroll_to = "section_5"
            st.rerun()
        if st.button("6️⃣ Distribuição de Espécies", key="nav_6", use_container_width=True):
            st.session_state.scroll_to = "section_6"
            st.rerun()
    
    with col_nav2:
        st.markdown("**🔬 Análises Avançadas**")
        if st.button("7️⃣ Análise por Espécie", key="nav_7", use_container_width=True):
            st.session_state.scroll_to = "section_7"
            st.rerun()
        if st.button("8️⃣ Comparação Temporal", key="nav_8", use_container_width=True):
            st.session_state.scroll_to = "section_8"
            st.rerun()
    
    with col_nav3:
        st.markdown("**🛠️ Recursos Adicionais**")
        if st.button("9️⃣ Exportação de Dados", key="nav_9", use_container_width=True):
            st.session_state.scroll_to = "section_9"
            st.rerun()
        if st.button("🔟 Interpretação de Resultados", key="nav_10", use_container_width=True):
            st.session_state.scroll_to = "section_10"
            st.rerun()
        if st.button("1️⃣1️⃣ Solução de Problemas", key="nav_11", use_container_width=True):
            st.session_state.scroll_to = "section_11"
            st.rerun()
    
    # Botão para mostrar todas as seções
    col_all1, col_all2, col_all3 = st.columns([1, 1, 1])
    with col_all2:
        if st.button("📖 Mostrar Todas as Seções", key="nav_all", use_container_width=True, type="primary"):
            if 'scroll_to' in st.session_state:
                del st.session_state.scroll_to
            st.rerun()
    
    st.divider()
    
    # Sistema de scroll automático
    if 'scroll_to' in st.session_state:
        target_section = st.session_state.scroll_to
        
        # Criar âncora visual para a seção selecionada
        section_names = {
            "section_1": "1. Carregamento de Dados",
            "section_2": "2. Configurações Iniciais", 
            "section_3": "3. Seleção de Antibióticos",
            "section_4": "4. Análise 2025/Atual",
            "section_5": "5. Análise Anos Anteriores",
            "section_6": "6. Distribuição de Espécies",
            "section_7": "7. Análise por Espécie",
            "section_8": "8. Comparação Temporal",
            "section_9": "9. Exportação de Dados",
            "section_10": "10. Interpretação de Resultados",
            "section_11": "11. Solução de Problemas"
        }
        
        if target_section in section_names:
            st.success(f"🎯 **Navegando para:** {section_names[target_section]}")
            st.markdown("---")
    
    # Função helper para controle de exibição
    def should_show_section(section_id):
        if 'scroll_to' in st.session_state:
            return st.session_state.scroll_to == section_id
        return True  # Mostra todas se não há navegação específica
    
    # Seção 1: Carregamento de Dados
    if should_show_section("section_1"):
        st.markdown("## 1. Carregamento de Dados")
        st.markdown("### 🎯 **Como começar:**")
        
        with st.expander("📁 Opções de Fonte de Dados", expanded=True):
            st.markdown("""
            **Opção 1: Arquivo Padrão (Recomendado)**
            - Use se você tem o arquivo `Cepas.xlsx` na pasta da aplicação
            - ✅ Automático e rápido
            - ✅ Não requer upload
            
            **Opção 2: Upload de Arquivo**
            - Use para analisar outros arquivos Excel
            - ✅ Flexível para diferentes datasets
            - ⚠️ Deve ter estrutura similar ao padrão
            
            **⚙️ Configurações Importantes:**
            - **Linha do cabeçalho**: Geralmente linha 2 (padrão)
            - **Limite para 2025**: MA181 ou superior (ajuste conforme seu critério)
            """)
    
    # Seção 2: Configurações Iniciais
    if should_show_section("section_2"):
        st.markdown("## 2. Configurações Iniciais")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.markdown("### ⚙️ **Barra Lateral - Configurações Principais**")
            st.markdown("""
            **📊 Filtros de Qualidade:**
            - **N mínimo testado**: Remove antibióticos com poucos testes
            - **Cobertura mínima %**: Remove antibióticos pouco testados
            - 💡 *Comece com valores baixos (0) e aumente conforme necessário*
            
            **📈 Configurações de Gráficos:**
            - **Top N por %R**: Quantos antibióticos mostrar no ranking
            - **Ordenar por**: Como ordenar o gráfico empilhado
            - **Gráficos interativos**: Liga/desliga Plotly (recomendado: ligado)
            """)
        
        with col_config2:
            st.markdown("### 🎨 **Personalização Visual**")
            st.markdown("""
            **📊 Rótulos das Barras:**
            - **Exibir rótulos**: Mostra percentuais dentro das barras
            - **Limiar de exibição**: Só mostra se % for maior que valor
            - **Casas decimais**: Precisão dos números
            - **Tamanho e direção**: Customização visual
            
            **🧬 Espécies (MALDI-TOF):**
            - **Tipo de gráfico**: Barra horizontal ou pizza
            - **Top N**: Quantas espécies mostrar
            - **Agrupar outros**: Combina espécies menos frequentes
            """)
    
    # Seção 3: Seleção de Antibióticos
    if should_show_section("section_3"):
        st.markdown("## 3. Seleção de Antibióticos")
        
        with st.expander("💊 Como Escolher Antibióticos", expanded=True):
            st.markdown("""
            **📋 Lista Disponível:**
            `GEN`, `TOB`, `AMI`, `ATM`, `CRO`, `CAZ`, `CTX`, `CFO`, `CPM`, `AMC`, `AMP`, `PPT`, `CZA`, `MER`, `IMP`, `CIP`, `LEV`, `SUT`, `POLI B`
            
            **✅ Estratégias de Seleção:**
            
            **Para Análise Completa:**
            - Selecione todos os disponíveis
            - Use filtros de N mínimo para remover os irrelevantes
            
            **Para Análise Focada:**
            - Selecione apenas sua classe de interesse (ex: só betalactâmicos)
            - Útil para apresentações específicas
            
            **Para Comparação:**
            - Mantenha a mesma seleção entre análises
            - Facilita comparações consistentes
            
            **⚠️ Importante:** Só aparecem antibióticos que existem no seu arquivo Excel
            """)
    
    
    # Seção 4: Análise 2025/Atual
    if should_show_section("section_4"):
        st.markdown("## 4. Análise 2025/Atual")
    
    with st.expander("📊 Análise do Período Atual (2025) - Passo a Passo", expanded=True):
        st.markdown("""
        ### 🎯 **Objetivo:**
        Analisar o perfil de suscetibilidade dos isolados mais recentes (código MA ≥ 181 por padrão).
        
        ### 📋 **Passo a Passo:**
        
        **Passo 1: Entenda os Dados**
        - Observe o **total de isolados** no grupo (mostrado no topo)
        - Se for 0, ajuste o "Limite numérico p/ 2025" na barra lateral
        - Se for muito baixo, verifique se os dados estão corretos
        
        **Passo 2: Analise a Tabela de Suscetibilidade**
        
        **📊 Colunas Principais:**
        - **Antibiótico**: Nome do antimicrobiano
        - **N**: Número de isolados testados (quanto maior, mais confiável)
        - **%R (fundo vermelho)**: Percentual de resistência - **MAIOR = PIOR**
        - **%S (fundo verde)**: Percentual de sensibilidade - **MAIOR = MELHOR**
        - **%I+SSD**: Percentual intermediário + dose dependente
        - **%Cobertura**: Percentual de isolados testados (N/Total × 100)
        
        **⚠️ Interpretação Crítica:**
        - Priorize antibióticos com **N alto** e **%Cobertura alta**
        - Desconfie de %R muito alto ou baixo com N pequeno
        - %R > 20% = considere alternativas; %R > 50% = evite uso empírico
        
        **Passo 3: Interprete os Gráficos**
        
        **📈 Gráfico "Top %R por antibiótico":**
        - Mostra ranking dos antibióticos com **maior resistência**
        - Antibióticos no topo = mais problemáticos para uso empírico
        - Use para identificar antimicrobianos a evitar
        
        **📊 Gráfico "Suscetibilidade dos isolados":**
        - Barras empilhadas: Verde (S) + Azul (I/SSD) + Vermelho (R) = 100%
        - Barras mais verdes = antibióticos melhores
        - Barras mais vermelhas = antibióticos problemáticos
        - Ordenação configurável na barra lateral
        
        **Passo 4: Use os Filtros**
        
        **🔧 Na Barra Lateral:**
        - **N mínimo testado**: Remove antibióticos com poucos testes
        - **Cobertura mínima %**: Remove antibióticos pouco testados
        - **Comece com 0** e aumente gradualmente se necessário
        
        **Passo 5: Exporte os Dados**
        - Use **"Baixar CSV"** para análises offline
        - Arquivo contém todos os dados da tabela formatados
        - Útil para relatórios e apresentações
        """)
    
    if should_show_section("section_5"):
        st.markdown("## 5. Análise Anos Anteriores")
    
    with st.expander("📆 Análise do Período Histórico - Passo a Passo", expanded=True):
        st.markdown("""
        ### 🎯 **Objetivo:**
        Analisar o perfil de suscetibilidade dos isolados históricos (código MA < 181 por padrão).
        
        ### 📋 **Funcionamento:**
        - **Mesma metodologia** da análise 2025/atual
        - **Interpretação idêntica** das tabelas e gráficos
        - **Controles iguais** de filtros e exportação
        
        ### 💡 **Dicas Específicas:**
        
        **Para Análise Isolada:**
        - Use para entender o perfil histórico da instituição
        - Identifique antibióticos que historicamente funcionavam bem
        - Observe padrões de resistência do passado
        
        **Para Preparar Comparação:**
        - Configure os **mesmos filtros** usados na análise 2025
        - Selecione os **mesmos antibióticos** para consistência
        - Anote antibióticos com boa/má performance histórica
        
        **Interpretação Contextual:**
        - Resistência histórica pode diferir muito da atual
        - Use como **baseline** para avaliar mudanças
        - Considere mudanças nas práticas clínicas entre períodos
        
        ### ⚙️ **Configurações Recomendadas:**
        - **N mínimo**: Mesmo valor usado na análise 2025
        - **Antibióticos**: Mesma seleção para comparabilidade
        - **Gráficos**: Mesmas configurações visuais
        """)
    
    if should_show_section("section_6"):
        st.markdown("## 6. Distribuição de Espécies")
    
    with st.expander("🧬 Análise da Distribuição de Espécies MALDI-TOF", expanded=True):
        st.markdown("""
        ### 🎯 **Objetivo:**
        Visualizar quais espécies são mais/menos frequentes no dataset total.
        
        ### 📋 **Passo a Passo:**
        
        **Passo 1: Entenda os Dados**
        - Tabela mostra **todas as espécies** identificadas por MALDI-TOF
        - Coluna **"Frequência"**: número absoluto de isolados
        - Coluna **"%"**: proporção relativa de cada espécie
        - Ordenação: mais frequentes primeiro
        
        **Passo 2: Configure a Visualização**
        
        **🎨 Na Barra Lateral - "Espécies (MALDI-TOF)":**
        
        **Tipo de Gráfico:**
        - **"Barra"**: Melhor para muitas espécies (> 5-8)
        - **"Pizza"**: Melhor para poucas espécies principais (≤ 5)
        
        **Top N espécies:**
        - Limita quantas espécies mostrar no gráfico
        - **3-5**: Visão executiva, só principais
        - **8-12**: Visão completa das mais relevantes
        - **15-20**: Visão detalhada, pode ficar poluído
        
        **Agrupar demais em "Outros":**
        - ✅ **Marcado**: Espécies não-top viram categoria "Outros"
        - ❌ **Desmarcado**: Mostra todas as Top N individualmente
        
        **Passo 3: Interprete os Resultados**
        
        **📊 Para Vigilância Epidemiológica:**
        - Identifique **espécies dominantes** na instituição
        - Observe **diversidade** vs **concentração** de espécies
        - Compare com literatura epidemiológica
        
        **🔬 Para Análises Posteriores:**
        - Espécies **mais frequentes** = análises mais robustas
        - Espécies **raras** = cuidado com interpretações
        - Use para priorizar quais espécies analisar em detalhes
        
        **💡 Para Relatórios:**
        - Gráfico de pizza = impacto visual para executivos
        - Gráfico de barras = precisão para análises técnicas
        - Tabela = dados exatos para auditoria
        
        **Passo 4: Identificar Padrões**
        
        **🔍 Questões a Considerar:**
        - Há **1-2 espécies dominantes** ou distribuição uniforme?
        - Espécies dominantes são **patógenos conhecidos** da instituição?
        - Diversidade sugere **amplo espectro** ou **surtos específicos**?
        
        ### ⚠️ **Limitações:**
        - Apenas isolados com MALDI-TOF válido (remove vazios e "*")
        - Não distingue períodos temporais (análise geral)
        - Frequência ≠ importância clínica (considere contexto)
        """)
    
    # Nota sobre boas práticas para análises básicas
    if should_show_section("section_4") or should_show_section("section_5") or should_show_section("section_6"):
        st.markdown("### 💡 **Dicas Gerais para Análises Básicas:**")
    
    st.info("""
    **🔧 Antes de Começar:**
    - Configure adequadamente a barra lateral
    - Selecione os antibióticos de interesse  
    - Ajuste filtros de qualidade conforme necessário
    
    **📊 Durante a Análise:**
    - Sempre observe N e %Cobertura antes de interpretar %R e %S
    - Compare múltiplos antibióticos, não confie em apenas um
    - Use os gráficos para identificação rápida de padrões
    
    **📁 Após a Análise:**
    - Exporte dados importantes para documentação
    - Anote configurações usadas para reprodutibilidade
    - Considere análises complementares (por espécie, temporal)
    """)
    
    # Nota sobre diferenças entre as abas básicas
    st.markdown("### � **Diferenças entre as Abas Básicas:**")
    
    col_diff1, col_diff2, col_diff3 = st.columns(3)
    
    with col_diff1:
        st.info("""
        **📅 2025/Atual**
        - Dados mais recentes
        - Reflete práticas atuais
        - Menor volume histórico
        - Use para decisões correntes
        """)
    
    with col_diff2:
        st.info("""
        **📆 Anos Anteriores**
        - Dados históricos acumulados
        - Maior volume de dados
        - Baseline institucional
        - Use para comparações
        """)
    
    with col_diff3:
        st.info("""
        **🧬 Distribuição de Espécies**
        - Todos os períodos juntos
        - Visão epidemiológica geral
        - Não temporal
        - Use para panorama geral
        """)
    
    # Seção 7: Análise por Espécie
    if should_show_section("section_7"):
        st.markdown("## 7. Análise por Espécie")
    
    with st.expander("🔬 Análise Detalhada por Espécie", expanded=True):
        st.markdown("""
        ### 🎯 **Quando Usar:**
        - Investigar resistência de espécies específicas
        - Preparar relatórios focados em patógenos importantes
        - Comparar perfis entre espécies do mesmo período
        
        ### 📋 **Passo a Passo:**
        
        **Passo 1: Escolha o Período**
        - Use as abas "Análise por Espécie - 2025/atual" ou "Anos anteriores"
        
        **Passo 2: Configure o N Mínimo**
        - Slider controla quantos isolados mínimos por espécie
        - **Recomendação**: 5 isolados (compromisso entre estabilidade e inclusão)
        - **Exploração**: 3 isolados (inclui mais espécies raras)
        - **Robustez**: 10+ isolados (só espécies muito frequentes)
        
        **Passo 3: Destaque Espécies (Opcional)**
        - Marque "Mostrar seleção de espécie" para análise rápida
        - Útil para verificar uma espécie específica antes da análise principal
        
        **Passo 4: Seleção Principal**
        - **Uma espécie**: Análise detalhada com todos os gráficos
        - **Todas as espécies**: Visão geral de todas válidas (pode ser longo)
        
        **Passo 5: Analise os Resultados**
        - **Tabela**: Perfil de resistência por antibiótico
        - **Top %R**: Antibióticos mais problemáticos para a espécie
        - **Gráfico Empilhado**: Perfil visual completo
        - **Tabela Detalhada**: Dados brutos por isolado (expandir)
        - **Resumo Comparativo**: Ranking entre espécies
        """)
    
    # Seção 8: Comparação Temporal
    if should_show_section("section_8"):
        st.markdown("## 8. Comparação Temporal")
    
    with st.expander("⏰ Análise de Evolução Temporal", expanded=True):
        st.markdown("""
        ### 🎯 **Objetivo:**
        Comparar a mesma espécie entre 2025/atual e anos anteriores para identificar tendências de resistência.
        
        ### 📋 **Passo a Passo Detalhado:**
        
        **Passo 1: Verifique a Disponibilidade**
        - A aplicação mostra automaticamente quantos isolados existem em cada período
        - "Espécies em Comum" indica quantas podem ser comparadas
        - Se não há espécies em comum, a análise não é possível
        
        **Passo 2: Configure os Filtros**
        - **N mínimo por período**: Espécie deve ter pelo menos este número em CADA período
        - **Padrão 3**: Compromisso entre inclusão e confiabilidade
        - **Aumente para 5-10**: Análises mais robustas, menos espécies
        
        **Passo 3: Selecione Antibióticos**
        - **Padrão**: Todos pré-selecionados
        - **Personalize**: Remova os que não interessam
        - **Dica**: Mantenha pelo menos 5-8 para análise significativa
        
        **Passo 4: Escolha a Espécie**
        - Lista ordenada por frequência total
        - Labels mostram N de isolados em cada período
        - **Dica**: Comece por espécies com N alto para resultados mais confiáveis
        
        **Passo 5: Analise os Resultados**
        
        **📊 Tabelas Comparativas:**
        - **Coluna 1**: Dados de 2025/atual
        - **Coluna 2**: Dados de anos anteriores  
        - **Coluna 3**: Variação (Δ) e tendências
        
        **🎨 Interpretação das Cores:**
        - **Vermelho**: Aumento da resistência (piora)
        - **Verde**: Diminuição da resistência (melhora)
        - **Cinza**: Mudança não significativa (estável)
        
        **📈 Gráficos Disponíveis:**
        - **Resistência: 2025 vs Anterior**: Barras lado a lado
        - **Sensibilidade: 2025 vs Anterior**: Barras lado a lado
        - **Variação da Resistência**: Mostra Δ%R com cores
        - **Perfil Completo**: Gráficos empilhados comparativos
        
        **📊 Resumo da Evolução:**
        - **Tendências de Resistência**: Quantos antibióticos aumentaram/diminuíram/estáveis
        - **Variações Médias**: Δ%R e Δ%S médios
        - **Maior Variação**: Antibiótico com maior mudança
        
        **Passo 6: Resumo Geral (Opcional)**
        - Marque "Mostrar resumo de todas as espécies válidas"
        - Tabela com todas as espécies e suas tendências
        - Útil para visão panorâmica ou relatórios executivos
        """)
    
    # Seção 9: Exportação de Dados
    if should_show_section("section_9"):
        st.markdown("## 9. Exportação de Dados")
    
    with st.expander("📥 Como Baixar e Usar os Dados", expanded=True):
        st.markdown("""
        ### 📊 **Tipos de Download Disponíveis:**
        
        **🔸 Tabelas de Suscetibilidade:**
        - Formato: CSV com codificação UTF-8
        - Conteúdo: Antibiótico, N, %R, %S, %I+SSD, %Cobertura, etc.
        - Onde: Todas as abas de análise têm botão "Baixar CSV"
        
        **🔸 Dados Detalhados por Isolado:**
        - Formato: CSV com resultados brutos
        - Conteúdo: Código UFPB, Espécie, resultados por antibiótico
        - Onde: Seção "Tabela detalhada por isolado" (expandir)
        
        **🔸 Resumos Comparativos:**
        - Formato: CSV com estatísticas por espécie
        - Conteúdo: Espécie, isolados, testes, %R médio, %S médio
        - Onde: Seções de resumo nas análises por espécie
        
        **🔸 Comparações Temporais:**
        - Formato: CSV com dados de ambos os períodos
        - Conteúdo: Dados 2025, dados anteriores, variações (Δ)
        - Onde: Aba de comparação temporal
        
        ### 💡 **Dicas de Uso dos Downloads:**
        
        **Para Relatórios:**
        - Use os CSVs de suscetibilidade para tabelas limpas
        - Importe no Excel/Word para formatação final
        
        **Para Análises Estatísticas:**
        - Use dados detalhados por isolado
        - Permitem análises customizadas em R/Python/SPSS
        
        **Para Auditoria:**
        - Compare dados brutos com resultados calculados
        - Verifique casos específicos de interesse
        
        **Para Apresentações:**
        - Use resumos comparativos para slides executivos
        - Dados já agregados e formatados
        """)
    
    # Seção 10: Interpretação de Resultados
    if should_show_section("section_10"):
        st.markdown("## 10. Interpretação de Resultados")
    
    with st.expander("🧠 Como Interpretar Corretamente os Dados", expanded=True):
        st.markdown("""
        ### 📊 **Métricas Principais:**
        
        **%R (Percentual de Resistência):**
        - **0-20%**: Resistência baixa (antibiótico ainda eficaz)
        - **20-50%**: Resistência moderada (usar com cautela)
        - **>50%**: Resistência alta (considerar alternativas)
        - **>80%**: Resistência muito alta (evitar uso empírico)
        
        **%S (Percentual de Sensibilidade):**
        - **>80%**: Excelente opção terapêutica
        - **60-80%**: Boa opção, monitorar tendências
        - **40-60%**: Opção limitada, avaliar contexto
        - **<40%**: Opção questionável para uso empírico
        
        **%Cobertura:**
        - **>90%**: Dados muito representativos
        - **70-90%**: Dados representativos
        - **50-70%**: Dados moderadamente representativos
        - **<50%**: Dados limitados, interpretar com cautela
        
        **N (Número de Isolados Testados):**
        - **>30**: Estatisticamente robusto
        - **10-30**: Moderadamente confiável
        - **5-10**: Limitado, usar com cautela
        - **<5**: Muito limitado, evitar conclusões definitivas
        
        ### ⚠️ **Armadilhas Comuns:**
        
        **🔸 N Pequeno vs Percentual Alto:**
        - Ex: 100% resistência com N=2 (só 2 isolados testados)
        - **Problema**: Conclusão baseada em amostra muito pequena
        - **Solução**: Sempre verificar N junto com %
        
        **🔸 Cobertura Baixa:**
        - Ex: Antibiótico testado em apenas 30% dos isolados
        - **Problema**: Viés de seleção (podem testar só casos graves)
        - **Solução**: Considerar % cobertura na interpretação
        
        **🔸 Misturar Espécies:**
        - Ex: Interpretar resistência geral sem considerar espécie
        - **Problema**: Espécies têm perfis muito diferentes
        - **Solução**: Sempre analisar por espécie quando possível
        
        **🔸 Confundir I/SSD com S:**
        - Ex: Considerar Intermediário como Sensível
        - **Problema**: Intermediário ≠ Sensível na prática clínica
        - **Solução**: Na dúvida, considere I/SSD como não-sensível
        
        ### 🎯 **Aplicações Práticas:**
        
        **Para Uso Clínico:**
        - Foque em %S para terapia empírica
        - Considere %R para evitar antibióticos problemáticos
        - Use dados por espécie, não gerais
        
        **Para Vigilância Epidemiológica:**
        - Compare tendências temporais
        - Monitore emergência de resistência
        - Identifique surtos ou mudanças de padrão
        
        **Para Política Institucional:**
        - Use dados robustos (N alto, cobertura boa)
        - Considere múltiplos períodos
        - Envolva equipe clínica na interpretação
        """)
    
    # Seção 11: Solução de Problemas
    if should_show_section("section_11"):
        st.markdown("## 11. Solução de Problemas")
    
    with st.expander("🔧 Problemas Comuns e Soluções", expanded=True):
        st.markdown("""
        ### ❌ **"Nenhum antibiótico atende aos filtros"**
        
        **Causa:** Filtros muito restritivos (N mínimo ou cobertura muito altos)
        **Solução:**
        - Reduza "N mínimo testado" para 0-5
        - Reduza "Cobertura mínima %" para 0-20%
        - Verifique se selecionou antibióticos existentes no arquivo
        
        ### ❌ **"Coluna MALDI-TOF não encontrada"**
        
        **Causa:** Nome da coluna diferente no arquivo
        **Solução:**
        - Verifique se a coluna existe e contém "MALDI" no nome
        - Renomeie a coluna no Excel para "MALDI-TOF" ou similar
        - Verifique se está na linha correta do cabeçalho
        
        ### ❌ **"Sem espécies válidas para comparação temporal"**
        
        **Causa:** Nenhuma espécie tem isolados suficientes em ambos os períodos
        **Solução:**
        - Reduza "N mínimo por período" para 1-2
        - Verifique se o "Limite para 2025" está correto
        - Confirme se há dados em ambos os períodos
        
        ### ❌ **"Gráficos não aparecem ou ficam estranhos"**
        
        **Causa:** Problemas com Plotly ou dados inconsistentes
        **Solução:**
        - Desmarque "Forçar gráficos interativos" na barra lateral
        - Verifique se há dados suficientes para o gráfico
        - Reduza o número de antibióticos/espécies selecionados
        
        ### ❌ **"Números não fazem sentido"**
        
        **Causa:** Interpretação incorreta ou dados problemáticos
        **Solução:**
        - Baixe o CSV e verifique os dados brutos
        - Compare com dados originais no Excel
        - Verifique a documentação metodológica
        - Considere filtros de qualidade
        
        ### ❌ **"App está lento ou travando"**
        
        **Causa:** Arquivo muito grande ou muitas seleções
        **Solução:**
        - Feche outras abas do navegador
        - Reduza número de antibióticos selecionados
        - Desmarque gráficos interativos
        - Reinicie o aplicativo
        
        ### ❌ **"Download não funciona"**
        
        **Causa:** Bloqueio do navegador ou dados vazios
        **Solução:**
        - Verifique se há dados na tabela antes de baixar
        - Tente outro navegador (Chrome recomendado)
        - Verifique configurações de download do navegador
        
        ### 📞 **Ainda com problemas?**
        
        **Verifique:**
        1. Estrutura do arquivo Excel (colunas necessárias)
        2. Configurações da barra lateral
        3. Seleções feitas em cada aba
        4. Mensagens de erro ou avisos na tela
        
        **Dicas gerais:**
        - Comece sempre com configurações padrão
        - Ajuste filtros gradualmente
        - Use dados de exemplo para testar
        - Documente configurações que funcionam
        """)
    
    # Footer da aba
    if not ('scroll_to' in st.session_state):
        st.divider()
        st.markdown("### 🎉 **Parabéns!**")
        st.success("""
        Agora você domina todas as funcionalidades da aplicação! 
        
        📊 **Lembre-se das boas práticas:**
        - Sempre verifique N e cobertura antes de interpretar
        - Use análises por espécie para decisões clínicas
        - Compare períodos para identificar tendências
        - Exporte dados para relatórios e auditorias
        
        🔬 **Para pesquisa e vigilância:**
        - Documente métodos e filtros utilizados
        - Mantenha consistência entre análises
        - Considere limitações dos dados
        - Envolva especialistas na interpretação
        """)
        
        st.info("💡 **Dica:** Salve este guia nos favoritos para consulta rápida!")

# ---------------------------
# Rodapé (guia único)
# ---------------------------
st.divider()
with st.expander("Guia rápido: uso, regras e cálculos", expanded=False):
        st.markdown(
                """
                • Como usar (barra lateral)
                - Passo 1 — Fonte dos dados
                    - Arquivo padrão (Cepas.xlsx): usa automaticamente o arquivo local se existir na pasta do app.
                    - Upload de .xlsx: envie outro arquivo para análise.
                - Passo 2 — Cabeçalho e leitura
                    - Linha do cabeçalho: 2 por padrão (porque a 1ª linha costuma ser título/legenda). Ajuste se a sua planilha tiver o cabeçalho em outra linha.
                - Passo 3 — Particionamento temporal
                    - Limite p/ 2025: separa os grupos a partir do número extraído de `CÓDIGO UFPB` após “MA”. Ex.: `MA180` ⇒ 180. Valores `>= limite` entram em “2025 / atual”; os demais, em “Anos anteriores”.
                - Passo 4 — Antibióticos e cobertura
                    - Antibióticos: selecione as colunas a considerar (somente as existentes na planilha aparecem).
                    - N mínimo testado: oculta antibióticos com menos de N resultados válidos.
                    - Cobertura mínima %: oculta antibióticos pouco testados no grupo (`N/Total`).
                - Passo 5 — Gráficos por antibiótico
                    - Top N por %R: define quantos antibióticos aparecem no gráfico horizontal de Top Resistência (%R).
                    - Ordenar gráfico empilhado por: ordena as barras por `%R`, `%S`, `%Cobertura` ou por nome.
                    - Rótulos das barras (empilhado): ative se quiser mostrar valores dentro das barras; ajuste limiar de exibição, casas decimais, tamanho e direção do texto.
                    - Forçar gráficos interativos (Plotly): liga/desliga versão interativa (útil para zoom e hover; desative se estiver pesado no navegador).
                - Passo 6 — Espécies (MALDI-TOF)
                    - Tipo de gráfico: “Barra” (horizontal) ou “Pizza”.
                    - Top N espécies: limita às mais frequentes.
                    - Agrupar demais em “Outros”: soma as demais espécies em um único item.
                - Dicas rápidas
                    - Se algum antibiótico “sumir”, verifique se os filtros de N mínimo/Cobertura mínima não estão altos demais.
                    - Se o Arquivo padrão não existir, use Upload.
                    - Use “Baixar CSV” para auditar e conferir casos específicos.

                • Como os dados são lidos
                - Normalização dos nomes de colunas (trim e troca de `–` por `-`).
                - Colunas‑chave: `CÓDIGO UFPB` (exato após normalização) e `MALDI-TOF` (aceita variações contendo “MALDI”).
                - Antibióticos considerados (usa apenas os que existem na planilha):
                    GEN, TOB, AMI, ATM, CRO, CAZ, CTX, CFO, CPM, AMC, AMP, PPT, CZA, MER, IMP, CIP, LEV, SUT, POLI B.

                • Regras e cálculos
                - `_NUM`: extraído de `CÓDIGO UFPB` com `MA(\\d+)` (ex.: `MA24B`, `MA180`).
                - Particionamento: `_NUM >= 180` ⇒ “2025 / atual”; `_NUM < 180` ⇒ “Anos anteriores”.
                - Resultado por célula: pega o primeiro rótulo válido em `{SSD, R, S, I}`; aceita variações (`SSD/I`, `SSD–`, `R *`, `S (ok)`), e `INTERMEDIARIO/INTERMEDIÁRIO` vira `I`. Ignora vazio, `*`, `-`, `NaN`.
                - Para cada antibiótico: `N`, contagens `R/S/I/SSD`, `%R`, `%S`, `%I+SSD` (base `N`), `Total`, `%Cobertura = 100*N/Total`, `Sem Resultado = Total - N`.
                - Ordenação da tabela: `%R` (desc), depois `%Cobertura` (desc) e nome.

                • Saídas
                - Tabela: `Antibiótico | N | Total | %Cobertura | R | S | I | SSD | %R | %S | %I+SSD | Sem Resultado` (percentuais com 1 casa decimal) + botão “Baixar CSV”.
                - Gráficos: Top %R (horizontal) e gráfico empilhado de `%S`, `%I+SSD`, `%R` com rótulos opcionais.
                - Espécies (MALDI-TOF): remove `""` e `*`; mostra barras horizontais (ou pizza) com Top N e “Outros”.

                • Salvaguardas
                - Não eliminamos testes válidos: qualquer célula que contenha `SSD|R|S|I` é contada uma vez.
                - Diferenças de `N` refletem a cobertura real (células vazias/asteriscos não contam).
                - Conferência: exporte o CSV e verifique casos específicos (ex.: `GEN` em `>=180`).

                • Fórmulas
                - `%R = 100 * R / N`
                - `%S = 100 * S / N`
                - `%I+SSD = 100 * (I + SSD) / N`
                - `%Cobertura = 100 * N / Total`
                """
        )
