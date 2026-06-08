import streamlit as st
import plotly.graph_objects as go
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def render_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Dark sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d1a 0%, #111128 60%, #0a1628 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * { color: #c8d8e8 !important; }
        [data-testid="stSidebar"] .stSlider > label { color: #a0b8cc !important; }
        [data-testid="stSidebar"] h2 { color: #ffffff !important; }
        [data-testid="stSidebar"] .stMarkdown p { color: #90a8bc !important; }

        /* Header kartı */
        .main-header {
            background: linear-gradient(135deg, #0f0f23 0%, #14213d 50%, #0e2d5c 100%);
            padding: 2rem 2.5rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(100,160,255,0.15);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: #7a9cbf;
            margin: 0;
            font-size: 0.9rem;
        }
        .main-header .badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-left: 0.5rem;
            vertical-align: middle;
        }
        .badge-blue  { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
        .badge-green { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
        .badge-red   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }

        /* Sonuç banner'ları */
        .alert-fake {
            background: linear-gradient(135deg, rgba(220,38,38,0.12), rgba(239,68,68,0.06));
            border: 1px solid rgba(220,38,38,0.45);
            border-left: 4px solid #dc2626;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            color: #fca5a5;
            font-weight: 600;
            font-size: 1.05rem;
            margin: 0.5rem 0;
        }
        .alert-real {
            background: linear-gradient(135deg, rgba(5,150,105,0.12), rgba(16,185,129,0.06));
            border: 1px solid rgba(5,150,105,0.45);
            border-left: 4px solid #059669;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            color: #6ee7b7;
            font-weight: 600;
            font-size: 1.05rem;
            margin: 0.5rem 0;
        }
        .alert-info {
            background: rgba(59,130,246,0.08);
            border: 1px solid rgba(59,130,246,0.25);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #93c5fd;
            font-size: 0.88rem;
            margin: 0.5rem 0;
        }

        /* İndir butonu */
        [data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #1e3a5f, #0e2d5c) !important;
            color: #93c5fd !important;
            border: 1px solid rgba(59,130,246,0.35) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background: linear-gradient(135deg, #254870, #1a3d70) !important;
            border-color: rgba(59,130,246,0.6) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Yapay Zeka Tabanlı Deepfake Tespit Sistemi
            <span class="badge badge-blue">Xception41</span>
            <span class="badge badge-green">MTCNN</span>
            <span class="badge badge-red">Hibrit P85</span>
        </h1>
        <p>İki aşamalı derin öğrenme pipeline'ı &nbsp;·&nbsp; AMP (FP16) hızlandırma
           &nbsp;·&nbsp; Savitzky-Golay sinyal filtresi &nbsp;·&nbsp; Hibrit P85 karar mekanizması</p>
    </div>
    """, unsafe_allow_html=True)

def plot_timeline(tum_kareler, tum_skorlar, esik_degeri, percentile_85_skor):
    fig = go.Figure()

    # Ham skor (şeffaf arka plan çizgisi)
    fig.add_trace(go.Scatter(
        x=tum_kareler, y=tum_skorlar,
        mode="lines", name="Ham Skor",
        line=dict(color="rgba(220,20,60,0.22)", width=1)
    ))

    # Savitzky-Golay yumuşatılmış eğri
    if SCIPY_AVAILABLE and len(tum_skorlar) >= 7:
        pencere = min(21, len(tum_skorlar))
        if pencere % 2 == 0:
            pencere -= 1
        if pencere >= 5:
            smooth = savgol_filter(tum_skorlar, window_length=pencere, polyorder=3)
            fig.add_trace(go.Scatter(
                x=tum_kareler, y=smooth.tolist(),
                mode="lines", name="Savitzky-Golay Egrisi",
                line=dict(color="crimson", width=2.5)
            ))

    # Eşik çizgisi
    fig.add_hline(
        y=esik_degeri * 100,
        line_dash="dash", line_color="orange",
        annotation_text=f"Karar Esigi: %{esik_degeri*100:.0f}",
        annotation_position="bottom right"
    )

    # P85 referans çizgisi
    fig.add_hline(
        y=percentile_85_skor,
        line_dash="dot", line_color="rgba(250,200,50,0.65)",
        annotation_text=f"P85: %{percentile_85_skor:.1f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Zaman Cizelgesi Deepfake Analizi",
        xaxis_title="Kare Numarasi",
        yaxis_title="Sahte Olasiligi (%)",
        yaxis=dict(range=[0, 100]),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,252,0.85)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
