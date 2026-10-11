from __future__ import annotations

import io
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pypdf import PdfReader

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# ASTER | ACADEMIC INTELLIGENCE
# A research-style student performance exploration workspace.
# ============================================================

st.set_page_config(
    page_title="Aster | Academic Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Design system ----------------------------
BG = "#F7F7F1"
PAPER = "#FFFFFF"
INK = "#183A2F"
MUTED = "#526A60"
GREEN = "#1D5B45"
GREEN_2 = "#2E795A"
MINT = "#B9D4C1"
PALE = "#E8F0E7"
GOLD = "#C89A45"
RED = "#B84A45"
GRID = "#D6E0D5"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    :root {{
        --aster-bg: {BG};
        --aster-paper: {PAPER};
        --aster-ink: {INK};
        --aster-muted: {MUTED};
        --aster-green: {GREEN};
        --aster-green2: {GREEN_2};
        --aster-mint: {MINT};
        --aster-pale: {PALE};
        --aster-gold: {GOLD};
    }}
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: var(--aster-ink);
    }}
    .stApp {{
        background: var(--aster-bg);
    }}
    [data-testid="stHeader"] {{
        background: rgba(247,247,241,0.94);
    }}
    [data-testid="stSidebar"] {{
        background: #183A2F;
        border-right: 1px solid #2A5846;
    }}
    [data-testid="stSidebar"] * {{
        color: #F7F7F1 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: 0.38rem 0.15rem;
        font-size: 0.98rem;
    }}
    h1, h2, h3 {{
        color: var(--aster-ink) !important;
        letter-spacing: -0.025em;
    }}
    h1, h2 {{
        font-family: 'Playfair Display', Georgia, serif !important;
    }}
    p, li, label, .stMarkdown {{
        color: var(--aster-ink);
    }}
    .eyebrow {{
        color: var(--aster-green2);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.19em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }}
    .hero {{
        background: linear-gradient(120deg, #E8F0E7 0%, #FDFCF7 70%, #F2E9D8 100%);
        border: 1px solid #D7E3D6;
        border-radius: 22px;
        padding: clamp(1.5rem, 4vw, 3.2rem);
        margin: 0.6rem 0 1.6rem 0;
        box-shadow: 0 12px 34px rgba(24,58,47,0.06);
    }}
    .hero h1 {{
        font-size: clamp(2.4rem, 5vw, 4.5rem);
        line-height: 1.08;
        margin: 0.25rem 0 1rem 0;
        max-width: 900px;
    }}
    .hero p {{
        color: #3D5D50;
        font-size: 1.1rem;
        line-height: 1.85;
        max-width: 920px;
        margin-bottom: 0;
    }}
    .section-kicker {{
        color: var(--aster-green2);
        font-weight: 700;
        font-size: 0.76rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 1.4rem 0 0.55rem 0;
    }}
    .body-copy {{
        color: #38574B;
        font-size: 1.02rem;
        line-height: 1.9;
    }}
    .quote-card {{
        background: #E7EFE5;
        border-left: 5px solid var(--aster-gold);
        border-radius: 0 16px 16px 0;
        padding: 1.25rem 1.5rem;
        margin: 1.2rem 0 1.6rem 0;
    }}
    .quote-card p {{
        font-family: 'Playfair Display', Georgia, serif;
        color: #234638;
        font-size: 1.3rem;
        line-height: 1.6;
        margin: 0;
    }}
    .quote-card span {{
        display: block;
        color: #60766A;
        margin-top: 0.6rem;
        font-size: 0.84rem;
        font-weight: 600;
    }}
    .metric-card {{
        background: var(--aster-paper);
        border: 1px solid #DDE5D9;
        border-radius: 16px;
        padding: 1.15rem 1.2rem;
        min-height: 126px;
        box-shadow: 0 7px 22px rgba(24,58,47,0.04);
    }}
    .metric-label {{
        color: #597267;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
    .metric-value {{
        color: #183A2F;
        font-size: clamp(1.6rem, 2.6vw, 2.35rem);
        font-weight: 700;
        line-height: 1.2;
    }}
    .metric-note {{
        color: #71857B;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }}
    .content-card {{
        background: #FFFFFF;
        border: 1px solid #DFE7DC;
        border-radius: 18px;
        padding: 1.35rem 1.45rem;
        margin: 0.7rem 0;
    }}
    .content-card h3 {{
        margin-top: 0;
    }}
    .callout {{
        background: #F1EBDD;
        border: 1px solid #E5D9C2;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #5B4C30;
        line-height: 1.75;
        margin: 0.8rem 0;
    }}
    .small-note {{
        color: #657B70;
        font-size: 0.86rem;
        line-height: 1.7;
    }}
    .footer {{
        border-top: 1px solid #D9E1D7;
        margin-top: 3rem;
        padding: 1.3rem 0 0.7rem 0;
        color: #71847A;
        font-size: 0.82rem;
        line-height: 1.7;
    }}
    /* Strong contrast for all buttons, including form submit buttons. */
    .stButton > button,
    .stFormSubmitButton > button,
    button[kind="primary"] {{
        background: #1D5B45 !important;
        color: #FFFFFF !important;
        border: 1px solid #1D5B45 !important;
        border-radius: 10px !important;
        min-height: 2.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(29,91,69,0.14);
    }}
    .stButton > button p,
    .stFormSubmitButton > button p,
    button[kind="primary"] p {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    button[kind="primary"]:hover {{
        background: #143F30 !important;
        color: #FFFFFF !important;
        border-color: #143F30 !important;
    }}
    [data-testid="stDownloadButton"] button {{
        background: #1D5B45 !important;
        color: #FFFFFF !important;
        border: 1px solid #1D5B45 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}
    [data-testid="stDownloadButton"] button * {{
        color: #FFFFFF !important;
    }}
    .stTextInput input, .stNumberInput input, .stSelectbox div,
    .stTextArea textarea {{
        border-radius: 9px !important;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid #DCE5DA;
        border-radius: 12px;
        overflow: hidden;
    }}
    div[data-baseweb="tab-list"] {{
        gap: 0.35rem;
    }}
    button[data-baseweb="tab"] {{
        color: #365849 !important;
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #1D5B45 !important;
        border-bottom-color: #1D5B45 !important;
    }}
    .stAlert {{
        border-radius: 12px;
    }}
    @media (max-width: 700px) {{
        .hero {{ padding: 1.35rem; }}
        .hero h1 {{ font-size: 2.35rem; }}
        .quote-card p {{ font-size: 1.08rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Data + model ----------------------------
FEATURES = [
    "Study Hours",
    "Attendance",
    "Previous Score",
    "Assignment Completion",
    "Sleep Hours",
]
TARGET = "Final Score"
REQUIRED_COLUMNS = FEATURES + [TARGET]


def generate_demo_data(n: int = 1200, seed: int = 27) -> pd.DataFrame:
    """Create repeatable synthetic data for demonstration and testing."""
    rng = np.random.default_rng(seed)
    study = rng.uniform(0.5, 10.0, n)
    attendance = rng.uniform(45, 100, n)
    previous = rng.uniform(35, 100, n)
    assignments = rng.uniform(30, 100, n)
    sleep = rng.uniform(4.0, 10.0, n)
    noise = rng.normal(0, 5.5, n)

    score = (
        20
        + study * 2.45
        + attendance * 0.17
        + previous * 0.34
        + assignments * 0.10
        + sleep * 0.65
        + noise
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame(
        {
            "Study Hours": np.round(study, 1),
            "Attendance": np.round(attendance, 1),
            "Previous Score": np.round(previous, 1),
            "Assignment Completion": np.round(assignments, 1),
            "Sleep Hours": np.round(sleep, 1),
            "Final Score": np.round(score, 1),
        }
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded columns and coerce the expected values to numbers."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing these required columns: "
            + ", ".join(missing)
            + ". Please check the column names and try again."
        )
    data = data[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET])
    if len(data) < 20:
        raise ValueError("At least 20 rows with a valid Final Score are required.")
    data = data.dropna(subset=FEATURES, how="all")
    data = data.dropna(subset=FEATURES)
    if data.empty:
        raise ValueError("No complete numeric records remain after cleaning.")
    for col in FEATURES:
        if col in ("Attendance", "Previous Score", "Assignment Completion"):
            data[col] = data[col].clip(0, 100)
        elif col in ("Study Hours", "Sleep Hours"):
            data[col] = data[col].clip(0, 24)
    data[TARGET] = data[TARGET].clip(0, 100)
    return data.reset_index(drop=True)


def fit_model(data: pd.DataFrame):
    """Fit and return a Random Forest model with held-out evaluation metrics."""
    x = data[FEATURES]
    y = data[TARGET]
    if len(data) >= 30:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.22, random_state=42
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan"),
        "y_test": np.asarray(y_test),
        "predictions": np.asarray(predictions),
        "test_count": int(len(y_test)),
    }
    return model, metrics


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF. Scanned/image PDFs need OCR first."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def find_subject_marks(text: str) -> pd.DataFrame:
    """Conservatively detect common subject/mark formats in text-based report cards."""
    known_subjects = [
        "Mathematics", "Maths", "Math", "Physics", "Chemistry", "Biology",
        "English", "Hindi", "Computer Science", "Computer Applications",
        "Social Science", "Social Studies", "Economics", "History", "Geography",
        "Science", "Programming", "Data Structures", "Artificial Intelligence",
    ]
    rows = []
    for subject in known_subjects:
        # Examples accepted: Mathematics 85/100, English: 78%, Physics - 91
        pattern = re.compile(
            rf"(?im)^\s*{re.escape(subject)}\s*[:\-–]?\s*(\d{{1,3}}(?:\.\d+)?)\s*(?:/\s*(\d{{1,3}}(?:\.\d+)?)|%)?",
        )
        for match in pattern.finditer(text):
            score = float(match.group(1))
            maximum = float(match.group(2)) if match.group(2) else (100.0 if score <= 100 else None)
            if maximum and maximum > 0 and score <= maximum:
                percent = score / maximum * 100
                rows.append({"Subject": subject, "Marks": score, "Out of": maximum, "Percentage": round(percent, 1)})
                break
    # Remove duplicate aliases for the same subject only when the exact subject is duplicated.
    if not rows:
        return pd.DataFrame(columns=["Subject", "Marks", "Out of", "Percentage"])
    return pd.DataFrame(rows).drop_duplicates(subset=["Subject"], keep="first").reset_index(drop=True)


def current_dataset() -> pd.DataFrame:
    if "aster_data" not in st.session_state:
        st.session_state.aster_data = generate_demo_data()
        st.session_state.aster_source = "Synthetic demonstration data"
    return st.session_state.aster_data


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quote_card(quote: str, attribution: str = "ASTER FIELD NOTE") -> None:
    st.markdown(
        f'<div class="quote-card"><p>“{quote}”</p><span>{attribution}</span></div>',
        unsafe_allow_html=True,
    )


def section_intro(kicker: str, title: str, paragraph: str) -> None:
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.header(title)
    st.markdown(f'<div class="body-copy">{paragraph}</div>', unsafe_allow_html=True)


def style_plot(fig, height: int = 420):
    """Apply high-contrast axis labels, ticks, grid lines, and backgrounds."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, Arial, sans-serif", size=14, color="#183A2F"),
        title_font=dict(size=19, color="#183A2F"),
        margin=dict(l=35, r=25, t=65, b=55),
        hoverlabel=dict(
            bgcolor="#183A2F",
            bordercolor="#183A2F",
            font=dict(color="#FFFFFF", size=13),
        ),
        legend=dict(
            font=dict(color="#183A2F", size=12),
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=14, color="#183A2F"),
        tickfont=dict(size=12, color="#274B3C"),
        showgrid=True,
        gridcolor="#D6E0D5",
        gridwidth=1.2,
        zeroline=True,
        zerolinecolor="#9BB2A1",
        linecolor="#8DA596",
        linewidth=1.2,
        ticks="outside",
        tickcolor="#8DA596",
    )
    fig.update_yaxes(
        title_font=dict(size=14, color="#183A2F"),
        tickfont=dict(size=12, color="#274B3C"),
        showgrid=True,
        gridcolor="#D6E0D5",
        gridwidth=1.2,
        zeroline=True,
        zerolinecolor="#9BB2A1",
        linecolor="#8DA596",
        linewidth=1.2,
        ticks="outside",
        tickcolor="#8DA596",
    )
    return fig


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.7rem 0 1.4rem 0;">
          <div style="font-family:'Playfair Display',Georgia,serif;font-size:2.7rem;font-weight:700;line-height:1.05;">aster.</div>
          <div style="font-size:0.76rem;letter-spacing:0.18em;margin-top:0.55rem;color:#C8DCCF;">ACADEMIC INTELLIGENCE</div>
        </div>
        <hr style="border-color:#416756;margin-bottom:1.2rem;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Workspace")
    page = st.radio(
        "Navigate Aster",
        ["Overview", "Performance Lab", "Student PDF Report", "Dataset Explorer", "Model Evaluation", "About Aster"],
        label_visibility="collapsed",
        key="aster_navigation",
    )
    st.markdown("---")
    st.markdown("### Data source")
    st.caption("Start with the included synthetic dataset, or upload a CSV of your own.")
    uploaded_file = st.file_uploader(
        "Upload student data (CSV)",
        type=["csv"],
        help="Required columns: Study Hours, Attendance, Previous Score, Assignment Completion, Sleep Hours, Final Score.",
    )
    pdf_file = st.file_uploader(
        "Upload an individual student's report card (PDF)",
        type=["pdf"],
        key="aster_student_pdf",
        help="Text-based PDF report cards are supported. Scanned PDFs may not contain extractable text.",
    )
    if pdf_file is not None and st.session_state.get("aster_pdf_name") != pdf_file.name:
        try:
            pdf_text = extract_pdf_text(pdf_file.getvalue())
            st.session_state.aster_pdf_text = pdf_text
            st.session_state.aster_pdf_name = pdf_file.name
            st.session_state.aster_pdf_marks = find_subject_marks(pdf_text)
            if pdf_text:
                st.success("PDF ready to review. Open Student PDF Report.")
            else:
                st.warning("No selectable text found. This may be a scanned PDF; OCR is needed.")
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            cleaned = clean_dataset(upload_df)
            st.session_state.aster_data = cleaned
            st.session_state.aster_source = f"Uploaded dataset: {uploaded_file.name}"
            st.success(f"Loaded {len(cleaned):,} valid records.")
        except Exception as exc:
            st.error(str(exc))
    if st.button("Restore demonstration data", use_container_width=True):
        st.session_state.aster_data = generate_demo_data()
        st.session_state.aster_source = "Synthetic demonstration data"
        st.rerun()
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.83rem;line-height:1.75;color:#D2E2D7;">'
        "<b>Research principle</b><br>Data can reveal patterns. It cannot tell the whole story of a student."
        "</div>",
        unsafe_allow_html=True,
    )

data = current_dataset()
model, evaluation = fit_model(data)
mean_score = float(data[TARGET].mean())
median_score = float(data[TARGET].median())
score_std = float(data[TARGET].std(ddof=0))
feature_importance = pd.DataFrame(
    {"Input": FEATURES, "Relative importance": model.feature_importances_}
).sort_values("Relative importance", ascending=True)
source_label = st.session_state.get("aster_source", "Synthetic demonstration data")

# ---------------------------- Header ----------------------------
st.markdown(
    """
    <div class="eyebrow">ACADEMIC RESEARCH WORKSPACE / 01</div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">ASTER / LEARNING ANALYTICS</div>
          <h1>Understanding the patterns behind student performance.</h1>
          <p>
            Aster is an interactive academic analytics workspace for exploring study habits,
            attendance, prior achievement, assignment completion and sleep alongside examination
            outcomes. It brings a small data-science workflow into one readable place: inspect
            the dataset, test a model, explore a hypothetical profile and reflect on what the
            results may—and may not—mean.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="callout">
          <b>ACTIVE DATASET</b> &nbsp; / &nbsp; {source_label}
          &nbsp; · &nbsp; {len(data):,} usable records
          &nbsp; · &nbsp; Random Forest regression
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">01 — DATASET SNAPSHOT</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Student records", f"{len(data):,}", "Rows currently included")
    with m2:
        metric_card("Mean final score", f"{mean_score:.1f}/100", "Average across records")
    with m3:
        metric_card("Model MAE", f"{evaluation['mae']:.2f} points", "Lower is better")
    with m4:
        r2_text = f"{evaluation['r2']:.3f}" if np.isfinite(evaluation["r2"]) else "N/A"
        metric_card("Model R²", r2_text, "Held-out evaluation score")

    st.markdown(
        '<p class="small-note">These metrics describe this dataset and this model run. They are not guarantees of how well the model will perform for a different school, student population or assessment.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">02 — WHAT IS THE DATASET TELLING US?</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.histogram(
            data,
            x=TARGET,
            nbins=16,
            title="Distribution of final scores",
            labels={TARGET: "Final score (out of 100)", "count": "Number of records"},
            color_discrete_sequence=[GREEN_2],
        )
        fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=0.8, opacity=0.95)
        fig.update_xaxes(range=[0, 100], dtick=10, title_text="Final score (out of 100)")
        fig.update_yaxes(title_text="Number of records", rangemode="tozero", dtick=50)
        st.plotly_chart(style_plot(fig, 440), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
        st.markdown(
            '<div class="small-note">Each bar groups students whose scores fall within a score interval. Taller bars indicate that more records fall into that interval. The distribution is a useful starting point for spotting clusters and unusually low or high values.</div>',
            unsafe_allow_html=True,
        )
    with right:
        imp = feature_importance.sort_values("Relative importance", ascending=True)
        fig = px.bar(
            imp,
            x="Relative importance",
            y="Input",
            orientation="h",
            title="Inputs used by the model",
            text=imp["Relative importance"].map(lambda x: f"{x:.2f}"),
            color="Relative importance",
            color_continuous_scale=["#D4E5D5", "#6A9C7F", "#1D5B45"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(
            title_text="Relative importance",
            tickformat=".2f",
            dtick=0.05,
            range=[0, max(0.4, float(imp["Relative importance"].max()) * 1.22)],
        )
        fig.update_yaxes(title_text="Input variable", tickfont=dict(size=12, color="#274B3C"))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_plot(fig, 440), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
        st.markdown(
            '<div class="small-note">Relative importance describes how much the fitted Random Forest relies on each input when making predictions. It does not prove that an input causes a higher or lower score.</div>',
            unsafe_allow_html=True,
        )

    quote_card(
        "A number can start a conversation, but it should never be the end of one.",
        "ASTER / A NOTE ON RESPONSIBLE INTERPRETATION",
    )

    st.markdown('<div class="section-kicker">03 — LOOKING FOR RELATIONSHIPS</div>', unsafe_allow_html=True)
    fig = px.scatter(
        data,
        x="Study Hours",
        y=TARGET,
        color="Attendance",
        color_continuous_scale=["#D2E4D5", "#5E9879", "#1D5B45"],
        opacity=0.72,
        hover_data=["Previous Score", "Assignment Completion", "Sleep Hours"],
        title="Study hours compared with final score",
        labels={
            "Study Hours": "Study hours per day",
            TARGET: "Final score (out of 100)",
            "Attendance": "Attendance (%)",
        },
    )
    fig.update_xaxes(range=[0, max(10, float(data["Study Hours"].max()) + 0.5)], dtick=1)
    fig.update_yaxes(range=[0, 105], dtick=10)
    st.plotly_chart(style_plot(fig, 520), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
    st.markdown(
        '<div class="body-copy">A scatter plot shows one record per point. Look for broad patterns rather than expecting every point to follow a straight line. Students with similar study hours may still receive different scores because achievement is influenced by many factors, some of which are not represented here.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">04 — READ THE RESULTS WITH CARE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="content-card">
              <h3>What the charts can help us explore</h3>
              <p class="body-copy">The visualisations can help us inspect the score distribution, compare observed inputs, identify broad associations and ask better questions about learning conditions. They are useful for exploration because they make patterns easier to see than a table of numbers alone.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="content-card">
              <h3>What the charts cannot establish</h3>
              <p class="body-copy">These charts do not explain every individual outcome, establish cause and effect, or measure motivation, wellbeing, teaching quality or personal circumstances. A model can be wrong, and its estimates should never be used as a final judgement of a student's ability or future.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 2: PERFORMANCE LAB
# ============================================================
elif page == "Performance Lab":
    section_intro(
        "02 — INTERACTIVE PREDICTION",
        "Performance Lab",
        "Enter a hypothetical student profile to obtain an estimate from the currently fitted model. Adjust the inputs to see how the estimate responds. The controls do not retrain the model, and the result should be treated as an educational demonstration rather than a promise about a real examination.",
    )
    quote_card(
        "Good analysis does not replace human judgement; it gives judgement better questions to ask.",
        "ASTER / FIELD NOTE",
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### Student profile")
        st.markdown(
            '<p class="small-note">Use realistic values. The ranges are input boundaries, not targets students are expected to reach.</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="large")
        with col1:
            study = st.slider(
                "Daily study hours",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
                help="Approximate focused study time in a typical day.",
            )
            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=85,
                help="Percentage of classes attended.",
            )
            previous = st.slider(
                "Previous examination score",
                min_value=0,
                max_value=100,
                value=70,
                help="Score on a previous assessment, out of 100.",
            )
        with col2:
            assignments = st.slider(
                "Assignment completion (%)",
                min_value=0,
                max_value=100,
                value=80,
                help="Approximate proportion of assignments completed.",
            )
            sleep = st.slider(
                "Average sleep hours",
                min_value=0.0,
                max_value=12.0,
                value=7.0,
                step=0.5,
                help="Average sleep per night.",
            )
            st.markdown(
                '<div class="callout"><b>Remember</b><br>Inputs are simplified indicators. They do not capture the full circumstances behind a student’s performance.</div>',
                unsafe_allow_html=True,
            )
        submitted = st.form_submit_button("Estimate final score", type="primary", use_container_width=True)

    if submitted:
        profile = pd.DataFrame(
            [{
                "Study Hours": study,
                "Attendance": attendance,
                "Previous Score": previous,
                "Assignment Completion": assignments,
                "Sleep Hours": sleep,
            }]
        )[FEATURES]
        score = float(np.clip(model.predict(profile)[0], 0, 100))
        st.markdown('<div class="section-kicker">YOUR MODEL ESTIMATE</div>', unsafe_allow_html=True)
        a, b = st.columns([1, 1.4], gap="large")
        with a:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "/100", "font": {"size": 42, "color": INK}},
                    title={"text": "Estimated final score", "font": {"size": 18, "color": INK}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickmode": "linear",
                            "dtick": 10,
                            "tickfont": {"size": 12, "color": INK},
                        },
                        "bar": {"color": GREEN_2, "thickness": 0.28},
                        "bgcolor": "#FFFFFF",
                        "bordercolor": "#B7C9B9",
                        "borderwidth": 2,
                        "steps": [
                            {"range": [0, 40], "color": "#F4DEDA"},
                            {"range": [40, 70], "color": "#F2E9D5"},
                            {"range": [70, 100], "color": "#DDECE0"},
                        ],
                        "threshold": {
                            "line": {"color": GOLD, "width": 4},
                            "thickness": 0.8,
                            "value": score,
                        },
                    },
                )
            )
            fig.update_layout(
                height=350,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font={"family": "DM Sans, Arial, sans-serif", "color": INK},
                margin={"l": 25, "r": 25, "t": 65, "b": 25},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
        with b:
            st.markdown(
                f"""
                <div class="content-card">
                  <h3>How to interpret this estimate</h3>
                  <p class="body-copy">For the values entered, the fitted model estimates a score of <b>{score:.1f} out of 100</b>. This number is an output of a statistical model trained on the active dataset. It is not an official result, a guarantee or a judgement of a student's potential.</p>
                  <p class="body-copy">The estimate may change if the training data changes. It can also be less reliable for profiles that differ substantially from the records used to fit the model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            profile_display = pd.DataFrame(
                {
                    "Input": FEATURES,
                    "Entered value": [study, attendance, previous, assignments, sleep],
                }
            )
            st.dataframe(profile_display, hide_index=True, use_container_width=True)
        st.download_button(
            "Download this profile as CSV",
            data=profile_display.to_csv(index=False).encode("utf-8"),
            file_name="aster_prediction_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="content-card">
              <h3>How this lab works</h3>
              <p class="body-copy">1. Set the input values. 2. Select <b>Estimate final score</b>. 3. Review the estimate and its context. The model uses the active dataset, so uploaded data can produce different results from the included demonstration data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 3: STUDENT PDF REPORT
# ============================================================
elif page == "Student PDF Report":
    section_intro(
        "03 — INDIVIDUAL REVIEW",
        "Student PDF Report",
        "Upload a student's academic report card, review the extracted text, verify any detected marks, and create a subject-wise performance summary. PDF text extraction is a best-effort aid; always confirm the values against the original document.",
    )
    if not st.session_state.get("aster_pdf_text"):
        st.info("Upload a text-based report-card PDF in the sidebar to begin. If the PDF is a scanned image, convert it with OCR first or enter marks manually below.")
        st.session_state.setdefault("aster_pdf_marks", pd.DataFrame(columns=["Subject", "Marks", "Out of", "Percentage"]))
        with st.form("manual_report_form"):
            student_name = st.text_input("Student name (optional)", value="Student")
            st.caption("Enter subject marks as percentages. You can use this option when the PDF is scanned or text extraction misses the marks.")
            manual_subjects = st.text_area("Subjects and percentages (one per line)", placeholder="Mathematics, 85\nPhysics, 78\nEnglish, 92")
            manual_submit = st.form_submit_button("Create performance report")
        if manual_submit:
            rows=[]
            for line in manual_subjects.splitlines():
                parts=[x.strip() for x in re.split(r"[,;:]", line, maxsplit=1)]
                if len(parts)==2:
                    try:
                        score=float(parts[1].replace("%", ""))
                        if 0 <= score <= 100:
                            rows.append({"Subject":parts[0], "Marks":score, "Out of":100.0, "Percentage":score})
                    except ValueError:
                        pass
            if not rows:
                st.error("Enter at least one valid subject and percentage, for example: Mathematics, 85")
            else:
                st.session_state.aster_pdf_marks=pd.DataFrame(rows)
                st.session_state.aster_pdf_student_name=student_name.strip() or "Student"
                st.session_state.aster_manual_report=True
                st.rerun()
    else:
        st.markdown(f"**Uploaded file:** `{st.session_state.get('aster_pdf_name', 'PDF report')}`")
        st.markdown("### Review the extracted document")
        extracted_text = st.session_state.aster_pdf_text
        with st.expander("Show extracted PDF text", expanded=True):
            st.text_area("Extracted text (read-only source preview)", value=extracted_text[:20000], height=260, disabled=True)
            if len(extracted_text) > 20000:
                st.caption("Preview limited to the first 20,000 characters.")
        if not extracted_text.strip():
            st.warning("No text was found in this PDF. It may be scanned. Use OCR or enter marks manually below.")
        with st.form("student_pdf_review_form"):
            student_name = st.text_input("Student name", value=st.session_state.get("aster_pdf_student_name", "Student"))
            st.markdown("#### Verify detected subject marks")
            detected = st.session_state.get("aster_pdf_marks", pd.DataFrame(columns=["Subject", "Marks", "Out of", "Percentage"]))
            if detected.empty:
                st.info("Aster could not confidently detect subject marks automatically. Add the marks manually in the table below.")
                detected = pd.DataFrame([{"Subject":"Mathematics", "Marks":0.0, "Out of":100.0, "Percentage":0.0}])
            edited = st.data_editor(
                detected,
                num_rows="dynamic",
                use_container_width=True,
                key="aster_pdf_marks_editor",
                column_config={
                    "Subject": st.column_config.TextColumn("Subject", required=True),
                    "Marks": st.column_config.NumberColumn("Marks earned", min_value=0.0, max_value=1000.0, step=1.0, required=True),
                    "Out of": st.column_config.NumberColumn("Maximum marks", min_value=1.0, max_value=1000.0, step=1.0, required=True),
                    "Percentage": st.column_config.NumberColumn("Percentage (auto-update after submit)", min_value=0.0, max_value=100.0, step=0.1),
                },
            )
            submitted = st.form_submit_button("Generate student performance report")
        if submitted:
            edited = edited.copy()
            edited["Marks"] = pd.to_numeric(edited["Marks"], errors="coerce")
            edited["Out of"] = pd.to_numeric(edited["Out of"], errors="coerce")
            edited = edited.dropna(subset=["Subject", "Marks", "Out of"])
            edited = edited[(edited["Subject"].astype(str).str.strip() != "") & (edited["Out of"] > 0)]
            edited["Percentage"] = (edited["Marks"] / edited["Out of"] * 100).clip(0, 100).round(1)
            if edited.empty:
                st.error("Please add at least one subject with valid marks and a maximum mark greater than zero.")
            else:
                st.session_state.aster_pdf_marks = edited.reset_index(drop=True)
                st.session_state.aster_pdf_student_name = student_name.strip() or "Student"
                st.session_state.aster_manual_report = False
                st.session_state.aster_pdf_report_ready = True
                st.rerun()

    # Render a report whenever reviewed marks exist and the user has submitted or used manual entry.
    marks = st.session_state.get("aster_pdf_marks", pd.DataFrame())
    report_ready = st.session_state.get("aster_pdf_report_ready", False) or st.session_state.get("aster_manual_report", False)
    if report_ready and not marks.empty:
        marks = marks.copy()
        if "Percentage" not in marks.columns:
            marks["Percentage"] = pd.to_numeric(marks["Marks"], errors="coerce")
        marks["Percentage"] = pd.to_numeric(marks["Percentage"], errors="coerce")
        marks = marks.dropna(subset=["Percentage"])
        if not marks.empty:
            average = float(marks["Percentage"].mean())
            best = marks.loc[marks["Percentage"].idxmax()]
            needs_support = marks[marks["Percentage"] < 60].sort_values("Percentage")
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Average across listed subjects", f"{average:.1f}%", "Calculated from reviewed marks")
            with c2: metric_card("Strongest subject", str(best["Subject"]), f"{float(best['Percentage']):.1f}%")
            with c3: metric_card("Subjects reviewed", str(len(marks)), "Only subjects present in this report")
            st.markdown("### Subject-wise performance")
            chart = px.bar(marks.sort_values("Percentage", ascending=False), x="Subject", y="Percentage", text="Percentage", title="Verified marks by subject", range_y=[0, 105], color="Percentage", color_continuous_scale=["#E8C7A0", "#B9D4C1", "#1D5B45"])
            chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            chart.update_yaxes(title_text="Percentage", ticksuffix="%", dtick=10)
            chart.update_layout(coloraxis_showscale=False, dragmode="pan")
            st.plotly_chart(style_plot(chart, 450), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
            if average >= 85:
                overall = "Strong performance across the subjects listed. Maintain consistent revision and keep reviewing any topics where marks were lost."
            elif average >= 70:
                overall = "Good overall progress, with room to strengthen selected topics. Use subject-level results to prioritise revision."
            elif average >= 50:
                overall = "Developing performance. A focused revision routine and regular practice may help strengthen understanding."
            else:
                overall = "The listed marks suggest that additional academic support and a manageable subject-by-subject study plan may be useful."
            st.markdown("### Performance interpretation")
            st.markdown(f"**Student:** {st.session_state.get('aster_pdf_student_name', 'Student')}")
            st.markdown(overall)
            if not needs_support.empty:
                st.markdown("**Subjects to prioritise for review:** " + ", ".join(f"{r['Subject']} ({float(r['Percentage']):.1f}%)" for _, r in needs_support.iterrows()) + ".")
            else:
                st.markdown("No listed subject is below 60%; continue to review mistakes and keep skills balanced.")
            st.dataframe(marks, use_container_width=True, hide_index=True)
            report_text = [f"ASTER — STUDENT PERFORMANCE REPORT", f"Student: {st.session_state.get('aster_pdf_student_name', 'Student')}", f"Average across listed subjects: {average:.1f}%", f"Strongest subject: {best['Subject']} ({float(best['Percentage']):.1f}%)", "", "Subject results:"]
            report_text += [f"- {r['Subject']}: {float(r['Marks']):g}/{float(r['Out of']):g} ({float(r['Percentage']):.1f}%)" for _, r in marks.iterrows()]
            report_text += ["", "Interpretation: " + overall, "", "Note: This report summarises marks reviewed by the user. It is not a diagnosis, a guarantee of future results, or a substitute for a teacher's assessment."]
            st.download_button("Download performance report (TXT)", data="\n".join(report_text).encode("utf-8"), file_name="aster_student_performance_report.txt", mime="text/plain")
            st.download_button("Download verified subject marks (CSV)", data=marks.to_csv(index=False).encode("utf-8"), file_name="aster_verified_subject_marks.csv", mime="text/csv")
            st.caption("Privacy note: Use fictional or consented student data. Review extracted values carefully before relying on the report.")

# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    section_intro(
        "03 — DATA INSPECTION",
        "Dataset Explorer",
        "Explore the records currently being used by Aster. Data inspection is an important part of any modelling workflow: before trusting a model, it is worth understanding the rows, the variables, the ranges and the missing values that shaped it.",
    )
    st.markdown(
        f'<div class="callout"><b>Current source:</b> {source_label}. This workspace currently contains {len(data):,} complete records.</div>',
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        metric_card("Rows", f"{len(data):,}", "Records after cleaning")
    with d2:
        metric_card("Input variables", str(len(FEATURES)), "Predictors used by model")
    with d3:
        metric_card("Missing cells", f"{int(data[REQUIRED_COLUMNS].isna().sum().sum()):,}", "In the active cleaned data")
    with d4:
        metric_card("Score range", f"{data[TARGET].min():.0f}–{data[TARGET].max():.0f}", "Observed final scores")

    tab_preview, tab_summary, tab_corr = st.tabs(["Records", "Summary statistics", "Relationships"])
    with tab_preview:
        st.markdown("### Inspect records")
        st.markdown(
            '<p class="small-note">Use the search box to focus on a subset of records. Values are shown as stored in the active dataset.</p>',
            unsafe_allow_html=True,
        )
        search = st.text_input("Filter rows by a value", placeholder="For example: 80")
        preview = data.copy()
        if search.strip():
            mask = preview.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
            preview = preview.loc[mask]
        st.caption(f"Showing {len(preview):,} of {len(data):,} rows.")
        st.dataframe(preview, hide_index=True, use_container_width=True, height=420)
        st.download_button(
            "Download active dataset as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="aster_active_dataset.csv",
            mime="text/csv",
        )
    with tab_summary:
        st.markdown("### Summary statistics")
        summary = data[REQUIRED_COLUMNS].describe().T.reset_index().rename(columns={"index": "Variable"})
        summary = summary.round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="body-copy">The mean is the arithmetic average; the median is the midpoint after sorting; and the standard deviation describes how spread out the values are. These summaries help identify unusual ranges, but they do not explain why individual records differ.</div>',
            unsafe_allow_html=True,
        )
    with tab_corr:
        st.markdown("### Correlation between numeric variables")
        corr = data[REQUIRED_COLUMNS].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#B85A4C", "#F8F8F2", "#1D5B45"],
            zmin=-1,
            zmax=1,
            title="Correlation matrix",
            labels={"color": "Correlation"},
        )
        fig.update_traces(textfont={"size": 12, "color": "#183A2F"})
        fig.update_layout(coloraxis_colorbar={"tickfont": {"color": INK}, "title": {"text": "Correlation", "font": {"color": INK}}})
        st.plotly_chart(style_plot(fig, 610), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
        st.markdown(
            '<div class="small-note">Correlation ranges from −1 to +1. Positive values indicate that two variables tend to increase together; negative values indicate that one tends to decrease as the other increases. Correlation alone does not establish causation.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================
elif page == "Model Evaluation":
    section_intro(
        "04 — MODEL VALIDATION",
        "Model Evaluation",
        "A model should be examined, not simply celebrated because it produces a prediction. This page reports held-out performance metrics and compares the model's estimates with the observed scores in the test split. These measurements help describe model error, but they do not guarantee future accuracy.",
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        metric_card("MAE", f"{evaluation['mae']:.2f}", "Average absolute error, in score points")
    with e2:
        metric_card("RMSE", f"{evaluation['rmse']:.2f}", "Penalises larger errors more strongly")
    with e3:
        r2_text = f"{evaluation['r2']:.3f}" if np.isfinite(evaluation["r2"]) else "N/A"
        metric_card("R²", r2_text, "Relative fit measure; interpret in context")

    quote_card(
        "A responsible model is not the one that claims certainty. It is the one that makes its limits visible.",
        "ASTER / MODEL REVIEW NOTE",
    )

    actual = evaluation["y_test"]
    predicted = evaluation["predictions"]
    eval_df = pd.DataFrame({"Observed score": actual, "Predicted score": predicted})
    eval_df["Residual"] = eval_df["Observed score"] - eval_df["Predicted score"]

    st.markdown("### Observed scores versus model estimates")
    fig = px.scatter(
        eval_df,
        x="Observed score",
        y="Predicted score",
        opacity=0.78,
        color_discrete_sequence=[GREEN_2],
        hover_data=["Residual"],
        title="Prediction comparison on held-out records",
    )
    fig.update_layout(dragmode="pan")
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=100, y1=100,
        line={"color": GOLD, "width": 2, "dash": "dash"},
    )
    fig.update_xaxes(range=[0, 105], dtick=10, title_text="Observed score (out of 100)")
    fig.update_yaxes(range=[0, 105], dtick=10, title_text="Predicted score (out of 100)")
    st.plotly_chart(style_plot(fig, 520), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
    st.markdown(
        '<div class="small-note">The dashed diagonal represents perfect agreement. Points closer to the line indicate smaller differences between observed and predicted values. Points farther away represent larger prediction errors.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Residual distribution")
        residual_fig = px.histogram(
            eval_df,
            x="Residual",
            nbins=18,
            title="Observed minus predicted score",
            color_discrete_sequence=[GOLD],
        )
        residual_fig.update_layout(dragmode="pan")
        residual_fig.update_xaxes(title_text="Residual (score points)", dtick=5)
        residual_fig.update_yaxes(title_text="Number of records", rangemode="tozero")
        st.plotly_chart(style_plot(residual_fig, 410), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})
    with right:
        st.markdown("### Relative feature importance")
        imp = feature_importance.sort_values("Relative importance", ascending=True)
        imp_fig = px.bar(
            imp,
            x="Relative importance",
            y="Input",
            orientation="h",
            text=imp["Relative importance"].map(lambda x: f"{x:.3f}"),
            color="Relative importance",
            color_continuous_scale=["#D4E5D5", "#6A9C7F", "#1D5B45"],
            title="Inputs used by the fitted model",
        )
        imp_fig.update_layout(dragmode="pan")
        imp_fig.update_traces(textposition="outside", cliponaxis=False)
        imp_fig.update_xaxes(title_text="Relative importance", tickformat=".2f", dtick=0.05)
        imp_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_plot(imp_fig, 410), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True, "responsive": True})

    st.markdown(
        """
        <div class="content-card">
          <h3>How to read these metrics</h3>
          <p class="body-copy"><b>MAE</b> is the average absolute difference between predictions and observed scores. <b>RMSE</b> also measures prediction error, but gives relatively more weight to larger misses. <b>R²</b> compares the model with a simple baseline based on the average target value; its interpretation depends on the data and the evaluation setup.</p>
          <p class="body-copy">A good evaluation process also asks whether the dataset is representative, whether the test set is large enough, whether information leaks between training and testing, and whether the intended use is appropriate. This project is a learning demonstration, not a validated school assessment tool.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download evaluation records as CSV",
        data=eval_df.round(3).to_csv(index=False).encode("utf-8"),
        file_name="aster_model_evaluation.csv",
        mime="text/csv",
    )

# ============================================================
# PAGE 6: ABOUT ASTER
# ============================================================
elif page == "About Aster":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">05 — PURPOSE, METHOD & RESPONSIBILITY</div>
          <h1>About Aster</h1>
          <p>
            Aster is a student-built academic intelligence prototype designed to make a
            basic machine-learning workflow understandable and inspectable. It combines
            dataset exploration, interactive prediction, visual analysis and model evaluation
            in one interface. Its purpose is not to reduce students to scores; it is to
            demonstrate how data tools can help people ask more precise questions about
            learning, while remaining honest about uncertainty and context.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quote_card(
        "Education is not a leaderboard. Every data point belongs to a person, and every person has a story the dataset cannot fully capture.",
        "ASTER / GUIDING PRINCIPLE",
    )

    st.markdown('<div class="section-kicker">THE IDEA</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="body-copy">
        Academic outcomes are shaped by many overlapping conditions: preparation, access to
        resources, classroom experience, consistency, wellbeing, assessment design and
        circumstances outside school. A dataset can represent only a limited slice of this
        picture. Aster starts with a small set of measurable inputs—study hours, attendance,
        previous scores, assignment completion and sleep—and gives users a place to inspect
        how those inputs relate to a recorded final score.

        The interface is intentionally organised as a research workspace rather than a single
        prediction box. The Overview page introduces the dataset and its patterns; the
        Performance Lab lets users test a hypothetical profile; the Dataset Explorer supports
        inspection and download; and Model Evaluation makes the model's errors and metrics
        visible. Each section is intended to encourage interpretation, not blind trust.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">THE TECHNICAL APPROACH</div>', unsafe_allow_html=True)
    tech1, tech2 = st.columns(2, gap="large")
    with tech1:
        st.markdown(
            """
            <div class="content-card">
              <h3>Tools and methods</h3>
              <p class="body-copy"><b>Python</b> provides the application logic. <b>Streamlit</b> creates the interactive interface. <b>Pandas</b> and <b>NumPy</b> handle tabular data and numerical operations. <b>Scikit-learn</b> fits a Random Forest regression model and calculates evaluation metrics. <b>Plotly</b> creates interactive charts with readable axes and hover details.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tech2:
        st.markdown(
            """
            <div class="content-card">
              <h3>What the model does</h3>
              <p class="body-copy">The model learns statistical patterns between the five input variables and the final scores in the active dataset. It is evaluated on a held-out split, then used to estimate scores for hypothetical profiles. If a CSV is uploaded, the data is checked for required columns and numeric values before the model is refitted.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">DATA, ETHICS & LIMITATIONS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="body-copy">
        The built-in dataset is synthetic demonstration data generated by code. It is not a
        real school dataset and should not be presented as evidence about actual students.
        Uploaded data remains within the running application session and is used to generate
        the visualisations and fit the model while the app is running; users should avoid
        uploading personal or identifiable student information to a public demonstration.

        The model may reproduce patterns or limitations present in the data. Feature importance
        is not a causal explanation, and a predicted score is not a promise. This prototype is
        not intended for admissions, grading, discipline, diagnosis, or any high-stakes decision
        about an individual student. Real-world use would require consent, privacy safeguards,
        representative data, careful validation, stakeholder review and ongoing monitoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">WHAT COULD COME NEXT?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content-card">
          <h3>Possible future development</h3>
          <ul>
            <li>Compare multiple regression models using the same held-out evaluation method.</li>
            <li>Add clearer dataset quality checks and a downloadable data-cleaning report.</li>
            <li>Explore confidence intervals or prediction ranges instead of showing a single estimate alone.</li>
            <li>Test the interface with educators and learners to improve accessibility and clarity.</li>
            <li>Document model assumptions and compare performance across appropriate, privacy-protected groups.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quote_card(
        "The goal is not to predict a student's worth. The goal is to make the evidence easier to question.",
        "ASTER / CLOSING NOTE",
    )

# ---------------------------- Footer ----------------------------
st.markdown(
    """
    <div class="footer">
      <b>ASTER / ACADEMIC INTELLIGENCE</b><br>
      A learning-oriented data science prototype built with Python, Streamlit, Pandas,
      NumPy, Scikit-learn and Plotly.<br>
      Demonstration estimates are not guaranteed outcomes. Synthetic data is used by default.
    </div>
    """,
    unsafe_allow_html=True,
)
