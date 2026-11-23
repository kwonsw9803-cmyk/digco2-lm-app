"""Altair charts (bundled with Streamlit) for the screening map and the parity plot."""

import altair as alt
import pandas as pd

from .screening import STATUS_BASE, STATUS_OTHER, STATUS_TARGET, STATUS_UNTESTED

STATUS_COLORS = {
    STATUS_TARGET: "#c8501f",
    STATUS_UNTESTED: "#2a78d6",
    STATUS_OTHER: "#7d8790",
    STATUS_BASE: "#7d8790",
}
# Shape repeats the status, so it never depends on colour alone.
STATUS_SHAPES = {
    STATUS_TARGET: "circle",
    STATUS_UNTESTED: "triangle-up",
    STATUS_OTHER: "square",
    STATUS_BASE: "diamond",
}


def screening_map(df, x_title, y_title, ink="#16191d"):
    """df columns: Element, x_value, y_value, Status."""
    scale = alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))
    base = alt.Chart(df).encode(
        x=alt.X("x_value:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("y_value:Q", title=y_title, scale=alt.Scale(zero=False)),
    )
    shapes = alt.Scale(domain=list(STATUS_SHAPES), range=list(STATUS_SHAPES.values()))
    legend = alt.Legend(title=None, orient="top", labelLimit=220, columns=2)
    points = base.mark_point(size=120, opacity=1, filled=True).encode(
        color=alt.Color("Status:N", scale=scale, legend=legend),
        shape=alt.Shape("Status:N", scale=shapes, legend=legend),
        tooltip=[
            "Element:N",
            "Status:N",
            alt.Tooltip("x_value:Q", title=x_title),
            alt.Tooltip("y_value:Q", title=y_title),
        ],
    )
    labels = base.mark_text(align="left", dx=7, dy=-7, fontSize=11, color=ink).encode(text="Element:N")
    return (points + labels).properties(height=420)


def parity_plot(result, x_title, ink="#16191d"):
    """result columns: Element, Target, LOOCV_prediction. Equal axes with the y = x line."""
    lo = float(min(result["Target"].min(), result["LOOCV_prediction"].min()))
    hi = float(max(result["Target"].max(), result["LOOCV_prediction"].max()))
    pad = (hi - lo) * 0.08 or 0.1
    dom = [lo - pad, hi + pad]
    xs = alt.X("Target:Q", title=x_title, scale=alt.Scale(domain=dom))
    ys = alt.Y("LOOCV_prediction:Q", title="LOOCV prediction", scale=alt.Scale(domain=dom))

    diag = (
        alt.Chart(pd.DataFrame({"Target": dom, "LOOCV_prediction": dom}))
        .mark_line(strokeDash=[4, 4], color="#7d8790")
        .encode(x=xs, y=ys)
    )
    base = alt.Chart(result).encode(x=xs, y=ys)
    points = base.mark_circle(size=90, color="#2a78d6", opacity=1).encode(
        tooltip=["Element:N", alt.Tooltip("Target:Q", format=".3f"), alt.Tooltip("LOOCV_prediction:Q", format=".3f")]
    )
    labels = base.mark_text(align="left", dx=6, dy=-6, fontSize=11, color=ink).encode(text="Element:N")
    return (diag + points + labels).properties(height=380)
