import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="B2B Sales Intelligence Dashboard", layout="wide")

# ---------------- LOAD ----------------
@st.cache_data
def load():
    df = pd.read_csv("Amazon Sale Report.csv", low_memory=False)
    return df

df = load()

# ---------------- CLEAN ----------------
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Amount","Qty","Date"])

# ---------------- HEADER ----------------
st.title("📊 B2B Sales Intelligence Dashboard")
st.caption("Executive • Regression • Future Forecast • Classification • Trends • Root Cause")

# ---------------- KPI ----------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Orders", f"{len(df):,}")
c2.metric("Total Sales", f"₹{df['Amount'].sum():,.0f}")
c3.metric("Average Order", f"₹{df['Amount'].mean():,.0f}")
c4.metric("States", df["ship-state"].nunique())

tabs = st.tabs([
    "Executive Dashboard",
    "Model A — Regression",
    "Future Sales Forecast",
    "Model B — Classification",
    "State Trends",
    "Root Cause"
])

# ======================================================
# ✅ TAB 1 — RICH EXECUTIVE DASHBOARD
# ======================================================

with tabs[0]:

    st.subheader("📈 Monthly Sales Trend")

    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month")["Amount"].sum().reset_index()

    fig = px.area(monthly, x="Month", y="Amount", color_discrete_sequence=["#00D4FF"])
    st.plotly_chart(fig, use_container_width=True)

    cA,cB = st.columns(2)

    with cA:
        st.subheader("🏆 Top Categories by Sales")
        cat = df.groupby("Category")["Amount"].sum().nlargest(8).reset_index()
        fig = px.bar(cat, x="Category", y="Amount", color="Amount")
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("📦 Order Status Distribution")
        stat = df["Status"].value_counts().reset_index()
        stat.columns = ["Status","Count"]
        fig = px.pie(stat, names="Status", values="Count", hole=.55)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ✅ TAB 2 — FEATURE REGRESSION
# ======================================================

with tabs[1]:

    st.subheader("Linear Regression — Predict Order Amount")

    mdf = df[["Qty","Amount","Size","Category","ship-state"]].dropna()

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    mdf["SizeE"] = le1.fit_transform(mdf["Size"])
    mdf["CatE"] = le2.fit_transform(mdf["Category"])
    mdf["StateE"] = le3.fit_transform(mdf["ship-state"])

    X = mdf[["Qty","SizeE","CatE","StateE"]]
    y = mdf["Amount"]

    model = LinearRegression().fit(X,y)

    st.success("Regression model trained")

    q = st.number_input("Quantity",1,50,3)
    s = st.selectbox("Size", mdf["Size"].unique())
    c = st.selectbox("Category", mdf["Category"].unique())
    stt = st.selectbox("State", mdf["ship-state"].unique())

    pred = model.predict([[q,
        le1.transform([s])[0],
        le2.transform([c])[0],
        le3.transform([stt])[0]]])[0]

    st.metric("Predicted Amount", f"₹{pred:,.0f}")

# ======================================================
# ✅ TAB 3 — TIME BASED FUTURE FORECAST
# ======================================================

with tabs[2]:

    st.subheader("📅 Future Sales Forecast (Time Based Linear Regression)")

    daily = df.groupby("Date")["Amount"].sum().reset_index()

    daily["DayIndex"] = np.arange(len(daily))

    X = daily[["DayIndex"]]
    y = daily["Amount"]

    tmodel = LinearRegression().fit(X,y)

    future_days = st.slider("Days to predict ahead", 7, 90, 30)

    future_index = np.arange(len(daily), len(daily)+future_days).reshape(-1,1)
    future_pred = tmodel.predict(future_index)

    future_dates = pd.date_range(daily["Date"].max(), periods=future_days+1)[1:]

    fdf = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_pred
    })

    fig = px.line(daily, x="Date", y="Amount", title="Historical Sales")
    fig.add_scatter(x=fdf["Date"], y=fdf["Predicted Sales"], mode="lines", name="Forecast")

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ✅ TAB 4 — CLASSIFICATION
# ======================================================

with tabs[3]:

    st.subheader("Sales Value Classification")

    bins = pd.qcut(df["Amount"], 3, labels=["Low","Medium","High"], duplicates="drop")
    dist = bins.value_counts().reset_index()
    dist.columns=["Class","Count"]

    fig = px.bar(dist, x="Class", y="Count", color="Class")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ✅ TAB 5 — STATE TRENDS
# ======================================================

with tabs[4]:

    st.subheader("State Sales Performance")

    s = df.groupby("ship-state")["Amount"].sum().nlargest(15).reset_index()
    fig = px.bar(s, x="ship-state", y="Amount", color="Amount")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ✅ TAB 6 — ROOT CAUSE
# ======================================================

with tabs[5]:

    st.subheader("Root Cause Indicators")

    cA,cB = st.columns(2)

    with cA:
        fig = px.box(df, x="Category", y="Amount")
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        corr = df[["Qty","Amount"]].corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
