import streamlit as st
import pandas as pd
import os
from datetime import datetime
import io
import plotly.express as px
import json

# GCS 관련 클래스 - 모든 스토리지 관련 기능을 캡슐화
class GCSHandler:
    def __init__(self):
        self.client = None
        self.bucket_name = "emotion-index-data"
        self.prefix = "final_anxiety_index"
        self.initialize()
    
    def initialize(self):
        try:
            from google.cloud import storage
            
            # secrets에서 읽도록 수정
            if "google" in st.secrets and "credentials_json" in st.secrets["google"]:
                credential_json = st.secrets["google"]["credentials_json"]
            else:
                credential_json = None
                st.error("❌ Google credentials not found in secrets")
            
            if credential_json:
                with open("/tmp/gcs_key.json", "w") as f:
                    f.write(credential_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs_key.json"
                
                # 클라이언트 초기화
                self.client = storage.Client()
                return True
            else:
                st.error("❌ Credentials not available")
                return False
                
        except Exception as e:
            st.exception(e)
            st.error("Failed to initialize Google Cloud Storage client")
            return False
    
    def list_available_dates(self):
        if not self.client:
            st.error("Storage client not initialized")
            return []
            
        try:
            bucket = self.client.bucket(self.bucket_name)
            blobs = self.client.list_blobs(bucket, prefix=self.prefix + "/")
            dates = set()
            for blob in blobs:
                parts = blob.name.split("/")
                if len(parts) > 2 and parts[1]:
                    dates.add(parts[1])
            return sorted(list(dates), reverse=True)
        except Exception as e:
            st.error(f"Error listing dates: {str(e)}")
            return []
    
    def load_anxiety_index(self, date):
        if not self.client:
            return None
            
        try:
            blob_path = f"{self.prefix}/{date}/anxiety_index_final.csv"
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            if blob.exists():
                data = blob.download_as_text()
                df = pd.read_csv(io.StringIO(data))
                df['Date'] = date  # 날짜 열 추가
                return df
            return None
        except Exception as e:
            st.error(f"Error loading anxiety index: {str(e)}")
            return None
    
    def load_text_file(self, date, filename):
        if not self.client:
            return "(GCS not initialized)"
            
        try:
            blob_path = f"{self.prefix}/{date}/{filename}"
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            if blob.exists():
                return blob.download_as_text()
            return "(File not found)"
        except Exception as e:
            st.error(f"Error loading text file: {str(e)}")
            return "(Error loading file)"
    
    def load_image(self, date, filename):
        if not self.client:
            return None
            
        try:
            blob_path = f"{self.prefix}/{date}/{filename}"
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            if blob.exists():
                return blob.download_as_bytes()
            return None
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None

# GCS 핸들러 생성
gcs = GCSHandler()

# 표시용 날짜 형식 변환
def format_date_display(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%b %d, %Y")  # Apr 08, 2025

# --- PAGE SWITCHING ---
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Time Series"])

# --- SIDEBAR ---
st.sidebar.title("📅 Select Date")
available_dates = gcs.list_available_dates()
selected_date = st.sidebar.selectbox("Choose a date:", available_dates if available_dates else ["No dates available"])

# Add creator information at the bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**FSMI Project**  
[GitHub Repository](https://github.com/Michael1004-ship/FSMI)
""")

if page == "Dashboard":
    st.title("📉 Real-time Anxiety Dashboard")

    # Note about time zone
    st.caption("Note: All times shown are in UTC. This dashboard reflects snapshots of sentiment around key US market hours.")

    # Data sources information
    with st.expander("About the Data Sources"):
        st.markdown("""
        ### News Sources
        The news data is collected from top financial news outlets, including:
        - The New York Times (nytimes.com)
        - Wall Street Journal (wsj.com)
        - Bloomberg (bloomberg.com)
        - CNBC (cnbc.com)
        - CNN (cnn.com)
        - Financial Times (ft.com)
        - Reuters (reuters.com)
        - Yahoo Finance (finance.yahoo.com)
        - Forbes (forbes.com)
        - MarketWatch (marketwatch.com)
        
        ### Reddit Communities
        The social media data is collected from the following subreddits:
        
        **Economics & Finance Related:**
        - r/economics, r/economy, r/MacroEconomics, r/EconMonitor
        - r/finance, r/investing, r/financialindependence, r/personalfinance
        - r/wallstreetbets, r/stocks, r/StockMarket, r/dividends
        
        All data is analyzed using natural language processing models to extract sentiment patterns and compute the anxiety index.
        """)

    # ① 오늘의 Anxiety Index - 주요 지표
    df_index = gcs.load_anxiety_index(selected_date)
    if df_index is not None and not df_index.empty:
        # 컬럼명 확인
        anxiety_col = "Anxiety Index" if "Anxiety Index" in df_index.columns else "anxiety_index"
        
        # Total Anxiety Index (크게 표시)
        if "Type" in df_index.columns and "Total" in df_index["Type"].values:
            total_row = df_index[df_index["Type"] == "Total"]
            total_score = float(total_row[anxiety_col].values[0])
            st.markdown("## 📈 Total Anxiety Index")
            st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{total_score:.2f}</h1>", unsafe_allow_html=True)
            
            # 수식 설명 접을 수 있는 섹션 수정
            with st.expander("🧠 How is the Anxiety Index calculated?"):
                st.markdown("### 🧮 Full Calculation Process")
                st.markdown("The **Anxiety Index** is calculated from financial news and Reddit in three steps:")

                st.markdown("#### 📰 Step 1: News Component")
                st.markdown("1. Compute raw news anxiety index (volume × intensity):")
                st.latex(r"News\_Anxiety_i = Ratio \times (Avg\_Negative\_Score_i)^{1.5}")

                st.markdown("2. Convert to Z-score:")
                st.latex(r"Z_i = \frac{News\_Anxiety_i - \mu}{\sigma}")

                st.markdown("3. Clip and exponentiate:")
                st.latex(r"Z\_clipped = \text{clip}(Z_i, -3, 3)")
                st.latex(r"News\_Component = \exp(Z\_clipped)")

                st.markdown("#### 💬 Step 2: Reddit Component")
                st.markdown("1. Get Z-scores from FinBERT and RoBERTa models:")
                st.latex(r"FinBERT\_Z_i,\quad RoBERTa\_Z_i")

                st.markdown("2. Combine and exponentiate:")
                st.latex(r"Reddit\_Z = \exp\left(\frac{FinBERT\_Z_i + RoBERTa\_Z_i}{2}\right)")

                st.markdown("3. Clip:")
                st.latex(r"Reddit\_Z\_clipped = \text{clip}(Reddit\_Z, -3, 3)")

                st.markdown("#### 🧩 Step 3: Final Index")
                st.latex(r"Total\_Anxiety_i = \exp\left(0.3 \cdot News\_Component + 0.7 \cdot Reddit\_Z\_clipped\right)")

                st.caption("Note: `Ratio`, `Avg Score`, and `Std` come from each day's summary statistics.")
            
            st.markdown("---")
        
        # ② News와 Reddit Combined만 작게 표시
        st.markdown("### 📊 Component Indexes")
        
        col1, col2 = st.columns(2)
        
        # News Anxiety Index
        if "Type" in df_index.columns and "News" in df_index["Type"].values:
            news_row = df_index[df_index["Type"] == "News"]
            news_score = float(news_row[anxiety_col].values[0])
            with col1:
                st.metric("News Anxiety Index", f"{news_score:.2f}")
        
        # Reddit Combined Anxiety Index
        if "Type" in df_index.columns:
            # "Reddit_Combined" 또는 "Reddit Combined" 찾기
            reddit_type = None
            for type_name in df_index["Type"].values:
                if "reddit" in str(type_name).lower() and "combined" in str(type_name).lower():
                    reddit_type = type_name
                    break
            
            if reddit_type:
                reddit_row = df_index[df_index["Type"] == reddit_type]
                reddit_score = float(reddit_row[anxiety_col].values[0])
                with col2:
                    st.metric("Reddit Combined Anxiety Index", f"{reddit_score:.2f}")
        
        # 추가 옵션: 원본 데이터 표시
        if st.checkbox("Show all components"):
            st.dataframe(df_index)
            
            # 용어 설명 부분을 여기로 이동
            with st.expander("ℹ️ What are 'Ratio', 'Avg Score', and 'Std'?"):
                st.markdown("""
### 📘 Component Terminology

These three components help explain how the **Anxiety Index** is calculated for the following sources:
- `News`
- `Reddit_FinBERT`
- `Reddit_RoBERTa`

---

| Term | Meaning | Applies to |
|------|---------|------------|
| **Ratio** | The proportion of documents classified as **negative** out of the total (e.g., 0.40 = 40% negative). | News & Reddit |
| **Avg Score** | The average **negative sentiment score** of the documents identified as negative only. A higher score means stronger negative tone. | News & Reddit |
| **Std** (Standard Deviation) | The degree of variation in the individual negative sentiment scores. A higher Std implies more emotional volatility. | News & Reddit |
""")

    else:
        st.warning("Anxiety index not available for this date.")

    # ② 감정 시각화
    st.markdown("### 🖼 Emotion Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**News**")
        img1 = gcs.load_image(selected_date, "news_umap_plot.png")
        img2 = gcs.load_image(selected_date, "news_emotion_distribution.png")
        if img1: st.image(img1, caption="News UMAP")
        if img2: st.image(img2, caption="News Emotion Distribution")

    with col2:
        st.markdown("**Reddit**")
        img3 = gcs.load_image(selected_date, "reddit_umap_plot.png")
        img4 = gcs.load_image(selected_date, "reddit_emotion_distribution.png")
        if img3: st.image(img3, caption="Reddit UMAP")
        if img4: st.image(img4, caption="Reddit Emotion Distribution")

    # ③ GPT 보고서
    st.markdown("### 📄 GPT Emotion Report")
    report = gcs.load_text_file(selected_date, "gpt_report_combined.txt")
    st.text_area("GPT Report", value=report, height=400, key="report")

    # ④ Appendix
    if st.checkbox("📑 Show Appendix (Representative Sentences)"):
        appendix = gcs.load_text_file(selected_date, "gpt_report_appendix.txt")
        st.text_area("Appendix", value=appendix, height=500, key="appendix")

elif page == "Time Series":
    st.title("Anxiety Index Time Series")

    # Note about time zone and market hours
    st.caption("Note: All times shown are in UTC. This dashboard reflects snapshots of sentiment around key US market hours.")
    
    # Data sources information
    with st.expander("About the Data Sources"):
        st.markdown("""
        ### News Sources
        The news data is collected from top financial news outlets, including:
        - The New York Times (nytimes.com)
        - Wall Street Journal (wsj.com)
        - Bloomberg (bloomberg.com)
        - CNBC (cnbc.com)
        - CNN (cnn.com)
        - Financial Times (ft.com)
        - Reuters (reuters.com)
        - Yahoo Finance (finance.yahoo.com)
        - Forbes (forbes.com)
        - MarketWatch (marketwatch.com)
        
        ### Reddit Communities
        The social media data is collected from the following subreddits:
        
        **Economics & Finance Related:**
        - r/economics, r/economy, r/MacroEconomics, r/EconMonitor
        - r/finance, r/investing, r/financialindependence, r/personalfinance
        - r/wallstreetbets, r/stocks, r/StockMarket, r/dividends
        
        All data is analyzed using natural language processing models to extract sentiment patterns and compute the anxiety index.
        """)

    # Track loaded dates
    processed_dates = set()
    all_data = []
    
    for d in available_dates:
        df = gcs.load_anxiety_index(d)
        if df is not None:
            # Add time information for twice daily data
            if 'timestamp' not in df.columns:
                # Check if this date has already been processed
                if d in processed_dates:
                    df['timestamp'] = f"{d} 21:00:00"  # Market closing time (evening)
                else:
                    df['timestamp'] = f"{d} 14:30:00"  # Market opening time (morning)
                    processed_dates.add(d)  # Record the date
            all_data.append(df)

    if all_data:
        full_df = pd.concat(all_data)
        
        # Verify column name
        anxiety_col = "Anxiety Index" if "Anxiety Index" in full_df.columns else "anxiety_index"
        
        # Basic data filtering
        full_df = full_df.dropna(subset=["Type"])
        
        # Create simple dataframe with time information
        chart_data = []
        
        for date in full_df["Date"].unique():
            date_df = full_df[full_df["Date"] == date]
            
            # There might be multiple timestamps for the same date
            if "timestamp" in date_df.columns:
                timestamps = date_df["timestamp"].unique()
            else:
                timestamps = [f"{date} 00:00:00"]
            
            for ts in timestamps:
                if "timestamp" in date_df.columns:
                    ts_df = date_df[date_df["timestamp"] == ts]
                else:
                    ts_df = date_df
                
                # Find Total value
                total_row = ts_df[ts_df["Type"].str.lower() == "total"]
                total_val = float(total_row[anxiety_col].iloc[0]) if not total_row.empty else None
                
                # Find News value
                news_row = ts_df[ts_df["Type"].str.lower() == "news"]
                news_val = float(news_row[anxiety_col].iloc[0]) if not news_row.empty else None
                
                # Find Reddit value (safely)
                reddit_val = None
                
                # Try Reddit Combined first
                reddit_combined = ts_df[
                    ts_df["Type"].str.lower().str.contains("reddit", na=False) & 
                    ts_df["Type"].str.lower().str.contains("combined", na=False)
                ]
                
                if not reddit_combined.empty:
                    reddit_val = float(reddit_combined[anxiety_col].iloc[0])
                else:
                    # Just look for Reddit rows
                    reddit_row = ts_df[ts_df["Type"].str.lower().str.contains("reddit", na=False)]
                    if not reddit_row.empty:
                        reddit_val = float(reddit_row[anxiety_col].iloc[0])
                
                # Distinguish between morning and evening
                is_closing = "21:00:00" in ts  # Indicates market closing (evening) data
                
                chart_data.append({
                    "DateTime": pd.to_datetime(ts),
                    "Date": pd.to_datetime(date).date(),
                    "Time": pd.to_datetime(ts).time(),
                    "Total": total_val,
                    "News": news_val,
                    "Reddit": reddit_val,
                    "IsClosing": is_closing  # Flag for closing data
                })
        
        # Prepare chart data
        chart_df = pd.DataFrame(chart_data).sort_values("DateTime")
        
        # Time range selection feature
        st.markdown("### 📊 Time Range Selection")
        time_range = st.radio(
            "Select time range:",
            ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "All Time"],
            horizontal=True
        )
        
        # Get the latest date
        if not chart_df.empty:
            latest_date = chart_df["DateTime"].max()
            
            # Filter based on time range
            if time_range == "1 Day":
                start_date = latest_date - pd.Timedelta(days=1)
                filtered_df = chart_df[chart_df["DateTime"] >= start_date]
                # 1 day includes all data (open+close)
            elif time_range == "1 Week":
                start_date = latest_date - pd.Timedelta(days=7)
                filtered_df = chart_df[chart_df["DateTime"] >= start_date]
                # 1 week includes all data (open+close)
            else:
                # 1 month and longer only includes closing values
                if time_range == "1 Month":
                    start_date = latest_date - pd.Timedelta(days=30)
                elif time_range == "3 Months":
                    start_date = latest_date - pd.Timedelta(days=90)
                elif time_range == "6 Months":
                    start_date = latest_date - pd.Timedelta(days=180)
                else:  # All Time
                    start_date = chart_df["DateTime"].min()
                
                # Apply time range and filter to only closing data
                filtered_df = chart_df[
                    (chart_df["DateTime"] >= start_date) & 
                    (chart_df["IsClosing"] == True)
                ]
            
            # Check if filtered data exists
            if not filtered_df.empty:
                title_range = time_range
                if time_range in ["1 Day", "1 Week"]:
                    time_display = "Showing both market opening (14:30 UTC) and closing (21:00 UTC) data"
                else:
                    time_display = "Showing only market closing (21:00 UTC) data"
                
                st.markdown(f"*Data period: {filtered_df['DateTime'].min().strftime('%Y-%m-%d %H:%M')} to {filtered_df['DateTime'].max().strftime('%Y-%m-%d %H:%M')} UTC*")
                st.markdown(f"*{time_display}*")
                
                # Set time display format
                hover_template = "%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}"
                
                # Display data as graph
                fig = px.line(filtered_df, x="DateTime", y=["Total", "News", "Reddit"], markers=True,
                            title=f"Anxiety Index Over Time ({title_range})",
                            labels={"value": "Anxiety Index", "variable": "Type"})
                
                # Adjust x-axis date format - include time for 1 day/1 week, date only for others
                if time_range in ["1 Day", "1 Week"]:
                    date_format = "%m-%d %H:%M"
                else:
                    date_format = "%Y-%m-%d"
                
                # Set marker and hover
                fig.update_traces(mode="lines+markers", marker=dict(size=8), hovertemplate=hover_template)
                fig.update_layout(
                    hovermode="x unified", 
                    xaxis_title="", 
                    yaxis_title="Anxiety Index",
                    xaxis=dict(
                        tickformat=date_format
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to show raw data
                if st.checkbox("Show raw data"):
                    display_cols = [col for col in filtered_df.columns if col != "IsClosing"]
                    st.dataframe(filtered_df[display_cols])
            else:
                st.warning(f"No data available for the selected time range ({time_range}).")
        else:
            st.warning("Could not create time series chart from available data.")
    else:
        st.warning("No data available.")
