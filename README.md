### FSMI: Finance Sentiment Market Index

### Overview

FSMI (Finance Sentiment Market Index) is a real-time sentiment indexing system that quantifies anxiety levels in the financial market using large-scale news and social media data.

While traditional financial indicators focus on economic metrics, volatility, and market flows, this project captures the emotional undercurrents of market participants — aiming to detect early signs of instability and provide a new lens for crisis prevention and investor psychology analysis.

### Key Features

News and Reddit data collection (via GDELT and Reddit API)

Sentiment analysis using FinBERT→ Converts positive/negative classifications into an anxiety score based on the proportion and intensity of negative sentiment

Daily and weekly anxiety index generation (separately for news and social media)

Time-lag model between news and social media

Formula:final_index = exp(0.3 * clipped_news_z + 0.7 * clipped_reddit_z).mean()

Each source is normalized via Z-score; then a weighted exponential average (30% news, 70% Reddit) is computed for nonlinear amplification

Fusion of FinBERT and RoBERTa sentiment outputs from Reddit for final score

Emotion structure clustering using RoBERTa embeddings + K-means

Automated cluster labeling and summary reporting via GPT

Real-time emotion monitoring dashboard (built with Streamlit)

### Project Structure

/dashboard/            # Streamlit-based sentiment dashboard
/building_index/       # Scripts for computing the daily anxiety index
/clustering/           # RoBERTa embedding and emotion clustering
/GPT/                  # GPT-based sentiment labeling and reporting
/scripts/              # Preprocessing and shared utilities
README.md              # This document

### How to Run

1. Clone the repository

git clone https://github.com/Michael1004-ship/FSMI.git
cd FSMI

2. Install dependencies

pip install -r requirements.txt

3. Run the index builder

python building_index/FSMI_operator.py

⚠️ Note: Currently includes hardcoded paths. Config file structure will be added in future updates.

### Roadmap

Time series visualization of the anxiety index (via Plotly, Altair)

Expansion into political sentiment indexing using news and social media

### Philosophy

"Emotion is not noise. Emotion is a structural force that moves markets."

Where behavioral economics often treats emotion as temporary deviation,FSMI treats emotion as a structured, quantifiable, and time-sensitive force,offering a new paradigm for real-time financial market analysis.

### Contact

This is a personal research project.Collaboration, academic publication, and productization inquiries are welcome.Please feel free to reach out!

[contact : wjdans1004@naver.com]

### License
This project is licensed for non-commercial research use.  
See [LICENSE](./LICENSE) for details.
