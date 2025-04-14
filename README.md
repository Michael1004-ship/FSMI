# FSMI: Financial Sentiment Market Index

## 📌 Overview

**FSMI (Financial Sentiment Market Index)** is a real-time sentiment indexing system that quantifies **market anxiety** using large-scale news and social media data.  
Unlike traditional financial indicators that rely on volatility or price signals, FSMI captures the **emotional architecture of financial discourse** — offering early signals of instability and insight into investor psychology.

FSMI is designed to function as a **leading sentiment indicator**, supporting behavioral analysis, crisis detection, and real-time monitoring of market emotion.

---

## 🔍 Key Features

- **Multisource Sentiment Collection**  
  - News data via GDELT  
  - Reddit data via Reddit API  

- **Model-Driven Emotion Scoring**  
  - FinBERT-based sentiment classification (news & social posts)  
  - RoBERTa embeddings for deep semantic emotion representation (Reddit)

- **Real-Time Index Generation**  
  - Daily and weekly anxiety scores  
  - Independent scoring for news vs. social media  
  - Z-score normalization + nonlinear transformation  

- **Emotion Structure Modeling**  
  - K-means clustering on RoBERTa embeddings  
  - Automated cluster labeling using GPT  
  - Interpretable sentiment archetypes (e.g., fear, distrust, greed)

- **Monitoring & Visualization**  
  - Streamlit-based dashboard (interactive, real-time)  
  - Planned migration to React + Next.js interface

---

## ⚠️ Intellectual Property & Security

> This project is protected by **patent-pending proprietary methodology.**  
> While model types and general workflow are open for research sharing, the **core index calculation logic, weighting strategy, and transformation formula** are not publicly disclosed.  
> Redistribution or re-implementation of the full index methodology is **strictly prohibited**.

---

## 🛠️ Project Structure

