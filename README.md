# 🤖 Autonomous AI Job Hunter

A high-performance, serverless AI agent that automates the "search and filter" phase of job hunting. It monitors LinkedIn in real-time, evaluates roles using **Gemini 2.5 Flash**, and only alerts you when a perfect match is found.

## 🎯 The Purpose
The manual process of job hunting is filled with noise. This project was built to transform a passive search into an active, high-signal pipeline. It handles the scraping, reading, and qualification of hundreds of jobs so you only spend time applying to the top 1%.

## 🚀 Key Use Cases
- **Real-Time Market Monitoring:** Automatically scans for "Python Backend" roles in Bengaluru every 20 minutes.
- **Instant CV/JD Gap Analysis:** Uses LLMs to identify specific skills missing from your CV for a given role (e.g., "Missing Kubernetes and AWS experience").
- **Deduplicated Alerting:** Never notifies you of the same job twice, even across multiple runs, using cloud-based state management.
- **Low-Latency Recruiting:** Be the first to apply by receiving Telegram alerts seconds after a job is posted.

## 🛠️ How It Works
1. **Scrape:** Connects to LinkedIn's Guest API to fetch the absolute newest jobs (`sortBy=DD`).
2. **Batch:** Groups jobs into batches of 6 to optimize API costs and avoid rate limits.
3. **Analyze:** Gemini 2.5 Flash compares the Job Description against your CV text using strict JSON schemas.
4. **Notify:** If a job exceeds a **75% match threshold**, a formatted HTML report is sent to your Telegram.
5. **Persist:** Evaluation history is saved to a private GitHub Gist, acting as a permanent cloud database.

## ⚡ Technical Stack
- **AI:** Google Gemini 2.5 Flash (via `google-genai`)
- **Validation:** Pydantic (Structured Outputs)
- **Scraping:** BeautifulSoup4 & Requests
- **Automation:** GitHub Actions (Cron-based)
- **Database:** GitHub Gist API
- **Notifications:** Telegram Bot API