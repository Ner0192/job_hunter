import os
import requests
from google import genai
from pydantic import BaseModel, Field

# --- 1. SETUP & CONFIG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. DEFINE THE OUTPUT SCHEMA USING PYDANTIC ---
class JobMatchEvaluation(BaseModel):
    match_score: int = Field(description="A score from 0 to 100 representing how well the CV matches the JD.")
    missing_critical_skills: list[str] = Field(description="A list of required skills in the JD missing from the CV.")
    verdict: str = Field(description="Must be exactly 'pass' or 'fail' based on a 75% match threshold.")
    summary: str = Field(description="A 1-sentence summary of why this job is a good fit.")

# --- 3. THE CV (Your Base Resume Text) ---
# You can load this from a .txt file, but keeping it inline is fine for V1.
MY_CV = """
Backend Software Engineer with 3+ years of experience.
Expert in Python, FastAPI, Django, Flask, Docker, and REST APIs.
Experience building GenAI pipelines and LLM integrations.
Strong background in system architecture, Pi-hole, and networking.
"""

# --- 4. CORE FUNCTIONS ---
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def evaluate_job(job_title: str, job_description: str) -> JobMatchEvaluation:
    prompt = f"""
    You are an expert technical recruiter. Evaluate the following Job Description against the provided CV.
    
    My CV:
    {MY_CV}
    
    Job Title: {job_title}
    Job Description:
    {job_description}
    """
    
    # Force Gemini to return data matching our Pydantic model
    response = client.models.generate_content(
        model='gemini-3.1-flash', # Flash is fast and cheap for this task
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': JobMatchEvaluation,
        },
    )
    # The SDK automatically parses the JSON into our Pydantic object
    return response.parsed

def fetch_jobs():
    # PLACEHOLDER: Replace this with your actual scraping logic/API call.
    # E.g., Using Proxycurl, Apify, or an RSS feed.
    return [
        {
            "title": "Senior Python Backend Engineer",
            "url": "https://linkedin.com/jobs/view/12345",
            "description": "Looking for a backend dev with 4+ years in Python, FastAPI, and Docker."
        },
        {
            "title": "Frontend React Developer",
            "url": "https://linkedin.com/jobs/view/67890",
            "description": "Must have deep knowledge of React, CSS, and Figma."
        }
    ]

# --- 5. THE MAIN PIPELINE ---
if __name__ == "__main__":
    print("Starting Job Hunt Pipeline...")
    jobs = fetch_jobs()
    
    for job in jobs:
        print(f"Evaluating: {job['title']}")
        evaluation: JobMatchEvaluation = evaluate_job(job['title'], job['description'])
        
        if evaluation.verdict == "pass" and evaluation.match_score >= 75:
            message = (
                f"?? <b>HIGH MATCH DETECTED ({evaluation.match_score}%)</b>\n\n"
                f"<b>Role:</b> {job['title']}\n"
                f"<b>Summary:</b> {evaluation.summary}\n"
                f"<b>Missing Skills:</b> {', '.join(evaluation.missing_critical_skills) or 'None!'}\n\n"
                f"?? <a href='{job['url']}'>Apply Here</a>"
            )
            send_telegram_message(message)
            print("Match found! Telegram sent.")
        else:
            print(f"Skipped. Score: {evaluation.match_score}%")
