import os
import sys  # Added to handle system exits
import time
import logging
from bs4 import BeautifulSoup
import requests
from google import genai
from pydantic import BaseModel, Field
from google.genai import errors

# from dotenv import load_dotenv  # For local use uncomment this


# --- 0. CONFIGURE LOGGING ---
class ColoredFormatter(logging.Formatter):
    # Define the ANSI color codes (Matching your Bash script!)
    CYAN = "\033[0;36m"
    ORANGE = "\033[38;5;214m"
    RED = "\033[0;31m"
    RESET = "\033[0m"

    # Map each log level to its specific color format
    # We use modern {} formatting. The ^8 tells Python to center the word in 8 spaces.
    FORMATS = {
        logging.INFO: f"{{asctime}} - [{CYAN}{{levelname:^8}}{RESET}] - {{message}}",
        logging.WARNING: f"{{asctime}} - [{ORANGE}{{levelname:^8}}{RESET}] - {{message}}",
        logging.ERROR: f"{{asctime}} - [{RED}{{levelname:^8}}{RESET}] - {{message}}",
        logging.CRITICAL: f"{{asctime}} - [{RED}{{levelname:^8}}{RESET}] - {{message}}",
    }

    def format(self, record):
        # Fetch the color format for the specific log level
        log_fmt = self.FORMATS.get(
            record.levelno, "{asctime} - {levelname:^8} - {message}"
        )

        # CRITICAL: Because we are using {} now, we must add style="{" here!
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)


# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate logs if the script is imported or run multiple times
if logger.hasHandlers():
    logger.handlers.clear()

# Attach the custom colored formatter to the console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# load_dotenv()  # For local use uncomment this

# --- 1. SETUP & CONFIG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GIT_PAT = os.getenv("GIT_PAT")
GIST_ID = os.getenv("GIST_ID")
MATCH_SCORE = int(os.getenv("MATCH_SCORE", 75))
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=GEMINI_API_KEY)


# --- 2. STATE MANAGEMENT (CLOUD GIST DEDUPLICATION) ---
def load_seen_jobs() -> set:
    if not GIT_PAT or not GIST_ID:
        logger.warning("GIT_PAT or GIST_ID missing. Deduplication disabled.")
        return set()

    headers = {
        "Authorization": f"Bearer {GIT_PAT}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        response = requests.get(
            f"https://api.github.com/gists/{GIST_ID}", headers=headers
        )
        response.raise_for_status()
        gist_data = response.json()

        if "seen_jobs.txt" in gist_data.get("files", {}):
            content = gist_data["files"]["seen_jobs.txt"].get("content", "")
            return set(content.splitlines())
        return set()
    except Exception as e:
        logger.error(f"Error loading from Gist: {e}")
        return set()


def save_seen_jobs(all_seen_jobs: set):
    if not GIT_PAT or not GIST_ID:
        return

    headers = {
        "Authorization": f"Bearer {GIT_PAT}",
        "Accept": "application/vnd.github.v3+json",
    }
    new_content = "\n".join(all_seen_jobs)
    payload = {"files": {"seen_jobs.txt": {"content": new_content}}}

    try:
        requests.patch(
            f"https://api.github.com/gists/{GIST_ID}", headers=headers, json=payload
        )
        logger.info("Successfully saved updated seen_jobs.txt to the cloud!")
    except Exception as e:
        logger.error(f"Error updating Gist: {e}")


# --- 3. BATCH OUTPUT SCHEMA ---
class JobMatchEvaluation(BaseModel):
    job_id: str = Field(description="The exact ID of the job being evaluated.")
    match_score: int = Field(description="A score from 0 to 100 representing CV match.")
    missing_critical_skills: list[str] = Field(
        description="Required skills missing from CV."
    )
    verdict: str = Field(
        description=f"Must be exactly 'pass' or 'fail' based on {MATCH_SCORE}% match."
    )
    summary: str = Field(
        description="A 1-sentence summary of why this job is a good fit."
    )


class BatchJobEvaluation(BaseModel):
    evaluations: list[JobMatchEvaluation]


# --- 4. THE CV ---
MY_CV = """
Backend Software Engineer with 3.5+ years of experience. 
Expertise in building secure, scalable, and high-performance systems, including GenAI pipelines and LLM integrations.
Key Skills: Python (Expert),FastAPI, Django, Flask, SQL, Go, Bash, REST APIs, Async Services, Microservices, Event-Driven Architecture, Jinja2, Langchain, Docker, CI/CD, GitHub Actions, Jenkins, Git, PostgreSQL, MySQL, MongoDB, Redis, Firebase, AWS (EC2, Lambda).
AI Experience: Built and integrated GenAI-powered assistants and conversational systems using LLM APIs and NLU models.
"""


# --- 5. CORE FUNCTIONS ---
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


def evaluate_jobs_batch(jobs_batch: list[dict]) -> BatchJobEvaluation | None:
    if not jobs_batch:
        return None

    jobs_text = ""
    for job in jobs_batch:
        jobs_text += f"--- JOB ID: {job['id']} ---\nTitle: {job['title']}\nDescription: {job['description']}\n\n"

    prompt = f"""
    You are an expert technical recruiter. Evaluate the following batch of Job Descriptions against the provided CV.
    
    My CV:
    {MY_CV}
    
    Jobs to Evaluate:
    {jobs_text}
    """

    # Extract the config so we can reuse it for both models
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": BatchJobEvaluation,
        "temperature": 0.1,
    }

    try:
        # 1. Try the primary workhorse model
        response = client.models.generate_content(
            model=MODEL,  # type: ignore
            contents=prompt,
            config=generation_config,  # type: ignore
        )
        return response.parsed  # type: ignore

    except errors.APIError as e:
        # 2. Catch traffic jam errors
        if e.code in [503, 500]:
            logger.warning(
                f"Main model busy ({e.code}). Falling back to {BACKUP_MODEL}..."
            )

            try:
                # Instantly reroute to the fallback model
                fallback_response = client.models.generate_content(
                    model=BACKUP_MODEL,
                    contents=prompt,
                    config=generation_config,  # type: ignore
                )
                logger.info(f"Successfully evaluated using {BACKUP_MODEL} fallback.")
                return fallback_response.parsed  # type: ignore

            except Exception as fallback_error:
                # 3. CRITICAL: If both models fail, raise an error to crash the pipeline
                raise RuntimeError(
                    f"Both primary and backup models failed! Fallback error: {fallback_error}"
                )
        else:
            # For other API errors (like bad keys), crash the pipeline
            raise RuntimeError(f"Unrecoverable Google API Error: {e.message}")

    except Exception as e:
        # Catch standard network timeouts and crash
        raise RuntimeError(f"Unexpected Network Error: {e}")


def fetch_jobs(seen_jobs: set):
    logger.info("Scraping LinkedIn (Sorted by Newest)...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    jobs_data = []

    for start_index in range(0, 50, 25):
        search_url = (
            f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            f"?keywords=Python%20Backend"
            f"&location=Bengaluru"
            f"&geoId=105214831"
            f"&f_TPR=r1200"  # Last 20 mins
            f"&f_E=2,3,4"
            f"&f_WT=2,3"
            f"&sortBy=DD"
            f"&start={start_index}"
        )

        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(
                    f"LinkedIn restricted access (Status {response.status_code})."
                )
                break

            soup = BeautifulSoup(response.text, "html.parser")
            job_cards = soup.find_all("li")
            if not job_cards:
                break

            for card in job_cards:
                base_card = card.find("div", class_="base-card")
                if not base_card:
                    continue

                job_urn = base_card.get("data-entity-urn")
                if not job_urn:
                    continue
                job_id = job_urn.split(":")[-1]  # type: ignore

                # Short-circuit logic: Stop scraping if we hit a known job
                if job_id in seen_jobs:
                    logger.info(
                        f"Encountered already processed job ({job_id}). Catch-up complete."
                    )
                    return jobs_data

                title = card.find("h3", class_="base-search-card__title").text.strip()  # type: ignore
                company = card.find("h4", class_="base-search-card__subtitle").text.strip()  # type: ignore
                logger.info(f"Fetching: {title} at {company}")

                desc_url = (
                    f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
                )
                desc_resp = requests.get(desc_url, headers=headers, timeout=10)
                desc_soup = BeautifulSoup(desc_resp.text, "html.parser")
                desc_box = desc_soup.find("div", class_="description__text")
                description = desc_box.text.strip() if desc_box else "No description."

                jobs_data.append(
                    {
                        "id": job_id,
                        "title": f"{title} @ {company}",
                        "url": f"https://www.linkedin.com/jobs/view/{job_id}",
                        "description": description,
                    }
                )
                time.sleep(3)

        except Exception as e:
            logger.error(f"Error scraping: {e}")
            break

    return jobs_data


# --- 6. THE MAIN PIPELINE ---
if __name__ == "__main__":
    logger.info("Starting Cloud-Optimized Job Hunt Pipeline...")

    seen_jobs = load_seen_jobs()
    newly_evaluated_jobs = set()  # Keep track of what we evaluate this run
    pipeline_failed = False  # Flag to track if the script should crash at the end

    jobs = fetch_jobs(seen_jobs)

    if not jobs:
        logger.info("No new jobs found since last run.")
    else:
        # Group jobs into chunks of 6
        BATCH_SIZE = 6
        job_batches = [
            jobs[i : i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)
        ]

        for batch in job_batches:
            logger.info(f"Evaluating batch of {len(batch)} jobs...")
            try:
                batch_result: BatchJobEvaluation = evaluate_jobs_batch(batch)  # type: ignore
                if not batch_result:
                    continue

                for evaluation in batch_result.evaluations:
                    original_job = next(
                        (j for j in batch if j["id"] == evaluation.job_id), None
                    )

                    if (
                        original_job
                        and evaluation.verdict == "pass"
                        and evaluation.match_score >= MATCH_SCORE
                    ):
                        message = (
                            f"🚨 <b>MATCH: {evaluation.match_score}%</b>\n\n"
                            f"<b>Role:</b> {original_job['title']}\n"
                            f"<b>Summary:</b> {evaluation.summary}\n"
                            f"<b>Missing Skills:</b> {', '.join(evaluation.missing_critical_skills) or 'None!'}\n\n"
                            f"🔗 <a href='{original_job['url']}'>Apply Here</a>"
                        )
                        send_telegram_message(message)
                        logger.info(f"Match found! ({original_job['title']})")
                    else:
                        title = (
                            original_job["title"] if original_job else evaluation.job_id
                        )
                        logger.info(
                            f"Skipped. ({title}) - Score: {evaluation.match_score}%"
                        )

                    # Track this ID so we save it to the cloud later
                    newly_evaluated_jobs.add(evaluation.job_id)
                    seen_jobs.add(evaluation.job_id)

                time.sleep(5)

            except RuntimeError as e:
                logger.error(f"PIPELINE CRASH INITIATED: {e}")
                pipeline_failed = True
                break  # Stop processing any more batches

            except Exception as e:
                logger.error(f"Unexpected Error: {e}")
                pipeline_failed = True
                break  # Stop processing any more batches

    # CRITICAL: Save jobs that WERE successfully evaluated before the crash happened
    if newly_evaluated_jobs:
        logger.info(f"Saving {len(newly_evaluated_jobs)} new jobs to GitHub Gist...")
        save_seen_jobs(seen_jobs)

    # Finally, trigger the hard failure for GitHub
    if pipeline_failed:
        logger.error(
            "Exiting with status code 1. GitHub Action will now mark as FAILED."
        )
        sys.exit(1)

    logger.info("Pipeline run complete.")
