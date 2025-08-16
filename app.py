"""
AI-Powered CareerSync Pro
A comprehensive AI career development platform with 75+ advanced features for resume optimization, interview preparation, skills assessment, and career coaching.
"""

import streamlit as st
import os
import json
import time
import re
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests
import hashlib
import sqlite3
from pathlib import Path
from collections import Counter

# Core data science libraries (optional)
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    HAS_NUMPY = True
except ImportError:
    HAS_PANDAS = False
    HAS_NUMPY = False

# Optional imports with graceful fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyMuPDF as fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import docx2txt
    HAS_DOCX2TXT = True
except ImportError:
    HAS_DOCX2TXT = False

try:
    from docx import Document
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import streamlit_authenticator as stauth
    HAS_STREAMLIT_AUTH = True
except ImportError:
    HAS_STREAMLIT_AUTH = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load environment variables
# --- Configuration ---
st.set_page_config(
    page_title="AI-Powered CareerSync Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/careersync-pro',
        'Report a bug': "https://github.com/your-repo/careersync-pro/issues",
        'About': "# AI-Powered CareerSync Pro\nComprehensive AI career development platform with 75+ advanced features!"
    }
)

# --- Global Variables and Constants ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]
ATS_KEYWORDS = {
    "tech": ["python", "javascript", "react", "node.js", "sql", "aws", "docker", "kubernetes", "git", "agile"],
    "finance": ["excel", "financial modeling", "risk management", "bloomberg", "sql", "python", "tableau"],
    "marketing": ["seo", "google analytics", "social media", "content marketing", "email marketing", "ppc"]
}

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'resume_versions' not in st.session_state:
    st.session_state.resume_versions = []
if 'job_descriptions' not in st.session_state:
    st.session_state.job_descriptions = []
if 'ats_history' not in st.session_state:
    st.session_state.ats_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Load NLP models with error handling
@st.cache_resource
def load_nlp_models():
    """Load and cache NLP models with graceful error handling"""
    nlp = None
    sentence_model = None

    # Try to load spaCy (optional)
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError, Exception):
            nlp = None
    except Exception:
        nlp = None

    # Try to load sentence transformers (optional)
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        sentence_model = None

    return nlp, sentence_model

# Initialize models safely
nlp = None
sentence_model = None

try:
    nlp, sentence_model = load_nlp_models()
except Exception:
    # If models fail to load, continue without them
    nlp = None
    sentence_model = None

# --- Database Functions ---
def init_database():
    """Initialize SQLite database for storing user data"""
    conn = sqlite3.connect('resume_optimizer.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Resume versions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resume_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            version_name TEXT NOT NULL,
            content TEXT NOT NULL,
            ats_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # ATS history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ats_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_content TEXT,
            job_description TEXT,
            ats_score REAL,
            keywords_matched INTEGER,
            total_keywords INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

# --- File Processing Functions ---
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF using available libraries"""
    text = ""

    if HAS_PDFPLUMBER:
        try:
            # Method 1: pdfplumber (better for structured text)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if text.strip():
                return text
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}")

    if HAS_PYMUPDF and not text.strip():
        try:
            # Method 2: PyMuPDF (fallback for complex layouts)
            file.seek(0)
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
            pdf_document.close()
        except Exception as e:
            st.warning(f"PyMuPDF failed: {e}")

    if not text.strip():
        st.error("❌ No PDF processing libraries available. Please install pdfplumber or PyMuPDF.")
        return ""

    return text

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file using available libraries"""
    text = ""

    if HAS_DOCX2TXT:
        try:
            # Method 1: docx2txt (simpler and often more reliable)
            file.seek(0)
            text = docx2txt.process(file)

            if text.strip():
                return text
        except Exception as e:
            st.warning(f"docx2txt failed: {e}")

    if HAS_PYTHON_DOCX and not text.strip():
        try:
            # Method 2: python-docx (fallback)
            file.seek(0)
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.warning(f"python-docx failed: {e}")

    if not text.strip():
        st.error("❌ No DOCX processing libraries available. Please install python-docx or docx2txt.")
        return ""

    return text

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        return content
    except Exception as e:
        st.error(f"Error extracting text from TXT: {e}")
        return ""

def extract_text_from_file(file) -> str:
    """Main function to extract text from uploaded file with comprehensive error handling"""
    if file is None:
        return ""

    try:
        # Validate file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            st.error("❌ File size too large. Please upload files smaller than 10MB.")
            return ""

        file_type = file.type.lower()

        # Validate file type
        if not any(supported_type in file_type for supported_type in ["pdf", "word", "docx", "text", "txt"]):
            st.error(f"❌ Unsupported file type: {file_type}. Please upload PDF, DOCX, or TXT files.")
            return ""

        # Extract text based on file type
        if "pdf" in file_type:
            text = extract_text_from_pdf(file)
        elif "word" in file_type or "docx" in file_type:
            text = extract_text_from_docx(file)
        elif "text" in file_type or "txt" in file_type:
            text = extract_text_from_txt(file)
        else:
            st.error(f"❌ Unsupported file type: {file_type}")
            return ""

        # Validate extracted text
        if not text or len(text.strip()) < 50:
            st.warning("⚠️ Very little text extracted from file. Please check if the file contains readable text.")
            return text

        if len(text) > 50000:
            st.warning("⚠️ Document is very long. Analysis may take longer than usual.")

        return text

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return ""

# --- NLP and Analysis Functions ---
def extract_keywords(text: str, pos_tags: List[str] = None) -> List[str]:
    """Extract keywords from text using spaCy or fallback method"""
    if not text:
        return []

    if nlp is not None:
        try:
            if pos_tags is None:
                pos_tags = ["NOUN", "PROPN", "ADJ"]

            doc = nlp(text.lower())
            keywords = []

            for token in doc:
                if (token.pos_ in pos_tags and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) > 2 and
                    token.text.isalpha()):
                    keywords.append(token.text)

            # Remove duplicates while preserving order
            return list(dict.fromkeys(keywords))
        except Exception:
            pass

    # Fallback method using simple text processing
    import re
    from collections import Counter

    # Simple keyword extraction without NLP
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'then', 'them', 'well', 'were', 'will', 'with', 'have', 'this', 'that', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'over', 'such', 'take', 'than', 'only', 'think', 'work', 'first', 'after', 'back', 'other', 'many', 'where', 'right', 'through', 'before', 'great', 'little', 'world', 'year', 'still', 'should', 'around', 'another', 'came', 'come', 'could', 'every', 'found', 'give', 'going', 'hand', 'high', 'keep', 'last', 'left', 'life', 'live', 'look', 'made', 'most', 'move', 'must', 'name', 'need', 'never', 'place', 'put', 'said', 'same', 'seem', 'show', 'small', 'tell', 'turn', 'used', 'want', 'ways', 'well', 'went', 'what', 'while', 'work', 'would', 'write', 'year', 'your'
    }

    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # Get most common words
    word_counts = Counter(keywords)
    return list(dict.fromkeys([word for word, count in word_counts.most_common(50)]))

def extract_skills(text: str) -> List[str]:
    """Extract technical skills and competencies"""
    skills_patterns = [
        r'\b(?:python|java|javascript|c\+\+|sql|html|css|react|angular|vue)\b',
        r'\b(?:aws|azure|gcp|docker|kubernetes|git|jenkins)\b',
        r'\b(?:machine learning|data science|artificial intelligence|deep learning)\b',
        r'\b(?:project management|agile|scrum|kanban)\b',
        r'\b(?:excel|powerpoint|word|outlook|salesforce|tableau)\b'
    ]

    skills = []
    text_lower = text.lower()

    for pattern in skills_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.extend(matches)

    return list(set(skills))

def calculate_ats_score(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Calculate comprehensive ATS score with detailed breakdown and validation"""
    # Input validation
    if not resume_text or not jd_text:
        return {
            "overall_score": 0,
            "breakdown": {
                "keyword_match": 0,
                "skills_match": 0,
                "semantic_similarity": 0,
                "format_structure": 0,
                "common_keywords": [],
                "common_skills": [],
                "missing_keywords": [],
                "missing_skills": []
            },
            "recommendations": ["Please provide both resume and job description"]
        }

    if len(resume_text.strip()) < 50:
        return {
            "overall_score": 0,
            "breakdown": {
                "keyword_match": 0,
                "skills_match": 0,
                "semantic_similarity": 0,
                "format_structure": 0,
                "common_keywords": [],
                "common_skills": [],
                "missing_keywords": [],
                "missing_skills": []
            },
            "recommendations": ["Resume text is too short for meaningful analysis"]
        }

    if len(jd_text.strip()) < 50:
        return {
            "overall_score": 0,
            "breakdown": {
                "keyword_match": 0,
                "skills_match": 0,
                "semantic_similarity": 0,
                "format_structure": 0,
                "common_keywords": [],
                "common_skills": [],
                "missing_keywords": [],
                "missing_skills": []
            },
            "recommendations": ["Job description is too short for meaningful analysis"]
        }

    try:
        # 1. Keyword matching score (40%)
        resume_keywords = set(extract_keywords(resume_text))
        jd_keywords = set(extract_keywords(jd_text))

        common_keywords = resume_keywords.intersection(jd_keywords)
        keyword_score = (len(common_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0

        # 2. Skills matching score (30%)
        resume_skills = set(extract_skills(resume_text))
        jd_skills = set(extract_skills(jd_text))

        common_skills = resume_skills.intersection(jd_skills)
        skills_score = (len(common_skills) / len(jd_skills)) * 100 if jd_skills else 0

        # 3. Semantic similarity score (20%)
        semantic_score = 0
        if sentence_model:
            try:
                resume_embedding = sentence_model.encode([resume_text])
                jd_embedding = sentence_model.encode([jd_text])
                semantic_score = cosine_similarity(resume_embedding, jd_embedding)[0][0] * 100
            except Exception as e:
                st.warning(f"⚠️ Semantic analysis unavailable: {e}")
                semantic_score = 0

        # 4. Format and structure score (10%)
        format_score = calculate_format_score(resume_text)

        # Calculate overall score
        overall_score = (
            keyword_score * 0.4 +
            skills_score * 0.3 +
            semantic_score * 0.2 +
            format_score * 0.1
        )

        breakdown = {
            "keyword_match": round(keyword_score, 2),
            "skills_match": round(skills_score, 2),
            "semantic_similarity": round(semantic_score, 2),
            "format_structure": round(format_score, 2),
            "common_keywords": list(common_keywords)[:50],  # Limit to prevent memory issues
            "common_skills": list(common_skills)[:50],
            "missing_keywords": list(jd_keywords - resume_keywords)[:50],
            "missing_skills": list(jd_skills - resume_skills)[:50]
        }

        recommendations = generate_ats_recommendations(breakdown)

        return {
            "overall_score": round(min(overall_score, 100), 2),  # Cap at 100%
            "breakdown": breakdown,
            "recommendations": recommendations
        }

    except Exception as e:
        st.error(f"❌ Error calculating ATS score: {str(e)}")
        return {
            "overall_score": 0,
            "breakdown": {
                "keyword_match": 0,
                "skills_match": 0,
                "semantic_similarity": 0,
                "format_structure": 0,
                "common_keywords": [],
                "common_skills": [],
                "missing_keywords": [],
                "missing_skills": []
            },
            "recommendations": ["Error occurred during analysis. Please try again."]
        }

def calculate_format_score(text: str) -> float:
    """Calculate formatting score based on resume structure"""
    score = 0

    # Check for common resume sections
    sections = ["experience", "education", "skills", "summary", "objective"]
    for section in sections:
        if section in text.lower():
            score += 20

    # Check for bullet points
    if "•" in text or "*" in text or "-" in text:
        score += 10

    # Check for dates (employment history)
    date_pattern = r'\b\d{4}\b|\b\d{1,2}/\d{4}\b|\b\d{1,2}-\d{4}\b'
    if re.search(date_pattern, text):
        score += 10

    # Check for contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

    if re.search(email_pattern, text):
        score += 5
    if re.search(phone_pattern, text):
        score += 5

    return min(score, 100)

def generate_ats_recommendations(breakdown: Dict) -> List[str]:
    """Generate specific recommendations based on ATS analysis"""
    recommendations = []

    if breakdown["keyword_match"] < 50:
        recommendations.append("Include more keywords from the job description in your resume")

    if breakdown["skills_match"] < 60:
        recommendations.append("Highlight technical skills that match the job requirements")

    if breakdown["semantic_similarity"] < 40:
        recommendations.append("Align your experience descriptions more closely with the job description")

    if breakdown["format_structure"] < 80:
        recommendations.append("Improve resume formatting with clear sections and bullet points")

    if len(breakdown["missing_keywords"]) > 10:
        recommendations.append("Consider adding these missing keywords: " +
                             ", ".join(breakdown["missing_keywords"][:5]))

    return recommendations

# --- AI Integration Functions ---
def call_openrouter_api(prompt: str, api_key: str, model: str = "anthropic/claude-3-haiku") -> str:
    """Call OpenRouter API for AI-powered text generation with comprehensive error handling"""
    if not api_key or not api_key.strip():
        st.error("❌ API key is required for AI features. Please enter your OpenRouter API key in the sidebar.")
        return "Error: API key required for AI features"

    # Validate prompt
    if not prompt or len(prompt.strip()) < 10:
        st.error("❌ Invalid prompt provided.")
        return "Error: Invalid prompt"

    # Rate limiting check (basic implementation)
    current_time = time.time()
    if 'last_api_call' not in st.session_state:
        st.session_state.last_api_call = 0

    if current_time - st.session_state.last_api_call < 2:  # 2 second rate limit
        st.warning("⚠️ Please wait a moment between API calls.")
        time.sleep(2)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://resume-optimizer.streamlit.app",
        "X-Title": "AI-Powered Career Platform"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert resume writer and career coach. Provide helpful, accurate, and professional advice."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        st.session_state.last_api_call = current_time

        with st.spinner("🤖 AI is processing your request..."):
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)

        # Handle different HTTP status codes
        if response.status_code == 401:
            st.error("❌ Invalid API key. Please check your OpenRouter API key.")
            return "Error: Invalid API key"
        elif response.status_code == 429:
            st.error("❌ Rate limit exceeded. Please wait a moment and try again.")
            return "Error: Rate limit exceeded"
        elif response.status_code == 500:
            st.error("❌ Server error. Please try again later.")
            return "Error: Server error"

        response.raise_for_status()

        result = response.json()

        # Validate response structure
        if "choices" not in result or not result["choices"]:
            st.error("❌ Invalid API response structure.")
            return "Error: Invalid API response"

        content = result["choices"][0]["message"]["content"]

        if not content or len(content.strip()) < 10:
            st.warning("⚠️ AI generated very short response. Please try again.")
            return "Error: Insufficient response from AI"

        return content

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return "Error: Request timed out"
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet connection.")
        return "Error: Connection error"
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {str(e)}")
        return f"Error: API request failed - {str(e)}"
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON response from API.")
        return "Error: Invalid JSON response"
    except KeyError as e:
        st.error(f"❌ Missing expected field in API response: {e}")
        return "Error: Invalid API response format"
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return f"Error: {str(e)}"

def rewrite_resume(resume_text: str, jd_text: str, api_key: str) -> str:
    """AI-powered resume rewriting optimized for ATS"""
    prompt = f"""
    You are an expert resume writer and ATS optimization specialist.

    Please rewrite the following resume to better match the job description while maintaining truthfulness and the candidate's actual experience.

    Focus on:
    1. Incorporating relevant keywords from the job description
    2. Improving ATS compatibility
    3. Enhancing readability for recruiters
    4. Quantifying achievements where possible
    5. Using action verbs and industry-specific terminology

    Job Description:
    {jd_text}

    Current Resume:
    {resume_text}

    Please provide an optimized version that maintains the candidate's authentic experience while improving job match and ATS score.
    """

    return call_openrouter_api(prompt, api_key)

def generate_cover_letter(resume_text: str, jd_text: str, api_key: str, company_name: str = "") -> str:
    """Generate AI-powered cover letter"""
    prompt = f"""
    Create a compelling cover letter based on the resume and job description provided.

    Requirements:
    1. Professional tone and structure
    2. Highlight relevant experience from the resume
    3. Address key requirements from the job description
    4. Show enthusiasm for the role and company
    5. Keep it concise (3-4 paragraphs)
    6. Include a strong opening and closing

    Company Name: {company_name if company_name else "[Company Name]"}

    Job Description:
    {jd_text}

    Resume:
    {resume_text}

    Please write a personalized cover letter that effectively connects the candidate's background to the job requirements.
    """

    return call_openrouter_api(prompt, api_key)

def generate_interview_questions(jd_text: str, api_key: str) -> List[str]:
    """Generate potential interview questions based on job description"""
    prompt = f"""
    Based on the following job description, generate 10 potential interview questions that a candidate should prepare for.

    Include a mix of:
    1. Technical questions related to required skills
    2. Behavioral questions using the STAR method
    3. Company/role-specific questions
    4. Situational questions

    Job Description:
    {jd_text}

    Please provide exactly 10 questions, one per line, without numbering.
    """

    response = call_openrouter_api(prompt, api_key)

    # Parse response into list
    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().isdigit()]
    return questions[:10]  # Ensure we return exactly 10 questions

def analyze_resume_sections(resume_text: str, jd_text: str, api_key: str) -> Dict[str, str]:
    """Analyze and provide feedback for each resume section"""
    prompt = f"""
    Analyze the following resume against the job description and provide specific feedback for each section.

    Please provide feedback in the following format:
    SUMMARY: [feedback for summary/objective section]
    SKILLS: [feedback for skills section]
    EXPERIENCE: [feedback for work experience section]
    EDUCATION: [feedback for education section]
    OVERALL: [overall recommendations]

    Job Description:
    {jd_text}

    Resume:
    {resume_text}

    Focus on how well each section aligns with the job requirements and provide specific improvement suggestions.
    """

    response = call_openrouter_api(prompt, api_key)

    # Parse response into sections
    sections = {}
    current_section = None
    current_content = []

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith(('SUMMARY:', 'SKILLS:', 'EXPERIENCE:', 'EDUCATION:', 'OVERALL:')):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.split(':')[0].lower()
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif current_section and line:
            current_content.append(line)

    # Add the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections

# --- Advanced AI Analysis Functions ---
def check_grammar_and_spelling(text: str, api_key: str) -> Dict[str, Any]:
    """AI-powered grammar and spell checking"""
    prompt = f"""
    Please analyze the following text for grammar errors, spelling mistakes, and writing improvements.

    Provide your response in this format:
    ERRORS_FOUND: [number of errors found]
    CORRECTIONS: [list each error and correction]
    SUGGESTIONS: [general writing improvement suggestions]
    CORRECTED_TEXT: [the fully corrected version]

    Text to analyze:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    # Parse response
    result = {
        "errors_found": 0,
        "corrections": [],
        "suggestions": [],
        "corrected_text": text
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('ERRORS_FOUND:'):
                try:
                    result["errors_found"] = int(re.findall(r'\d+', line)[0])
                except:
                    result["errors_found"] = 0
            elif line.startswith('CORRECTIONS:'):
                current_section = "corrections"
            elif line.startswith('SUGGESTIONS:'):
                current_section = "suggestions"
            elif line.startswith('CORRECTED_TEXT:'):
                current_section = "corrected_text"
                result["corrected_text"] = ""
            elif current_section == "corrections" and line:
                result["corrections"].append(line)
            elif current_section == "suggestions" and line:
                result["suggestions"].append(line)
            elif current_section == "corrected_text" and line:
                result["corrected_text"] += line + "\n"
    except Exception:
        pass

    return result

def analyze_action_verbs(text: str, api_key: str) -> Dict[str, Any]:
    """Analyze action verbs strength and suggest improvements"""
    prompt = f"""
    Analyze the action verbs in this resume text. Identify weak verbs and suggest stronger alternatives.

    Provide response in this format:
    WEAK_VERBS: [list weak verbs found]
    STRONG_ALTERNATIVES: [suggest stronger verbs for each weak one]
    MISSING_STRONG_VERBS: [suggest additional strong verbs for this industry]
    IMPROVED_SENTENCES: [rewrite 3-5 key sentences with stronger verbs]

    Resume text:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "weak_verbs": [],
        "strong_alternatives": [],
        "missing_strong_verbs": [],
        "improved_sentences": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('WEAK_VERBS:'):
                current_section = "weak_verbs"
            elif line.startswith('STRONG_ALTERNATIVES:'):
                current_section = "strong_alternatives"
            elif line.startswith('MISSING_STRONG_VERBS:'):
                current_section = "missing_strong_verbs"
            elif line.startswith('IMPROVED_SENTENCES:'):
                current_section = "improved_sentences"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def detect_quantification_opportunities(text: str, api_key: str) -> Dict[str, Any]:
    """Detect missing quantification and suggest metrics"""
    prompt = f"""
    Analyze this resume for missing quantification opportunities. Identify statements that could be strengthened with numbers, percentages, or metrics.

    Provide response in this format:
    MISSING_METRICS: [list statements that need quantification]
    SUGGESTED_METRICS: [suggest specific types of metrics for each]
    QUANTIFIED_EXAMPLES: [rewrite statements with example metrics]
    METRIC_TYPES: [list relevant metric types for this role]

    Resume text:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "missing_metrics": [],
        "suggested_metrics": [],
        "quantified_examples": [],
        "metric_types": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('MISSING_METRICS:'):
                current_section = "missing_metrics"
            elif line.startswith('SUGGESTED_METRICS:'):
                current_section = "suggested_metrics"
            elif line.startswith('QUANTIFIED_EXAMPLES:'):
                current_section = "quantified_examples"
            elif line.startswith('METRIC_TYPES:'):
                current_section = "metric_types"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def analyze_readability(text: str, api_key: str) -> Dict[str, Any]:
    """Analyze text readability and complexity"""
    prompt = f"""
    Analyze the readability and complexity of this resume text. Provide a readability assessment.

    Provide response in this format:
    READABILITY_SCORE: [score from 1-10, where 10 is most readable]
    COMPLEXITY_LEVEL: [Simple/Moderate/Complex]
    ISSUES: [list readability issues found]
    IMPROVEMENTS: [suggest specific improvements]
    SIMPLIFIED_VERSION: [rewrite 2-3 complex sentences in simpler form]

    Resume text:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "readability_score": 7,
        "complexity_level": "Moderate",
        "issues": [],
        "improvements": [],
        "simplified_version": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('READABILITY_SCORE:'):
                try:
                    result["readability_score"] = int(re.findall(r'\d+', line)[0])
                except:
                    result["readability_score"] = 7
            elif line.startswith('COMPLEXITY_LEVEL:'):
                result["complexity_level"] = line.split(':', 1)[1].strip()
            elif line.startswith('ISSUES:'):
                current_section = "issues"
            elif line.startswith('IMPROVEMENTS:'):
                current_section = "improvements"
            elif line.startswith('SIMPLIFIED_VERSION:'):
                current_section = "simplified_version"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def categorize_skills(text: str, api_key: str) -> Dict[str, Any]:
    """Categorize skills into hard skills vs soft skills"""
    prompt = f"""
    Analyze the skills mentioned in this resume and categorize them.

    Provide response in this format:
    HARD_SKILLS: [list technical/hard skills found]
    SOFT_SKILLS: [list soft/interpersonal skills found]
    MISSING_HARD_SKILLS: [suggest relevant hard skills for this role]
    MISSING_SOFT_SKILLS: [suggest relevant soft skills for this role]
    SKILL_STRENGTH: [rate overall skill portfolio from 1-10]

    Resume text:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "hard_skills": [],
        "soft_skills": [],
        "missing_hard_skills": [],
        "missing_soft_skills": [],
        "skill_strength": 7
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('HARD_SKILLS:'):
                current_section = "hard_skills"
            elif line.startswith('SOFT_SKILLS:'):
                current_section = "soft_skills"
            elif line.startswith('MISSING_HARD_SKILLS:'):
                current_section = "missing_hard_skills"
            elif line.startswith('MISSING_SOFT_SKILLS:'):
                current_section = "missing_soft_skills"
            elif line.startswith('SKILL_STRENGTH:'):
                try:
                    result["skill_strength"] = int(re.findall(r'\d+', line)[0])
                except:
                    result["skill_strength"] = 7
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def generate_resume_bullets(basic_info: str, api_key: str) -> List[str]:
    """Generate professional resume bullet points from basic information"""
    prompt = f"""
    Transform this basic work information into professional, ATS-optimized resume bullet points.
    Use strong action verbs, include quantifiable achievements where possible, and make them compelling.

    Generate 5-8 professional bullet points based on this information:
    {basic_info}

    Format each bullet point on a new line starting with "•"
    Focus on achievements, impact, and results rather than just duties.
    """

    response = call_openrouter_api(prompt, api_key)

    # Extract bullet points
    bullets = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            bullets.append(line.lstrip('•-* '))

    return bullets

def quantify_achievements(text: str, api_key: str) -> Dict[str, Any]:
    """Suggest specific metrics and quantifications for achievements"""
    prompt = f"""
    Analyze these achievements and suggest specific, realistic metrics to quantify them.

    For each achievement, provide:
    1. The original statement
    2. Suggested metrics (percentages, numbers, timeframes)
    3. Quantified version

    Format response as:
    ORIGINAL: [original statement]
    METRICS: [suggested specific metrics]
    QUANTIFIED: [rewritten with metrics]
    ---

    Achievements to analyze:
    {text}
    """

    response = call_openrouter_api(prompt, api_key)

    achievements = []
    current_achievement = {}

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('ORIGINAL:'):
            if current_achievement:
                achievements.append(current_achievement)
            current_achievement = {"original": line.split(':', 1)[1].strip()}
        elif line.startswith('METRICS:'):
            current_achievement["metrics"] = line.split(':', 1)[1].strip()
        elif line.startswith('QUANTIFIED:'):
            current_achievement["quantified"] = line.split(':', 1)[1].strip()
        elif line == '---' and current_achievement:
            achievements.append(current_achievement)
            current_achievement = {}

    if current_achievement:
        achievements.append(current_achievement)

    return {"achievements": achievements}

def suggest_personal_branding(resume_text: str, jd_text: str, api_key: str) -> Dict[str, Any]:
    """Generate personal branding suggestions"""
    prompt = f"""
    Based on this resume and target job, suggest personal branding strategies.

    Provide response in this format:
    BRAND_STATEMENT: [2-3 sentence personal brand statement]
    KEY_STRENGTHS: [3-5 key strengths to highlight]
    UNIQUE_VALUE: [what makes this candidate unique]
    LINKEDIN_HEADLINE: [optimized LinkedIn headline]
    ELEVATOR_PITCH: [30-second elevator pitch]
    BRAND_KEYWORDS: [keywords to use consistently across platforms]

    Resume:
    {resume_text}

    Target Job:
    {jd_text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "brand_statement": "",
        "key_strengths": [],
        "unique_value": "",
        "linkedin_headline": "",
        "elevator_pitch": "",
        "brand_keywords": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('BRAND_STATEMENT:'):
                result["brand_statement"] = line.split(':', 1)[1].strip()
            elif line.startswith('KEY_STRENGTHS:'):
                current_section = "key_strengths"
            elif line.startswith('UNIQUE_VALUE:'):
                result["unique_value"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('LINKEDIN_HEADLINE:'):
                result["linkedin_headline"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('ELEVATOR_PITCH:'):
                result["elevator_pitch"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('BRAND_KEYWORDS:'):
                current_section = "brand_keywords"
            elif current_section == "key_strengths" and line:
                result["key_strengths"].append(line)
            elif current_section == "brand_keywords" and line:
                result["brand_keywords"].append(line)
    except Exception:
        pass

    return result

def optimize_linkedin_profile(resume_text: str, api_key: str) -> Dict[str, Any]:
    """Generate LinkedIn profile optimization suggestions"""
    prompt = f"""
    Based on this resume, provide LinkedIn profile optimization suggestions.

    Provide response in this format:
    HEADLINE: [optimized LinkedIn headline]
    SUMMARY: [compelling LinkedIn summary/about section]
    SKILLS_TO_ADD: [skills to add to LinkedIn skills section]
    EXPERIENCE_TIPS: [how to optimize experience descriptions]
    CONTENT_IDEAS: [5 content ideas for posts/articles]
    NETWORKING_STRATEGY: [networking approach for this profile]

    Resume:
    {resume_text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "headline": "",
        "summary": "",
        "skills_to_add": [],
        "experience_tips": [],
        "content_ideas": [],
        "networking_strategy": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('HEADLINE:'):
                result["headline"] = line.split(':', 1)[1].strip()
            elif line.startswith('SUMMARY:'):
                result["summary"] = line.split(':', 1)[1].strip()
            elif line.startswith('SKILLS_TO_ADD:'):
                current_section = "skills_to_add"
            elif line.startswith('EXPERIENCE_TIPS:'):
                current_section = "experience_tips"
            elif line.startswith('CONTENT_IDEAS:'):
                current_section = "content_ideas"
            elif line.startswith('NETWORKING_STRATEGY:'):
                current_section = "networking_strategy"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def suggest_career_progression(resume_text: str, api_key: str) -> Dict[str, Any]:
    """Suggest career progression and next steps"""
    prompt = f"""
    Based on this resume, suggest career progression opportunities and next steps.

    Provide response in this format:
    CURRENT_LEVEL: [current career level assessment]
    NEXT_ROLES: [3-5 logical next role titles]
    SKILL_GAPS: [skills needed for advancement]
    TIMELINE: [realistic timeline for progression]
    ACTION_STEPS: [specific steps to take]
    CERTIFICATIONS: [relevant certifications to pursue]

    Resume:
    {resume_text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "current_level": "",
        "next_roles": [],
        "skill_gaps": [],
        "timeline": "",
        "action_steps": [],
        "certifications": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('CURRENT_LEVEL:'):
                result["current_level"] = line.split(':', 1)[1].strip()
            elif line.startswith('TIMELINE:'):
                result["timeline"] = line.split(':', 1)[1].strip()
            elif line.startswith('NEXT_ROLES:'):
                current_section = "next_roles"
            elif line.startswith('SKILL_GAPS:'):
                current_section = "skill_gaps"
            elif line.startswith('ACTION_STEPS:'):
                current_section = "action_steps"
            elif line.startswith('CERTIFICATIONS:'):
                current_section = "certifications"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

# --- Interview Preparation Functions ---
def generate_company_specific_questions(company_name: str, jd_text: str, api_key: str) -> List[str]:
    """Generate company-specific interview questions"""
    prompt = f"""
    Generate 10 company-specific interview questions for {company_name} based on this job description.
    Include questions about company culture, values, recent news, and role-specific challenges.

    Job Description:
    {jd_text}

    Provide exactly 10 questions, one per line.
    """

    response = call_openrouter_api(prompt, api_key)

    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and not line.isdigit() and len(line) > 10:
            # Remove numbering if present
            line = re.sub(r'^\d+\.?\s*', '', line)
            questions.append(line)

    return questions[:10]

def generate_behavioral_questions(jd_text: str, api_key: str) -> Dict[str, List[str]]:
    """Generate behavioral interview questions with STAR method guidance"""
    prompt = f"""
    Generate behavioral interview questions based on this job description.
    Organize them by category and provide STAR method guidance.

    Provide response in this format:
    LEADERSHIP: [3 leadership questions]
    TEAMWORK: [3 teamwork questions]
    PROBLEM_SOLVING: [3 problem-solving questions]
    COMMUNICATION: [3 communication questions]
    ADAPTABILITY: [3 adaptability questions]

    Job Description:
    {jd_text}
    """

    response = call_openrouter_api(prompt, api_key)

    categories = {
        "leadership": [],
        "teamwork": [],
        "problem_solving": [],
        "communication": [],
        "adaptability": []
    }

    current_category = None
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('LEADERSHIP:'):
            current_category = "leadership"
        elif line.startswith('TEAMWORK:'):
            current_category = "teamwork"
        elif line.startswith('PROBLEM_SOLVING:') or line.startswith('PROBLEM-SOLVING:'):
            current_category = "problem_solving"
        elif line.startswith('COMMUNICATION:'):
            current_category = "communication"
        elif line.startswith('ADAPTABILITY:'):
            current_category = "adaptability"
        elif current_category and line and not line.isdigit():
            line = re.sub(r'^\d+\.?\s*', '', line)
            if len(line) > 10:
                categories[current_category].append(line)

    return categories

def generate_technical_questions(jd_text: str, api_key: str) -> Dict[str, List[str]]:
    """Generate technical interview questions and coding challenges"""
    prompt = f"""
    Generate technical interview questions based on this job description.
    Include different difficulty levels and types.

    Provide response in this format:
    BASIC_CONCEPTS: [5 basic technical questions]
    INTERMEDIATE: [5 intermediate questions]
    ADVANCED: [3 advanced questions]
    CODING_CHALLENGES: [3 coding challenge descriptions]
    SYSTEM_DESIGN: [2 system design questions]

    Job Description:
    {jd_text}
    """

    response = call_openrouter_api(prompt, api_key)

    categories = {
        "basic_concepts": [],
        "intermediate": [],
        "advanced": [],
        "coding_challenges": [],
        "system_design": []
    }

    current_category = None
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('BASIC_CONCEPTS:'):
            current_category = "basic_concepts"
        elif line.startswith('INTERMEDIATE:'):
            current_category = "intermediate"
        elif line.startswith('ADVANCED:'):
            current_category = "advanced"
        elif line.startswith('CODING_CHALLENGES:'):
            current_category = "coding_challenges"
        elif line.startswith('SYSTEM_DESIGN:'):
            current_category = "system_design"
        elif current_category and line and not line.isdigit():
            line = re.sub(r'^\d+\.?\s*', '', line)
            if len(line) > 10:
                categories[current_category].append(line)

    return categories

def generate_salary_negotiation_guide(jd_text: str, experience_level: str, api_key: str) -> Dict[str, Any]:
    """Generate salary negotiation guidance and scripts"""
    prompt = f"""
    Provide salary negotiation guidance for this role and experience level.

    Experience Level: {experience_level}
    Job Description: {jd_text}

    Provide response in this format:
    RESEARCH_TIPS: [how to research salary ranges]
    NEGOTIATION_TIMING: [when to negotiate]
    OPENING_SCRIPTS: [3 opening negotiation scripts]
    COUNTER_OFFERS: [how to handle counter offers]
    NON_SALARY_BENEFITS: [other benefits to negotiate]
    COMMON_MISTAKES: [mistakes to avoid]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "research_tips": [],
        "negotiation_timing": "",
        "opening_scripts": [],
        "counter_offers": [],
        "non_salary_benefits": [],
        "common_mistakes": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('RESEARCH_TIPS:'):
                current_section = "research_tips"
            elif line.startswith('NEGOTIATION_TIMING:'):
                result["negotiation_timing"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('OPENING_SCRIPTS:'):
                current_section = "opening_scripts"
            elif line.startswith('COUNTER_OFFERS:'):
                current_section = "counter_offers"
            elif line.startswith('NON_SALARY_BENEFITS:'):
                current_section = "non_salary_benefits"
            elif line.startswith('COMMON_MISTAKES:'):
                current_section = "common_mistakes"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

# --- Communication Templates Functions ---
def generate_email_templates(purpose: str, company_name: str, api_key: str) -> Dict[str, str]:
    """Generate professional email templates"""
    prompt = f"""
    Generate professional email templates for: {purpose}
    Company: {company_name}

    Provide 3 different templates in this format:
    TEMPLATE_1: [first template]
    ---
    TEMPLATE_2: [second template]
    ---
    TEMPLATE_3: [third template]

    Make them professional, concise, and personalized.
    """

    response = call_openrouter_api(prompt, api_key)

    templates = {}
    current_template = None
    current_content = []

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('TEMPLATE_'):
            if current_template:
                templates[current_template] = '\n'.join(current_content).strip()
            current_template = line.split(':')[0].lower()
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif line == '---':
            if current_template:
                templates[current_template] = '\n'.join(current_content).strip()
                current_template = None
                current_content = []
        elif current_template:
            current_content.append(line)

    if current_template:
        templates[current_template] = '\n'.join(current_content).strip()

    return templates

def generate_networking_messages(industry: str, purpose: str, api_key: str) -> List[str]:
    """Generate networking message templates"""
    prompt = f"""
    Generate 5 professional networking messages for {industry} industry.
    Purpose: {purpose}

    Make them:
    - Personalized and authentic
    - Brief but engaging
    - Professional tone
    - Clear call to action

    Provide 5 different messages, each on a separate line starting with "MESSAGE:"
    """

    response = call_openrouter_api(prompt, api_key)

    messages = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('MESSAGE:'):
            messages.append(line.split(':', 1)[1].strip())

    return messages

def generate_thank_you_notes(interview_type: str, interviewer_name: str, api_key: str) -> Dict[str, str]:
    """Generate thank you note templates"""
    prompt = f"""
    Generate thank you note templates for after a {interview_type} interview.
    Interviewer: {interviewer_name}

    Provide 3 different styles:
    FORMAL: [formal thank you note]
    CONVERSATIONAL: [conversational thank you note]
    BRIEF: [brief thank you note]

    Include specific details about the interview and reiterate interest.
    """

    response = call_openrouter_api(prompt, api_key)

    notes = {}
    current_style = None
    current_content = []

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith(('FORMAL:', 'CONVERSATIONAL:', 'BRIEF:')):
            if current_style:
                notes[current_style] = '\n'.join(current_content).strip()
            current_style = line.split(':')[0].lower()
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif current_style:
            current_content.append(line)

    if current_style:
        notes[current_style] = '\n'.join(current_content).strip()

    return notes

# --- Resume Templates & Formatting Functions ---
def generate_resume_template(industry: str, style: str, api_key: str) -> Dict[str, str]:
    """Generate HTML/CSS resume template based on industry and style"""
    prompt = f"""
    Create a professional resume template for {industry} industry in {style} style.

    Provide response in this format:
    HTML_TEMPLATE: [complete HTML structure with placeholders]
    CSS_STYLES: [complete CSS styling]
    LAYOUT_TIPS: [specific layout recommendations]
    COLOR_SCHEME: [recommended colors for this industry]
    FONT_RECOMMENDATIONS: [best fonts for this style]

    Make it ATS-friendly and modern. Include placeholders like {{NAME}}, {{EMAIL}}, {{EXPERIENCE}}, etc.
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "html_template": "",
        "css_styles": "",
        "layout_tips": [],
        "color_scheme": "",
        "font_recommendations": []
    }

    try:
        lines = response.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith('HTML_TEMPLATE:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "html_template"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            elif line.startswith('CSS_STYLES:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "css_styles"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            elif line.startswith('LAYOUT_TIPS:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "layout_tips"
                current_content = []
            elif line.startswith('COLOR_SCHEME:'):
                if current_section == "layout_tips":
                    result[current_section] = current_content
                result["color_scheme"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('FONT_RECOMMENDATIONS:'):
                current_section = "font_recommendations"
                current_content = []
            elif current_section == "layout_tips" and line:
                current_content.append(line)
            elif current_section == "font_recommendations" and line:
                result["font_recommendations"].append(line)
            elif current_section in ["html_template", "css_styles"] and line:
                current_content.append(line)

        # Handle last section
        if current_section in ["html_template", "css_styles"]:
            result[current_section] = '\n'.join(current_content).strip()
        elif current_section == "layout_tips":
            result[current_section] = current_content
    except Exception:
        pass

    return result

def analyze_resume_format(resume_text: str, api_key: str) -> Dict[str, Any]:
    """Analyze and optimize resume format structure"""
    prompt = f"""
    Analyze this resume's format and structure. Provide optimization recommendations.

    Provide response in this format:
    FORMAT_SCORE: [score from 1-10]
    STRUCTURE_ISSUES: [list formatting problems]
    LAYOUT_IMPROVEMENTS: [specific layout suggestions]
    SECTION_ORGANIZATION: [how to reorganize sections]
    ATS_COMPATIBILITY: [ATS formatting recommendations]
    VISUAL_ENHANCEMENTS: [visual improvement suggestions]

    Resume:
    {resume_text}
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "format_score": 7,
        "structure_issues": [],
        "layout_improvements": [],
        "section_organization": [],
        "ats_compatibility": [],
        "visual_enhancements": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('FORMAT_SCORE:'):
                try:
                    result["format_score"] = int(re.findall(r'\d+', line)[0])
                except:
                    result["format_score"] = 7
            elif line.startswith('STRUCTURE_ISSUES:'):
                current_section = "structure_issues"
            elif line.startswith('LAYOUT_IMPROVEMENTS:'):
                current_section = "layout_improvements"
            elif line.startswith('SECTION_ORGANIZATION:'):
                current_section = "section_organization"
            elif line.startswith('ATS_COMPATIBILITY:'):
                current_section = "ats_compatibility"
            elif line.startswith('VISUAL_ENHANCEMENTS:'):
                current_section = "visual_enhancements"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def suggest_style_recommendations(industry: str, role_level: str, api_key: str) -> Dict[str, Any]:
    """AI suggests colors, fonts, layouts for specific industry/role"""
    prompt = f"""
    Suggest style recommendations for a {role_level} level professional in {industry} industry.

    Provide response in this format:
    COLOR_PALETTE: [primary, secondary, accent colors with hex codes]
    FONT_COMBINATIONS: [header font, body font, accent font]
    LAYOUT_STYLE: [modern, classic, creative, minimal]
    SPACING_GUIDELINES: [margin, padding, line spacing recommendations]
    VISUAL_ELEMENTS: [icons, borders, graphics suggestions]
    INDUSTRY_STANDARDS: [what's expected in this industry]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "color_palette": [],
        "font_combinations": [],
        "layout_style": "",
        "spacing_guidelines": [],
        "visual_elements": [],
        "industry_standards": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('COLOR_PALETTE:'):
                current_section = "color_palette"
            elif line.startswith('FONT_COMBINATIONS:'):
                current_section = "font_combinations"
            elif line.startswith('LAYOUT_STYLE:'):
                result["layout_style"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('SPACING_GUIDELINES:'):
                current_section = "spacing_guidelines"
            elif line.startswith('VISUAL_ELEMENTS:'):
                current_section = "visual_elements"
            elif line.startswith('INDUSTRY_STANDARDS:'):
                current_section = "industry_standards"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

# --- Skills Assessment & Learning Functions ---
def generate_skills_quiz(skill_name: str, difficulty: str, api_key: str) -> Dict[str, Any]:
    """Generate custom skills quiz for any skill"""
    prompt = f"""
    Create a {difficulty} level quiz for {skill_name} skill with 10 questions.

    Provide response in this format:
    QUESTION_1: [question text]
    OPTIONS_1: [A) option1, B) option2, C) option3, D) option4]
    ANSWER_1: [correct answer letter]
    EXPLANATION_1: [why this is correct]

    Continue for all 10 questions. Make questions practical and relevant.
    """

    response = call_openrouter_api(prompt, api_key)

    questions = []
    current_question = {}

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('QUESTION_'):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line.split(':', 1)[1].strip()}
        elif line.startswith('OPTIONS_'):
            current_question["options"] = line.split(':', 1)[1].strip()
        elif line.startswith('ANSWER_'):
            current_question["answer"] = line.split(':', 1)[1].strip()
        elif line.startswith('EXPLANATION_'):
            current_question["explanation"] = line.split(':', 1)[1].strip()

    if current_question:
        questions.append(current_question)

    return {"questions": questions, "skill": skill_name, "difficulty": difficulty}

def create_learning_path(current_skills: str, target_role: str, api_key: str) -> Dict[str, Any]:
    """Design personalized learning path"""
    prompt = f"""
    Create a personalized learning path for someone with these skills: {current_skills}
    Target role: {target_role}

    Provide response in this format:
    SKILL_GAPS: [skills needed for target role]
    LEARNING_PHASES: [Phase 1, Phase 2, Phase 3 with timeframes]
    RESOURCES: [free courses, books, tutorials for each skill]
    MILESTONES: [measurable goals and checkpoints]
    TIMELINE: [realistic timeline for completion]
    PRACTICE_PROJECTS: [hands-on projects to build skills]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "skill_gaps": [],
        "learning_phases": [],
        "resources": [],
        "milestones": [],
        "timeline": "",
        "practice_projects": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('SKILL_GAPS:'):
                current_section = "skill_gaps"
            elif line.startswith('LEARNING_PHASES:'):
                current_section = "learning_phases"
            elif line.startswith('RESOURCES:'):
                current_section = "resources"
            elif line.startswith('MILESTONES:'):
                current_section = "milestones"
            elif line.startswith('TIMELINE:'):
                result["timeline"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('PRACTICE_PROJECTS:'):
                current_section = "practice_projects"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def recommend_free_courses(skill_name: str, learning_style: str, api_key: str) -> List[Dict[str, str]]:
    """Recommend free courses from Coursera, edX, YouTube"""
    prompt = f"""
    Recommend free courses for learning {skill_name} for someone who prefers {learning_style} learning.

    Provide 8 course recommendations in this format:
    COURSE_1: [course name] | [platform] | [duration] | [description]
    COURSE_2: [course name] | [platform] | [duration] | [description]

    Include courses from Coursera, edX, YouTube, Khan Academy, freeCodeCamp, etc.
    """

    response = call_openrouter_api(prompt, api_key)

    courses = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('COURSE_') and '|' in line:
            parts = line.split(':', 1)[1].strip().split('|')
            if len(parts) >= 4:
                courses.append({
                    "name": parts[0].strip(),
                    "platform": parts[1].strip(),
                    "duration": parts[2].strip(),
                    "description": parts[3].strip()
                })

    return courses

# --- Advanced Interview Tools ---
def conduct_mock_interview(job_description: str, interview_type: str, api_key: str) -> Dict[str, Any]:
    """AI conducts full mock interview session"""
    prompt = f"""
    Conduct a {interview_type} mock interview for this job. Ask 5 relevant questions one by one.

    Job Description: {job_description}

    Provide response in this format:
    QUESTION_1: [first interview question]
    FOLLOW_UP_1: [potential follow-up question]
    QUESTION_2: [second interview question]
    FOLLOW_UP_2: [potential follow-up question]

    Continue for 5 questions. Make them realistic and job-specific.
    """

    response = call_openrouter_api(prompt, api_key)

    questions = []
    current_q = {}

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('QUESTION_'):
            if current_q:
                questions.append(current_q)
            current_q = {"question": line.split(':', 1)[1].strip()}
        elif line.startswith('FOLLOW_UP_'):
            current_q["follow_up"] = line.split(':', 1)[1].strip()

    if current_q:
        questions.append(current_q)

    return {"questions": questions, "interview_type": interview_type}

def analyze_interview_response(question: str, answer: str, api_key: str) -> Dict[str, Any]:
    """Analyze and improve interview answers"""
    prompt = f"""
    Analyze this interview response and provide detailed feedback.

    Question: {question}
    Answer: {answer}

    Provide response in this format:
    SCORE: [score from 1-10]
    STRENGTHS: [what was good about the answer]
    WEAKNESSES: [areas for improvement]
    IMPROVED_VERSION: [better version of the answer]
    TIPS: [specific tips for this type of question]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "score": 7,
        "strengths": [],
        "weaknesses": [],
        "improved_version": "",
        "tips": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    result["score"] = int(re.findall(r'\d+', line)[0])
                except:
                    result["score"] = 7
            elif line.startswith('STRENGTHS:'):
                current_section = "strengths"
            elif line.startswith('WEAKNESSES:'):
                current_section = "weaknesses"
            elif line.startswith('IMPROVED_VERSION:'):
                result["improved_version"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('TIPS:'):
                current_section = "tips"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def provide_body_language_tips(interview_type: str, api_key: str) -> Dict[str, List[str]]:
    """AI provides posture and gesture advice"""
    prompt = f"""
    Provide body language and presentation tips for {interview_type} interviews.

    Provide response in this format:
    POSTURE_TIPS: [sitting, standing, walking tips]
    HAND_GESTURES: [appropriate hand movements]
    EYE_CONTACT: [eye contact guidelines]
    FACIAL_EXPRESSIONS: [expression recommendations]
    VOICE_TIPS: [tone, pace, volume guidance]
    COMMON_MISTAKES: [body language mistakes to avoid]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "posture_tips": [],
        "hand_gestures": [],
        "eye_contact": [],
        "facial_expressions": [],
        "voice_tips": [],
        "common_mistakes": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('POSTURE_TIPS:'):
                current_section = "posture_tips"
            elif line.startswith('HAND_GESTURES:'):
                current_section = "hand_gestures"
            elif line.startswith('EYE_CONTACT:'):
                current_section = "eye_contact"
            elif line.startswith('FACIAL_EXPRESSIONS:'):
                current_section = "facial_expressions"
            elif line.startswith('VOICE_TIPS:'):
                current_section = "voice_tips"
            elif line.startswith('COMMON_MISTAKES:'):
                current_section = "common_mistakes"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def suggest_interview_outfit(industry: str, role_level: str, interview_type: str, api_key: str) -> Dict[str, Any]:
    """AI recommends appropriate interview attire"""
    prompt = f"""
    Suggest appropriate interview outfit for {role_level} level {interview_type} interview in {industry} industry.

    Provide response in this format:
    FORMAL_OPTION: [formal outfit description]
    BUSINESS_CASUAL: [business casual option]
    ACCESSORIES: [appropriate accessories]
    COLORS: [recommended colors]
    AVOID: [what not to wear]
    GROOMING: [grooming and hygiene tips]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "formal_option": "",
        "business_casual": "",
        "accessories": [],
        "colors": [],
        "avoid": [],
        "grooming": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('FORMAL_OPTION:'):
                result["formal_option"] = line.split(':', 1)[1].strip()
            elif line.startswith('BUSINESS_CASUAL:'):
                result["business_casual"] = line.split(':', 1)[1].strip()
            elif line.startswith('ACCESSORIES:'):
                current_section = "accessories"
            elif line.startswith('COLORS:'):
                current_section = "colors"
            elif line.startswith('AVOID:'):
                current_section = "avoid"
            elif line.startswith('GROOMING:'):
                current_section = "grooming"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

# --- Portfolio & Project Showcase Functions ---
def generate_portfolio_content(project_info: str, target_audience: str, api_key: str) -> Dict[str, str]:
    """AI writes compelling project descriptions"""
    prompt = f"""
    Create compelling portfolio content for this project targeting {target_audience}.

    Project Info: {project_info}

    Provide response in this format:
    PROJECT_TITLE: [catchy project title]
    EXECUTIVE_SUMMARY: [brief compelling overview]
    DETAILED_DESCRIPTION: [comprehensive project description]
    TECHNOLOGIES_USED: [technical stack and tools]
    KEY_ACHIEVEMENTS: [measurable results and impact]
    CHALLENGES_SOLVED: [problems addressed]
    LESSONS_LEARNED: [insights gained]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "project_title": "",
        "executive_summary": "",
        "detailed_description": "",
        "technologies_used": "",
        "key_achievements": "",
        "challenges_solved": "",
        "lessons_learned": ""
    }

    try:
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('PROJECT_TITLE:'):
                result["project_title"] = line.split(':', 1)[1].strip()
            elif line.startswith('EXECUTIVE_SUMMARY:'):
                result["executive_summary"] = line.split(':', 1)[1].strip()
            elif line.startswith('DETAILED_DESCRIPTION:'):
                result["detailed_description"] = line.split(':', 1)[1].strip()
            elif line.startswith('TECHNOLOGIES_USED:'):
                result["technologies_used"] = line.split(':', 1)[1].strip()
            elif line.startswith('KEY_ACHIEVEMENTS:'):
                result["key_achievements"] = line.split(':', 1)[1].strip()
            elif line.startswith('CHALLENGES_SOLVED:'):
                result["challenges_solved"] = line.split(':', 1)[1].strip()
            elif line.startswith('LESSONS_LEARNED:'):
                result["lessons_learned"] = line.split(':', 1)[1].strip()
    except Exception:
        pass

    return result

def calculate_project_impact(project_description: str, api_key: str) -> Dict[str, Any]:
    """AI quantifies project results and impact"""
    prompt = f"""
    Analyze this project and suggest quantifiable impact metrics.

    Project: {project_description}

    Provide response in this format:
    IMPACT_METRICS: [specific measurable outcomes]
    BUSINESS_VALUE: [business impact and value created]
    USER_BENEFITS: [benefits to end users]
    TECHNICAL_IMPROVEMENTS: [technical enhancements achieved]
    SUGGESTED_NUMBERS: [realistic numbers to highlight impact]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "impact_metrics": [],
        "business_value": [],
        "user_benefits": [],
        "technical_improvements": [],
        "suggested_numbers": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('IMPACT_METRICS:'):
                current_section = "impact_metrics"
            elif line.startswith('BUSINESS_VALUE:'):
                current_section = "business_value"
            elif line.startswith('USER_BENEFITS:'):
                current_section = "user_benefits"
            elif line.startswith('TECHNICAL_IMPROVEMENTS:'):
                current_section = "technical_improvements"
            elif line.startswith('SUGGESTED_NUMBERS:'):
                current_section = "suggested_numbers"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def create_case_study(project_info: str, target_role: str, api_key: str) -> Dict[str, str]:
    """Transform projects into compelling case studies"""
    prompt = f"""
    Transform this project into a compelling case study for {target_role} applications.

    Project Info: {project_info}

    Provide response in this format:
    CASE_STUDY_TITLE: [compelling title]
    PROBLEM_STATEMENT: [what problem was solved]
    SOLUTION_APPROACH: [how you approached the solution]
    IMPLEMENTATION: [what you built/did]
    RESULTS: [quantified outcomes]
    REFLECTION: [what you learned and would do differently]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "case_study_title": "",
        "problem_statement": "",
        "solution_approach": "",
        "implementation": "",
        "results": "",
        "reflection": ""
    }

    try:
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('CASE_STUDY_TITLE:'):
                result["case_study_title"] = line.split(':', 1)[1].strip()
            elif line.startswith('PROBLEM_STATEMENT:'):
                result["problem_statement"] = line.split(':', 1)[1].strip()
            elif line.startswith('SOLUTION_APPROACH:'):
                result["solution_approach"] = line.split(':', 1)[1].strip()
            elif line.startswith('IMPLEMENTATION:'):
                result["implementation"] = line.split(':', 1)[1].strip()
            elif line.startswith('RESULTS:'):
                result["results"] = line.split(':', 1)[1].strip()
            elif line.startswith('REFLECTION:'):
                result["reflection"] = line.split(':', 1)[1].strip()
    except Exception:
        pass

    return result

# --- AI Career Coach Functions ---
def career_chatbot_response(question: str, user_context: str, api_key: str) -> str:
    """24/7 AI career chatbot"""
    prompt = f"""
    You are an expert career coach. Answer this career question with personalized advice.

    User Context: {user_context}
    Question: {question}

    Provide practical, actionable advice in a friendly, professional tone.
    Keep response concise but comprehensive.
    """

    return call_openrouter_api(prompt, api_key)

def provide_career_advice(situation: str, goals: str, api_key: str) -> Dict[str, Any]:
    """Personalized career guidance"""
    prompt = f"""
    Provide personalized career advice for this situation.

    Current Situation: {situation}
    Career Goals: {goals}

    Provide response in this format:
    IMMEDIATE_ACTIONS: [3-5 immediate steps to take]
    SHORT_TERM_GOALS: [goals for next 3-6 months]
    LONG_TERM_STRATEGY: [1-2 year career strategy]
    POTENTIAL_OBSTACLES: [challenges to anticipate]
    SUCCESS_METRICS: [how to measure progress]
    RESOURCES: [helpful resources and tools]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "immediate_actions": [],
        "short_term_goals": [],
        "long_term_strategy": "",
        "potential_obstacles": [],
        "success_metrics": [],
        "resources": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('IMMEDIATE_ACTIONS:'):
                current_section = "immediate_actions"
            elif line.startswith('SHORT_TERM_GOALS:'):
                current_section = "short_term_goals"
            elif line.startswith('LONG_TERM_STRATEGY:'):
                result["long_term_strategy"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('POTENTIAL_OBSTACLES:'):
                current_section = "potential_obstacles"
            elif line.startswith('SUCCESS_METRICS:'):
                current_section = "success_metrics"
            elif line.startswith('RESOURCES:'):
                current_section = "resources"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

def research_company_insights(company_name: str, api_key: str) -> Dict[str, Any]:
    """Research company from public information"""
    prompt = f"""
    Provide comprehensive research insights about {company_name}.

    Provide response in this format:
    COMPANY_OVERVIEW: [brief company description]
    BUSINESS_MODEL: [how they make money]
    RECENT_NEWS: [recent developments and news]
    COMPANY_CULTURE: [culture and values insights]
    GROWTH_TRAJECTORY: [growth and expansion plans]
    CHALLENGES: [current challenges they face]
    INTERVIEW_TIPS: [specific tips for interviewing here]
    """

    response = call_openrouter_api(prompt, api_key)

    result = {
        "company_overview": "",
        "business_model": "",
        "recent_news": [],
        "company_culture": "",
        "growth_trajectory": "",
        "challenges": [],
        "interview_tips": []
    }

    try:
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('COMPANY_OVERVIEW:'):
                result["company_overview"] = line.split(':', 1)[1].strip()
            elif line.startswith('BUSINESS_MODEL:'):
                result["business_model"] = line.split(':', 1)[1].strip()
            elif line.startswith('RECENT_NEWS:'):
                current_section = "recent_news"
            elif line.startswith('COMPANY_CULTURE:'):
                result["company_culture"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('GROWTH_TRAJECTORY:'):
                result["growth_trajectory"] = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('CHALLENGES:'):
                current_section = "challenges"
            elif line.startswith('INTERVIEW_TIPS:'):
                current_section = "interview_tips"
            elif current_section and line:
                result[current_section].append(line)
    except Exception:
        pass

    return result

# --- Job Matching Functions ---
def calculate_job_match_percentage(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Calculate job match percentage using multiple factors"""
    if not resume_text or not jd_text:
        return {"match_percentage": 0, "factors": {}}

    # Factor 1: Keyword overlap
    resume_keywords = set(extract_keywords(resume_text))
    jd_keywords = set(extract_keywords(jd_text))
    keyword_overlap = len(resume_keywords.intersection(jd_keywords)) / len(jd_keywords) if jd_keywords else 0

    # Factor 2: Skills overlap
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    skills_overlap = len(resume_skills.intersection(jd_skills)) / len(jd_skills) if jd_skills else 0

    # Factor 3: Semantic similarity
    semantic_similarity = 0
    if sentence_model:
        try:
            resume_embedding = sentence_model.encode([resume_text])
            jd_embedding = sentence_model.encode([jd_text])
            semantic_similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        except Exception:
            pass

    # Factor 4: Experience level match (basic heuristic)
    experience_match = calculate_experience_match(resume_text, jd_text)

    # Weighted average
    match_percentage = (
        keyword_overlap * 0.3 +
        skills_overlap * 0.3 +
        semantic_similarity * 0.25 +
        experience_match * 0.15
    ) * 100

    factors = {
        "keyword_overlap": round(keyword_overlap * 100, 2),
        "skills_overlap": round(skills_overlap * 100, 2),
        "semantic_similarity": round(semantic_similarity * 100, 2),
        "experience_match": round(experience_match * 100, 2)
    }

    return {
        "match_percentage": round(match_percentage, 2),
        "factors": factors
    }

def calculate_experience_match(resume_text: str, jd_text: str) -> float:
    """Calculate experience level match between resume and job description"""
    # Extract years of experience mentioned
    resume_years = extract_years_experience(resume_text)
    jd_years = extract_years_experience(jd_text)

    if not jd_years:
        return 0.8  # Default score if no experience requirement specified

    if not resume_years:
        return 0.3  # Low score if no experience found in resume

    # Calculate match based on experience gap
    max_resume_years = max(resume_years)
    min_jd_years = min(jd_years)

    if max_resume_years >= min_jd_years:
        return 1.0
    else:
        # Partial credit based on how close they are
        return max(0.2, max_resume_years / min_jd_years)

def extract_years_experience(text: str) -> List[int]:
    """Extract years of experience from text"""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*years?\s*in',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*yrs?\s*in'
    ]

    years = []
    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(match) for match in matches])

    return years

# --- UI Helper Functions ---
def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .home-title {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .score-excellent { color: #00ff00; font-weight: bold; }
    .score-good { color: #90EE90; font-weight: bold; }
    .score-average { color: #FFD700; font-weight: bold; }
    .score-poor { color: #FF6B6B; font-weight: bold; }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: rgba(102, 126, 234, 0.1);
    }

    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_score_color_class(score: float) -> str:
    """Get CSS class based on score"""
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-average"
    else:
        return "score-poor"

def display_score_with_progress(title: str, score: float, max_score: float = 100):
    """Display score with progress bar and color coding"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**{title}**")
        st.progress(score / max_score)

    with col2:
        score_class = get_score_color_class(score)
        st.markdown(f'<div class="{score_class}">{score:.1f}%</div>', unsafe_allow_html=True)

def create_sidebar_navigation():
    """Create sidebar navigation"""
    with st.sidebar:
        st.markdown('<div class="main-header">🚀 AI Career Hub</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Navigation
        pages = {
            "🏠 Home": "Home",
            "📤 Upload": "Upload",
            "📊 Analysis": "Analysis",
            "✨ Resume Tools": "ResumeTools",
            "❓ Interview Mastery": "InterviewMastery",
            "📚 Skills & Learning": "SkillsLearning",
            "📁 Portfolio & Projects": "Portfolio",
            "🤖 AI Career Coach": "CareerCoach",
            "📋 Comparison": "Comparison",
            "📈 Analytics": "Analytics",
            "📄 Reports": "Reports",
            "⚙️ Settings": "Settings"
        }

        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key

        st.markdown("---")

        # API Key input
        st.markdown("**🔑 API Configuration**")
        api_key = st.text_input("OpenRouter API Key", type="password", key="api_key")

        # API key is automatically stored in session state with key="api_key"
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required for AI features")

        st.markdown("---")

        # Theme toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=True, key="sidebar_dark_mode")

        # Quick stats
        if st.session_state.get('resume_text') and st.session_state.get('jd_text'):
            st.markdown("**📊 Quick Stats**")
            ats_result = calculate_ats_score(st.session_state.resume_text, st.session_state.jd_text)
            st.metric("ATS Score", f"{ats_result['overall_score']:.1f}%")

            match_result = calculate_job_match_percentage(st.session_state.resume_text, st.session_state.jd_text)
            st.metric("Job Match", f"{match_result['match_percentage']:.1f}%")

        return api_key

# --- Page Functions ---
def show_home_page():
    """Display home page"""
    st.markdown('<div class="home-title">AI-Powered Career Platform</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Complete Career Platform</h3>
        <p>75+ AI-powered features: ATS optimization, resume templates, mock interviews, skills assessment, portfolio tools & career coaching.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 Advanced AI Tools</h3>
        <p>AI resume rewriting, interview simulation, learning paths, company research & personalized career strategy.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Professional Analysis</h3>
        <p>ATS scoring, job matching, skills categorization, readability assessment & competitive positioning insights.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start section
    st.subheader("🚀 Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Upload Resume & Job Description", use_container_width=True):
            st.session_state.current_page = "Upload"
            st.rerun()

    with col2:
        if st.button("📊 View Sample Analysis", use_container_width=True):
            st.session_state.current_page = "Analysis"
            st.rerun()

    # Features overview
    st.markdown("---")
    st.subheader("✨ Key Features")

    features = [
        "📄 Multi-format resume upload (PDF, DOCX, TXT)",
        "🎯 Advanced ATS scoring algorithm",
        "🔍 Keyword gap analysis",
        "✨ AI-powered resume optimization",
        "💼 Job match percentage calculation",
        "📝 Cover letter generation",
        "❓ Interview question preparation",
        "📊 Analytics and progress tracking",
        "🏭 Industry-specific templates",
        "📱 Mobile-responsive design"
    ]

    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"• {feature}")

def show_upload_page():
    """Display upload page"""
    st.header("📤 Upload Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Upload")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        resume_file = st.file_uploader(
            "Choose your resume file",
            type=SUPPORTED_FILE_TYPES,
            help="Supported formats: PDF, DOCX, TXT"
        )

        if resume_file:
            st.success(f"✅ Uploaded: {resume_file.name}")

            with st.spinner("Extracting text..."):
                resume_text = extract_text_from_file(resume_file)
                st.session_state.resume_text = resume_text
                st.session_state.resume_filename = resume_file.name

            # Show preview
            with st.expander("📖 Preview Resume Content"):
                st.text_area("Resume Text", resume_text, height=200, disabled=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("💼 Job Description")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        # Option 1: Text input
        jd_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Copy and paste the job description from the job posting..."
        )

        # Option 2: File upload
        st.markdown("**OR**")
        jd_file = st.file_uploader(
            "Upload job description file",
            type=SUPPORTED_FILE_TYPES,
            help="Upload a file containing the job description"
        )

        if jd_file:
            jd_text = extract_text_from_file(jd_file)
            st.success(f"✅ Uploaded: {jd_file.name}")

        if jd_text:
            st.session_state.jd_text = jd_text

            # Company name input
            company_name = st.text_input("Company Name (Optional)", placeholder="e.g., Google, Microsoft")
            if company_name:
                st.session_state.company_name = company_name

        st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    if st.session_state.get('resume_text') and st.session_state.get('jd_text'):
        st.markdown("---")
        st.success("✅ Both documents uploaded successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Start Analysis", use_container_width=True):
                st.session_state.current_page = "Analysis"
                st.rerun()

        with col2:
            if st.button("✨ Optimize Resume", use_container_width=True):
                st.session_state.current_page = "Optimization"
                st.rerun()

        with col3:
            if st.button("📋 Compare Jobs", use_container_width=True):
                st.session_state.current_page = "Comparison"
                st.rerun()

def show_analysis_page():
    """Display analysis page with comprehensive ATS and job match analysis"""
    st.header("📊 Resume Analysis")

    if not st.session_state.get('resume_text') or not st.session_state.get('jd_text'):
        st.warning("⚠️ Please upload both resume and job description first.")
        if st.button("📤 Go to Upload Page"):
            st.session_state.current_page = "Upload"
            st.rerun()
        return

    # Calculate scores
    with st.spinner("Analyzing your resume..."):
        ats_result = calculate_ats_score(st.session_state.resume_text, st.session_state.jd_text)
        match_result = calculate_job_match_percentage(st.session_state.resume_text, st.session_state.jd_text)

    # Overall scores section
    st.subheader("🎯 Overall Scores")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_score_with_progress("ATS Score", ats_result['overall_score'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_score_with_progress("Job Match", match_result['match_percentage'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Calculate readiness score
        readiness_score = (ats_result['overall_score'] + match_result['match_percentage']) / 2
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_score_with_progress("Apply Readiness", readiness_score)
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed breakdown
    st.markdown("---")
    st.subheader("📈 Detailed Breakdown")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 ATS Analysis", "🔍 Keyword Analysis", "💼 Job Matching", "📋 Recommendations", "🔬 Advanced Analysis"])

    with tab1:
        st.markdown("### ATS Score Breakdown")

        breakdown = ats_result['breakdown']

        col1, col2 = st.columns(2)

        with col1:
            display_score_with_progress("Keyword Match", breakdown['keyword_match'])
            display_score_with_progress("Skills Match", breakdown['skills_match'])

        with col2:
            display_score_with_progress("Semantic Similarity", breakdown['semantic_similarity'])
            display_score_with_progress("Format & Structure", breakdown['format_structure'])

        # Common keywords
        if breakdown['common_keywords']:
            st.success(f"✅ **Matched Keywords ({len(breakdown['common_keywords'])}):**")
            st.write(", ".join(breakdown['common_keywords'][:20]))

        # Missing keywords
        if breakdown['missing_keywords']:
            st.warning(f"⚠️ **Missing Keywords ({len(breakdown['missing_keywords'])}):**")
            st.write(", ".join(breakdown['missing_keywords'][:20]))

    with tab2:
        st.markdown("### Keyword Analysis")

        resume_keywords = extract_keywords(st.session_state.resume_text)
        jd_keywords = extract_keywords(st.session_state.jd_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Resume Keywords**")
            if resume_keywords:
                if HAS_PANDAS and HAS_PLOTLY:
                    # Create keyword frequency chart
                    keyword_freq = pd.DataFrame({
                        'Keyword': resume_keywords[:15],
                        'Count': [1] * min(15, len(resume_keywords))
                    })

                    fig = px.bar(keyword_freq, x='Count', y='Keyword', orientation='h',
                               title="Top Resume Keywords")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Simple text display without charts
                    st.write("**Top Resume Keywords:**")
                    for i, keyword in enumerate(resume_keywords[:15], 1):
                        st.write(f"{i}. {keyword}")

        with col2:
            st.markdown("**Job Description Keywords**")
            if jd_keywords:
                if HAS_PANDAS and HAS_PLOTLY:
                    keyword_freq = pd.DataFrame({
                        'Keyword': jd_keywords[:15],
                        'Count': [1] * min(15, len(jd_keywords))
                    })

                    fig = px.bar(keyword_freq, x='Count', y='Keyword', orientation='h',
                               title="Top JD Keywords")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Simple text display without charts
                    st.write("**Top JD Keywords:**")
                    for i, keyword in enumerate(jd_keywords[:15], 1):
                        st.write(f"{i}. {keyword}")

        # Skills analysis
        st.markdown("### Skills Analysis")
        resume_skills = extract_skills(st.session_state.resume_text)
        jd_skills = extract_skills(st.session_state.jd_text)

        col1, col2 = st.columns(2)

        with col1:
            if resume_skills:
                st.success(f"**Your Skills ({len(resume_skills)}):**")
                for skill in resume_skills[:10]:
                    st.write(f"• {skill}")

        with col2:
            if jd_skills:
                st.info(f"**Required Skills ({len(jd_skills)}):**")
                for skill in jd_skills[:10]:
                    st.write(f"• {skill}")

    with tab3:
        st.markdown("### Job Match Analysis")

        factors = match_result['factors']

        if HAS_PLOTLY:
            # Create radar chart for match factors
            categories = list(factors.keys())
            values = list(factors.values())

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Match Score'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Job Match Factors"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple display without charts
            st.write("**Job Match Factors:**")
            for factor, score in factors.items():
                st.write(f"• {factor.replace('_', ' ').title()}: {score:.1f}%")

        # Factor details
        for factor, score in factors.items():
            display_score_with_progress(factor.replace('_', ' ').title(), score)

    with tab4:
        st.markdown("### Recommendations")

        recommendations = ats_result['recommendations']

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.success("🎉 Great job! Your resume is well-optimized for this position.")

        # Action items
        st.markdown("### 🎯 Action Items")

        action_items = []

        if ats_result['overall_score'] < 70:
            action_items.append("Improve ATS score by incorporating more relevant keywords")

        if match_result['match_percentage'] < 60:
            action_items.append("Better align your experience with job requirements")

        if len(ats_result['breakdown']['missing_keywords']) > 5:
            action_items.append("Add missing technical skills and keywords")

        if ats_result['breakdown']['format_structure'] < 80:
            action_items.append("Improve resume formatting and structure")

        if not action_items:
            st.success("✅ Your resume looks great! Consider minor optimizations in the Optimization tab.")
        else:
            for item in action_items:
                st.markdown(f"• {item}")

    with tab5:
        st.markdown("### 🔬 Advanced AI Analysis")

        api_key = st.session_state.get('api_key')
        if not api_key:
            st.warning("⚠️ OpenRouter API key required for advanced analysis features.")
            return

        # Advanced analysis options
        analysis_type = st.selectbox(
            "Choose Analysis Type",
            ["Grammar & Spelling", "Action Verbs", "Quantification", "Readability", "Skills Analysis"],
            key="advanced_analysis_type"
        )

        if st.button("🔍 Run Advanced Analysis", key="run_advanced_analysis"):

            if analysis_type == "Grammar & Spelling":
                with st.spinner("Checking grammar and spelling..."):
                    result = check_grammar_and_spelling(st.session_state.resume_text, api_key)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Errors Found", result['errors_found'])
                    with col2:
                        score = max(0, 100 - (result['errors_found'] * 5))
                        st.metric("Grammar Score", f"{score}%")

                    if result['corrections']:
                        st.subheader("📝 Corrections Needed")
                        for correction in result['corrections'][:10]:
                            st.write(f"• {correction}")

                    if result['suggestions']:
                        st.subheader("💡 Writing Suggestions")
                        for suggestion in result['suggestions'][:5]:
                            st.write(f"• {suggestion}")

                    if result['corrected_text'] != st.session_state.resume_text:
                        with st.expander("📄 Corrected Version"):
                            st.text_area("Corrected Text", result['corrected_text'], height=300, key="corrected_text_display")

            elif analysis_type == "Action Verbs":
                with st.spinner("Analyzing action verbs..."):
                    result = analyze_action_verbs(st.session_state.resume_text, api_key)

                    col1, col2 = st.columns(2)
                    with col1:
                        if result['weak_verbs']:
                            st.subheader("⚠️ Weak Verbs Found")
                            for verb in result['weak_verbs'][:10]:
                                st.write(f"• {verb}")

                    with col2:
                        if result['strong_alternatives']:
                            st.subheader("💪 Stronger Alternatives")
                            for alt in result['strong_alternatives'][:10]:
                                st.write(f"• {alt}")

                    if result['improved_sentences']:
                        st.subheader("✨ Improved Sentences")
                        for sentence in result['improved_sentences'][:5]:
                            st.write(f"• {sentence}")

            elif analysis_type == "Quantification":
                with st.spinner("Detecting quantification opportunities..."):
                    result = detect_quantification_opportunities(st.session_state.resume_text, api_key)

                    if result['missing_metrics']:
                        st.subheader("📊 Missing Quantification")
                        for metric in result['missing_metrics'][:8]:
                            st.write(f"• {metric}")

                    if result['quantified_examples']:
                        st.subheader("✨ Quantified Examples")
                        for example in result['quantified_examples'][:5]:
                            st.write(f"• {example}")

                    if result['metric_types']:
                        st.subheader("📈 Relevant Metrics")
                        for metric_type in result['metric_types'][:8]:
                            st.write(f"• {metric_type}")

            elif analysis_type == "Readability":
                with st.spinner("Analyzing readability..."):
                    result = analyze_readability(st.session_state.resume_text, api_key)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Readability Score", f"{result['readability_score']}/10")
                    with col2:
                        st.metric("Complexity Level", result['complexity_level'])
                    with col3:
                        color = "🟢" if result['readability_score'] >= 7 else "🟡" if result['readability_score'] >= 5 else "🔴"
                        st.metric("Status", f"{color} {'Good' if result['readability_score'] >= 7 else 'Needs Work'}")

                    if result['issues']:
                        st.subheader("⚠️ Readability Issues")
                        for issue in result['issues'][:8]:
                            st.write(f"• {issue}")

                    if result['improvements']:
                        st.subheader("💡 Improvements")
                        for improvement in result['improvements'][:8]:
                            st.write(f"• {improvement}")

            elif analysis_type == "Skills Analysis":
                with st.spinner("Analyzing skills..."):
                    result = categorize_skills(st.session_state.resume_text, api_key)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🔧 Hard Skills")
                        for skill in result['hard_skills'][:10]:
                            st.write(f"• {skill}")

                        if result['missing_hard_skills']:
                            st.subheader("⚠️ Missing Hard Skills")
                            for skill in result['missing_hard_skills'][:8]:
                                st.write(f"• {skill}")

                    with col2:
                        st.subheader("🤝 Soft Skills")
                        for skill in result['soft_skills'][:10]:
                            st.write(f"• {skill}")

                        if result['missing_soft_skills']:
                            st.subheader("⚠️ Missing Soft Skills")
                            for skill in result['missing_soft_skills'][:8]:
                                st.write(f"• {skill}")

                    st.metric("Overall Skill Strength", f"{result['skill_strength']}/10")

def show_resume_tools_page():
    """Display resume tools page with templates, AI rewrite, and optimization"""
    st.header("✨ Resume Tools")

    if not st.session_state.get('resume_text') or not st.session_state.get('jd_text'):
        st.warning("⚠️ Please upload both resume and job description first.")
        return

    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key to use AI features.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Templates & Formatting", "✨ AI Rewrite & Optimization", "📝 Cover Letter Generator", "🔍 Section Analysis"])

    with tab1:
        st.subheader("AI-Powered Resume Rewrite")

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🚀 Optimize Resume", use_container_width=True):
                with st.spinner("AI is optimizing your resume..."):
                    optimized_resume = rewrite_resume(
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        api_key
                    )
                    st.session_state.optimized_resume = optimized_resume

        if st.session_state.get('optimized_resume'):
            st.markdown("### 📄 Optimized Resume")

            # Show before/after comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Resume**")
                st.text_area("", st.session_state.resume_text, height=400, disabled=True, key="original_resume")

            with col2:
                st.markdown("**Optimized Resume**")
                optimized_text = st.text_area("", st.session_state.optimized_resume, height=400, key="optimized_resume")

                # Update if user edits
                if optimized_text != st.session_state.optimized_resume:
                    st.session_state.optimized_resume = optimized_text

            # Save version
            col1, col2, col3 = st.columns(3)

            with col1:
                version_name = st.text_input("Version Name", value=f"Optimized_{datetime.now().strftime('%Y%m%d_%H%M')}")

            with col2:
                if st.button("💾 Save Version"):
                    if 'resume_versions' not in st.session_state:
                        st.session_state.resume_versions = []

                    st.session_state.resume_versions.append({
                        'name': version_name,
                        'content': st.session_state.optimized_resume,
                        'created_at': datetime.now(),
                        'ats_score': calculate_ats_score(st.session_state.optimized_resume, st.session_state.jd_text)['overall_score']
                    })
                    st.success("✅ Version saved!")

            with col3:
                # Download options
                st.download_button(
                    "📥 Download TXT",
                    st.session_state.optimized_resume,
                    file_name=f"optimized_resume_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    with tab2:
        st.subheader("AI Cover Letter Generator")

        company_name = st.text_input("Company Name", value=st.session_state.get('company_name', ''))

        if st.button("✉️ Generate Cover Letter", use_container_width=True):
            with st.spinner("Generating personalized cover letter..."):
                cover_letter = generate_cover_letter(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                    api_key,
                    company_name
                )
                st.session_state.cover_letter = cover_letter

        if st.session_state.get('cover_letter'):
            st.markdown("### 📝 Generated Cover Letter")

            cover_letter_text = st.text_area(
                "",
                st.session_state.cover_letter,
                height=400,
                key="cover_letter_edit"
            )

            # Update if user edits
            if cover_letter_text != st.session_state.cover_letter:
                st.session_state.cover_letter = cover_letter_text

            st.download_button(
                "📥 Download Cover Letter",
                st.session_state.cover_letter,
                file_name=f"cover_letter_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

    with tab3:
        st.subheader("Section-by-Section Analysis")

        if st.button("🔍 Analyze Sections", use_container_width=True):
            with st.spinner("Analyzing resume sections..."):
                section_analysis = analyze_resume_sections(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                    api_key
                )
                st.session_state.section_analysis = section_analysis

        if st.session_state.get('section_analysis'):
            sections = st.session_state.section_analysis

            for section_name, feedback in sections.items():
                if feedback.strip():
                    with st.expander(f"📋 {section_name.title()} Section"):
                        st.write(feedback)

    with tab4:
        st.subheader("Interview Preparation")

        if st.button("❓ Generate Interview Questions", use_container_width=True):
            with st.spinner("Generating interview questions..."):
                questions = generate_interview_questions(st.session_state.jd_text, api_key)
                st.session_state.interview_questions = questions

        if st.session_state.get('interview_questions'):
            st.markdown("### 🎯 Potential Interview Questions")

            for i, question in enumerate(st.session_state.interview_questions, 1):
                with st.expander(f"Question {i}"):
                    st.write(question)

                    # Space for user to write answers
                    answer = st.text_area(f"Your Answer", key=f"answer_{i}", height=100)

                    if answer:
                        if 'interview_answers' not in st.session_state:
                            st.session_state.interview_answers = {}
                        st.session_state.interview_answers[i] = answer


def show_comparison_page():
    """Display comparison page for multiple jobs and resume versions"""
    st.header("📋 Comparison Dashboard")

    tab1, tab2 = st.tabs(["💼 Multiple Jobs", "📄 Resume Versions"])

    with tab1:
        st.subheader("Compare Multiple Job Descriptions")

        if not st.session_state.get('resume_text'):
            st.warning("⚠️ Please upload your resume first.")
            return

        # Add new job description
        with st.expander("➕ Add New Job Description"):
            col1, col2 = st.columns(2)

            with col1:
                job_title = st.text_input("Job Title")
                company = st.text_input("Company")

            with col2:
                location = st.text_input("Location")
                salary = st.text_input("Salary Range (Optional)")

            jd_text = st.text_area("Job Description", height=200)

            if st.button("➕ Add Job") and job_title and jd_text:
                if 'job_descriptions' not in st.session_state:
                    st.session_state.job_descriptions = []

                job_data = {
                    'id': len(st.session_state.job_descriptions) + 1,
                    'title': job_title,
                    'company': company,
                    'location': location,
                    'salary': salary,
                    'description': jd_text,
                    'added_at': datetime.now()
                }

                st.session_state.job_descriptions.append(job_data)
                st.success(f"✅ Added job: {job_title} at {company}")

        # Display comparison table
        if st.session_state.get('job_descriptions'):
            st.markdown("### 📊 Job Comparison Results")

            comparison_data = []

            for job in st.session_state.job_descriptions:
                ats_result = calculate_ats_score(st.session_state.resume_text, job['description'])
                match_result = calculate_job_match_percentage(st.session_state.resume_text, job['description'])

                comparison_data.append({
                    'Job Title': job['title'],
                    'Company': job['company'],
                    'ATS Score': f"{ats_result['overall_score']:.1f}%",
                    'Job Match': f"{match_result['match_percentage']:.1f}%",
                    'Readiness': f"{(ats_result['overall_score'] + match_result['match_percentage']) / 2:.1f}%",
                    'Location': job['location'],
                    'Salary': job['salary']
                })

            if HAS_PANDAS:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                if HAS_PLOTLY:
                    # Visualization
                    fig = px.scatter(
                        df,
                        x='ATS Score',
                        y='Job Match',
                        size='Readiness',
                        hover_data=['Job Title', 'Company'],
                        title="Job Opportunities Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple table display
                st.write("**Job Comparison Results:**")
                for i, job in enumerate(comparison_data, 1):
                    st.write(f"**{i}. {job['Job Title']} at {job['Company']}**")
                    st.write(f"   • ATS Score: {job['ATS Score']}")
                    st.write(f"   • Job Match: {job['Job Match']}")
                    st.write(f"   • Readiness: {job['Readiness']}")
                    st.write("---")

    with tab2:
        st.subheader("Resume Version Comparison")

        if st.session_state.get('resume_versions'):
            st.markdown("### 📄 Saved Resume Versions")

            version_data = []

            for version in st.session_state.resume_versions:
                version_data.append({
                    'Version Name': version['name'],
                    'ATS Score': f"{version['ats_score']:.1f}%",
                    'Created': version['created_at'].strftime('%Y-%m-%d %H:%M'),
                    'Word Count': len(version['content'].split())
                })

            if HAS_PANDAS:
                df = pd.DataFrame(version_data)
                st.dataframe(df, use_container_width=True)
            else:
                # Simple table display
                st.write("**Resume Versions:**")
                for version in version_data:
                    st.write(f"• **{version['Version Name']}** - {version['ATS Score']} - {version['Created']}")

            # Version selector for detailed view
            selected_versions = st.multiselect(
                "Select versions to compare",
                options=[v['name'] for v in st.session_state.resume_versions],
                max_selections=3
            )

            if selected_versions:
                cols = st.columns(len(selected_versions))

                for i, version_name in enumerate(selected_versions):
                    version = next(v for v in st.session_state.resume_versions if v['name'] == version_name)

                    with cols[i]:
                        st.markdown(f"**{version_name}**")
                        st.text_area("", version['content'], height=300, disabled=True, key=f"version_{i}")
                        st.metric("ATS Score", f"{version['ats_score']:.1f}%")
        else:
            st.info("💡 No saved resume versions yet. Create optimized versions in the Optimization tab.")

def show_analytics_page():
    """Display analytics dashboard"""
    st.header("📈 Analytics Dashboard")

    if not st.session_state.get('ats_history'):
        st.info("📊 No analytics data yet. Start analyzing resumes to see your progress!")
        return

    if HAS_PANDAS and HAS_NUMPY:
        # Create sample data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        scores = np.random.normal(65, 15, 30)
        scores = np.clip(scores, 0, 100)

        analytics_data = pd.DataFrame({
            'Date': dates,
            'ATS_Score': scores,
            'Job_Match': scores + np.random.normal(0, 5, 30)
        })

        # Progress chart
        if HAS_PLOTLY:
            fig = px.line(analytics_data, x='Date', y=['ATS_Score', 'Job_Match'],
                          title="ATS Score Progress Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Sample Analytics Data:**")
            st.write(f"Average ATS Score: {analytics_data['ATS_Score'].mean():.1f}%")
            st.write(f"Best ATS Score: {analytics_data['ATS_Score'].max():.1f}%")
    else:
        # Simple metrics without data science libraries
        analytics_data = None

    col1, col2, col3, col4 = st.columns(4)

    if analytics_data is not None:
        with col1:
            st.metric("Average ATS Score", f"{analytics_data['ATS_Score'].mean():.1f}%",
                     delta=f"{np.random.uniform(-5, 5):.1f}%")

        with col2:
            st.metric("Best ATS Score", f"{analytics_data['ATS_Score'].max():.1f}%")
    else:
        with col1:
            st.metric("Average ATS Score", "65.0%", delta="2.5%")

        with col2:
            st.metric("Best ATS Score", "85.0%")

    with col3:
        st.metric("Applications", "12", delta="3")

    with col4:
        st.metric("Interviews", "4", delta="2")

    # Skills improvement tracking
    st.subheader("🎯 Skills Development")

    if HAS_PANDAS and HAS_PLOTLY:
        skills_data = pd.DataFrame({
            'Skill': ['Python', 'Machine Learning', 'SQL', 'Project Management', 'Communication'],
            'Current Level': [85, 70, 90, 65, 80],
            'Target Level': [95, 85, 95, 80, 90]
        })

        fig = px.bar(skills_data, x='Skill', y=['Current Level', 'Target Level'],
                     title="Skills Gap Analysis", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Simple skills display
        skills = [
            ('Python', 85, 95),
            ('Machine Learning', 70, 85),
            ('SQL', 90, 95),
            ('Project Management', 65, 80),
            ('Communication', 80, 90)
        ]

        st.write("**Skills Development Progress:**")
        for skill, current, target in skills:
            st.write(f"• **{skill}**: {current}% → {target}% (Target)")
            st.progress(current / 100)









def show_interview_mastery_page():
    """Display comprehensive interview mastery page"""
    st.header("❓ Interview Mastery")

    if not st.session_state.get('jd_text'):
        st.warning("⚠️ Please upload a job description first to get targeted interview preparation.")
        if st.button("📤 Go to Upload Page"):
            st.session_state.current_page = "Upload"
            st.rerun()
        return

    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key to use AI-powered interview preparation.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎭 Mock Interview Simulator", "🏢 Company-Specific Questions", "🎭 Behavioral Practice", "🔧 Technical Prep", "💰 Salary Negotiation", "📧 Communication Templates"])

    with tab1:
        st.subheader("🎭 Mock Interview Simulator")

        interview_type = st.selectbox(
            "Interview Type",
            ["Behavioral", "Technical", "Case Study", "Panel", "Phone"],
            key="mock_interview_type_mastery"
        )

        if st.button("🎭 Start Mock Interview", key="start_mock_interview_mastery"):
            with st.spinner("Preparing interview questions..."):
                mock_result = conduct_mock_interview(st.session_state.jd_text, interview_type, api_key)

                st.markdown(f"### 🎤 {interview_type} Mock Interview")

                for i, question_data in enumerate(mock_result['questions'], 1):
                    with st.expander(f"Question {i}"):
                        st.write(f"**Question:** {question_data.get('question', 'N/A')}")

                        if question_data.get('follow_up'):
                            st.write(f"**Follow-up:** {question_data['follow_up']}")

                        # Answer input
                        answer = st.text_area(f"Your Answer", key=f"mock_answer_mastery_{i}", height=120)

                        if st.button(f"Analyze Answer {i}", key=f"analyze_mock_mastery_{i}"):
                            if answer:
                                with st.spinner("Analyzing your response..."):
                                    analysis = analyze_interview_response(question_data['question'], answer, api_key)

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Score", f"{analysis['score']}/10")

                                    if analysis['strengths']:
                                        st.subheader("✅ Strengths")
                                        for strength in analysis['strengths']:
                                            st.write(f"• {strength}")

                                    if analysis['weaknesses']:
                                        st.subheader("⚠️ Areas for Improvement")
                                        for weakness in analysis['weaknesses']:
                                            st.write(f"• {weakness}")

                                    if analysis['improved_version']:
                                        st.subheader("✨ Improved Version")
                                        st.write(analysis['improved_version'])

                                    if analysis['tips']:
                                        st.subheader("💡 Tips")
                                        for tip in analysis['tips']:
                                            st.write(f"• {tip}")
                            else:
                                st.warning("Please provide an answer first.")

    with tab2:
        st.subheader("🏢 Company-Specific Interview Questions")

        company_name = st.text_input("Company Name", placeholder="e.g., Google, Microsoft, Apple", key="interview_company_name_mastery")

        if st.button("Generate Company Questions", key="generate_company_questions_mastery"):
            if company_name:
                with st.spinner(f"Generating interview questions for {company_name}..."):
                    questions = generate_company_specific_questions(company_name, st.session_state.jd_text, api_key)

                    st.markdown(f"### 🎯 Interview Questions for {company_name}")

                    for i, question in enumerate(questions, 1):
                        with st.expander(f"Question {i}"):
                            st.write(question)

                            # Space for user to prepare answer
                            answer = st.text_area(f"Your Answer", key=f"company_answer_mastery_{i}", height=100)

                            if answer:
                                if 'company_interview_answers_mastery' not in st.session_state:
                                    st.session_state.company_interview_answers_mastery = {}
                                st.session_state.company_interview_answers_mastery[i] = answer
            else:
                st.warning("Please enter a company name.")

    with tab3:
        st.subheader("🎭 Behavioral Interview Questions")
        st.markdown("**Practice using the STAR method:** Situation, Task, Action, Result")

        if st.button("Generate Behavioral Questions", key="generate_behavioral_questions_mastery"):
            with st.spinner("Generating behavioral questions..."):
                categories = generate_behavioral_questions(st.session_state.jd_text, api_key)

                for category, questions in categories.items():
                    if questions:
                        st.markdown(f"### 🎯 {category.replace('_', ' ').title()} Questions")

                        for i, question in enumerate(questions, 1):
                            with st.expander(f"{category.title()} Question {i}"):
                                st.write(question)

                                # STAR method template
                                st.markdown("**STAR Method Template:**")
                                col1, col2 = st.columns(2)

                                with col1:
                                    situation = st.text_area("Situation", key=f"{category}_situation_mastery_{i}", height=80)
                                    task = st.text_area("Task", key=f"{category}_task_mastery_{i}", height=80)

                                with col2:
                                    action = st.text_area("Action", key=f"{category}_action_mastery_{i}", height=80)
                                    result = st.text_area("Result", key=f"{category}_result_mastery_{i}", height=80)

    with tab4:
        st.subheader("🔧 Technical Interview Questions")

        if st.button("Generate Technical Questions", key="generate_technical_questions_mastery"):
            with st.spinner("Generating technical questions..."):
                categories = generate_technical_questions(st.session_state.jd_text, api_key)

                for category, questions in categories.items():
                    if questions:
                        st.markdown(f"### 💻 {category.replace('_', ' ').title()}")

                        for i, question in enumerate(questions, 1):
                            with st.expander(f"Question {i}"):
                                st.write(question)

                                # Space for technical answer
                                answer = st.text_area(f"Your Answer/Solution", key=f"tech_{category}_mastery_{i}", height=120)

    with tab5:
        st.subheader("💰 Salary Negotiation Preparation")

        experience_level = st.selectbox(
            "Your Experience Level",
            ["Entry Level (0-2 years)", "Mid Level (3-7 years)", "Senior Level (8-15 years)", "Executive Level (15+ years)"],
            key="interview_experience_level_mastery"
        )

        if st.button("Generate Negotiation Guide", key="generate_interview_salary_guide_mastery"):
            with st.spinner("Generating salary negotiation guide..."):
                result = generate_salary_negotiation_guide(st.session_state.jd_text, experience_level, api_key)

                st.markdown("### 💼 Your Negotiation Strategy")

                if result['negotiation_timing']:
                    st.subheader("⏰ When to Negotiate")
                    st.info(result['negotiation_timing'])

                if result['research_tips']:
                    st.subheader("🔍 Research Tips")
                    for tip in result['research_tips']:
                        st.write(f"• {tip}")

                if result['opening_scripts']:
                    st.subheader("🎭 Negotiation Scripts")
                    for i, script in enumerate(result['opening_scripts'], 1):
                        with st.expander(f"Script Option {i}"):
                            st.write(script)

                col1, col2 = st.columns(2)
                with col1:
                    if result['non_salary_benefits']:
                        st.subheader("🎁 Other Benefits")
                        for benefit in result['non_salary_benefits']:
                            st.write(f"• {benefit}")

                with col2:
                    if result['common_mistakes']:
                        st.subheader("⚠️ Avoid These Mistakes")
                        for mistake in result['common_mistakes']:
                            st.write(f"• {mistake}")

    with tab6:
        st.subheader("📧 Interview Communication Templates")

        communication_type = st.selectbox(
            "Communication Type",
            ["Thank You Email", "Follow-up Email", "Interview Confirmation", "Availability Response", "Decline Offer"],
            key="interview_communication_type_mastery"
        )

        col1, col2 = st.columns(2)
        with col1:
            interviewer_name = st.text_input("Interviewer Name", key="interview_interviewer_name_mastery")
        with col2:
            interview_type = st.selectbox(
                "Interview Type",
                ["Phone", "Video", "In-person", "Panel", "Technical"],
                key="interview_type_comm_mastery"
            )

        if st.button("Generate Communication Template", key="generate_interview_comm_mastery"):
            if interviewer_name:
                with st.spinner("Generating communication templates..."):
                    if communication_type == "Thank You Email":
                        templates = generate_thank_you_notes(interview_type, interviewer_name, api_key)

                        st.markdown("### 📧 Thank You Email Templates")
                        for style, content in templates.items():
                            with st.expander(f"📝 {style.title()} Style"):
                                st.text_area("", content, height=200, key=f"interview_thankyou_mastery_{style}")

                    else:
                        templates = generate_email_templates(communication_type, "Company", api_key)

                        st.markdown(f"### 📧 {communication_type} Templates")
                        for template_name, content in templates.items():
                            with st.expander(f"📝 {template_name.replace('_', ' ').title()}"):
                                st.text_area("", content, height=200, key=f"interview_comm_mastery_{template_name}")
            else:
                st.warning("Please enter the interviewer's name.")

def show_skills_learning_page():
    """Display skills assessment and learning page"""
    st.header("📚 Skills & Learning")

    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key to use AI-powered learning tools.")
        return

    tab1, tab2, tab3 = st.tabs(["📚 Skills Assessment & Learning", "🎓 Course Recommendations", "📊 Progress Tracking"])

    with tab1:
        st.subheader("📚 Skills Assessment & Learning")

        # Skills Quiz Generator
        st.markdown("### 🧠 Skills Quiz Generator")

        col1, col2 = st.columns(2)
        with col1:
            skill_name = st.text_input("Skill to Test", placeholder="e.g., Python, Project Management, SQL", key="quiz_skill")
        with col2:
            difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"], key="quiz_difficulty")

        if st.button("🧠 Generate Skills Quiz", key="generate_quiz"):
            if skill_name:
                with st.spinner("Generating custom skills quiz..."):
                    quiz_result = generate_skills_quiz(skill_name, difficulty, api_key)

                    st.markdown(f"### 📝 {skill_name} Quiz ({difficulty} Level)")

                    for i, question in enumerate(quiz_result['questions'], 1):
                        with st.expander(f"Question {i}"):
                            st.write(f"**Q{i}:** {question.get('question', 'N/A')}")
                            if question.get('options'):
                                st.write(f"**Options:** {question['options']}")
                            if question.get('answer'):
                                st.write(f"**Answer:** {question['answer']}")
                            if question.get('explanation'):
                                st.write(f"**Explanation:** {question['explanation']}")
            else:
                st.warning("Please enter a skill name.")

        st.markdown("---")

        # Learning Path Creator
        st.markdown("### 🛤️ Personalized Learning Path")

        current_skills = st.text_area(
            "Current Skills",
            placeholder="e.g., Basic Python, Some SQL, Project coordination experience...",
            height=100,
            key="current_skills_learning"
        )

        target_role = st.text_input(
            "Target Role",
            placeholder="e.g., Data Scientist, Product Manager, Full Stack Developer",
            key="target_role_learning"
        )

        if st.button("🛤️ Create Learning Path", key="create_learning_path"):
            if current_skills and target_role:
                with st.spinner("Creating personalized learning path..."):
                    learning_result = create_learning_path(current_skills, target_role, api_key)

                    st.markdown("### 📚 Your Personalized Learning Path")

                    if learning_result['timeline']:
                        st.info(f"⏰ **Estimated Timeline:** {learning_result['timeline']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if learning_result['skill_gaps']:
                            st.subheader("🎯 Skills to Develop")
                            for skill in learning_result['skill_gaps']:
                                st.write(f"• {skill}")

                        if learning_result['learning_phases']:
                            st.subheader("📅 Learning Phases")
                            for phase in learning_result['learning_phases']:
                                st.write(f"• {phase}")

                    with col2:
                        if learning_result['milestones']:
                            st.subheader("🏆 Milestones")
                            for milestone in learning_result['milestones']:
                                st.write(f"• {milestone}")

                        if learning_result['practice_projects']:
                            st.subheader("🛠️ Practice Projects")
                            for project in learning_result['practice_projects']:
                                st.write(f"• {project}")

                    if learning_result['resources']:
                        st.subheader("📖 Learning Resources")
                        for resource in learning_result['resources']:
                            st.write(f"• {resource}")
            else:
                st.warning("Please fill in both current skills and target role.")

    with tab2:
        st.subheader("🎓 Free Course Recommendations")

        col1, col2 = st.columns(2)
        with col1:
            course_skill = st.text_input("Skill to Learn", key="course_skill")
        with col2:
            learning_style = st.selectbox(
                "Learning Style",
                ["Visual", "Hands-on", "Reading", "Video-based", "Interactive"],
                key="learning_style"
            )

        if st.button("🎓 Find Free Courses", key="find_courses"):
            if course_skill:
                with st.spinner("Finding free courses..."):
                    courses = recommend_free_courses(course_skill, learning_style, api_key)

                    st.markdown(f"### 📚 Free Courses for {course_skill}")

                    for course in courses:
                        with st.expander(f"📖 {course['name']} ({course['platform']})"):
                            st.write(f"**Platform:** {course['platform']}")
                            st.write(f"**Duration:** {course['duration']}")
                            st.write(f"**Description:** {course['description']}")
            else:
                st.warning("Please enter a skill name.")

    with tab3:
        st.subheader("📊 Progress Tracking")
        st.info("Track your learning progress and skill development over time.")

        # Simple progress tracking interface
        if st.button("📊 View Learning Progress", key="view_progress"):
            st.markdown("### 📈 Your Learning Journey")

            # Sample progress data (in real app, this would come from user data)
            progress_data = {
                "Python": 75,
                "SQL": 60,
                "Machine Learning": 45,
                "Project Management": 80
            }

            for skill, progress in progress_data.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{skill}**")
                    st.progress(progress / 100)
                with col2:
                    st.metric("", f"{progress}%")

def show_portfolio_projects_page():
    """Display portfolio and projects page"""
    st.header("📁 Portfolio & Projects")

    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key to use AI-powered portfolio tools.")
        return

    tab1, tab2, tab3 = st.tabs(["📁 Portfolio Tools", "🛠️ Project Showcase", "📖 Case Studies"])

    with tab1:
        st.subheader("📁 Portfolio Content Generator")

        project_info = st.text_area(
            "Project Information",
            placeholder="Describe your project: what it does, technologies used, your role, challenges faced...",
            height=150,
            key="portfolio_project_info"
        )

        target_audience = st.selectbox(
            "Target Audience",
            ["Hiring Managers", "Technical Recruiters", "Fellow Developers", "Clients", "General Public"],
            key="portfolio_audience"
        )

        if st.button("📝 Generate Portfolio Content", key="generate_portfolio"):
            if project_info:
                with st.spinner("Creating compelling portfolio content..."):
                    portfolio_result = generate_portfolio_content(project_info, target_audience, api_key)

                    st.markdown("### 📁 Generated Portfolio Content")

                    if portfolio_result['project_title']:
                        st.subheader("📌 Project Title")
                        st.write(portfolio_result['project_title'])

                    if portfolio_result['executive_summary']:
                        st.subheader("📋 Executive Summary")
                        st.write(portfolio_result['executive_summary'])

                    if portfolio_result['detailed_description']:
                        st.subheader("📖 Detailed Description")
                        st.write(portfolio_result['detailed_description'])

                    col1, col2 = st.columns(2)
                    with col1:
                        if portfolio_result['technologies_used']:
                            st.subheader("🛠️ Technologies Used")
                            st.write(portfolio_result['technologies_used'])

                        if portfolio_result['key_achievements']:
                            st.subheader("🏆 Key Achievements")
                            st.write(portfolio_result['key_achievements'])

                    with col2:
                        if portfolio_result['challenges_solved']:
                            st.subheader("🎯 Challenges Solved")
                            st.write(portfolio_result['challenges_solved'])

                        if portfolio_result['lessons_learned']:
                            st.subheader("💡 Lessons Learned")
                            st.write(portfolio_result['lessons_learned'])
            else:
                st.warning("Please provide project information.")

    with tab2:
        st.subheader("🛠️ Project Showcase Builder")

        # Project Impact Calculator
        st.markdown("### 📊 Project Impact Calculator")

        project_description = st.text_area(
            "Project Description",
            placeholder="Describe what your project accomplished, who it helped, what problems it solved...",
            height=120,
            key="impact_project_desc"
        )

        if st.button("📊 Calculate Project Impact", key="calculate_impact"):
            if project_description:
                with st.spinner("Analyzing project impact..."):
                    impact_result = calculate_project_impact(project_description, api_key)

                    st.markdown("### 📈 Project Impact Analysis")

                    col1, col2 = st.columns(2)
                    with col1:
                        if impact_result['impact_metrics']:
                            st.subheader("📊 Impact Metrics")
                            for metric in impact_result['impact_metrics']:
                                st.write(f"• {metric}")

                        if impact_result['business_value']:
                            st.subheader("💼 Business Value")
                            for value in impact_result['business_value']:
                                st.write(f"• {value}")

                    with col2:
                        if impact_result['user_benefits']:
                            st.subheader("👥 User Benefits")
                            for benefit in impact_result['user_benefits']:
                                st.write(f"• {benefit}")

                        if impact_result['technical_improvements']:
                            st.subheader("⚙️ Technical Improvements")
                            for improvement in impact_result['technical_improvements']:
                                st.write(f"• {improvement}")

                    if impact_result['suggested_numbers']:
                        st.subheader("🔢 Suggested Quantifications")
                        for number in impact_result['suggested_numbers']:
                            st.write(f"• {number}")
            else:
                st.warning("Please provide a project description.")

        st.markdown("---")

        # Project Presentation Builder
        st.markdown("### 🎯 Project Presentation Builder")

        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name", key="showcase_project_name")
            project_type = st.selectbox(
                "Project Type",
                ["Web Application", "Mobile App", "Data Analysis", "Machine Learning", "API Development", "Desktop Software", "Other"],
                key="showcase_project_type"
            )

        with col2:
            tech_stack = st.text_input("Tech Stack", placeholder="e.g., React, Node.js, MongoDB", key="showcase_tech_stack")
            project_duration = st.text_input("Duration", placeholder="e.g., 3 months, 6 weeks", key="showcase_duration")

        project_details = st.text_area(
            "Project Details",
            placeholder="Detailed description of the project, your role, challenges, solutions...",
            height=150,
            key="showcase_project_details"
        )

        if st.button("🎯 Build Project Showcase", key="build_showcase"):
            if project_name and project_details:
                with st.spinner("Building your project showcase..."):
                    showcase_result = build_project_showcase(project_name, project_type, tech_stack, project_details, api_key)

                    st.markdown(f"### 🚀 {project_name} Showcase")

                    if showcase_result['elevator_pitch']:
                        st.subheader("🎤 Elevator Pitch")
                        st.info(showcase_result['elevator_pitch'])

                    if showcase_result['technical_highlights']:
                        st.subheader("⚙️ Technical Highlights")
                        for highlight in showcase_result['technical_highlights']:
                            st.write(f"• {highlight}")

                    if showcase_result['demo_script']:
                        st.subheader("🎬 Demo Script")
                        st.write(showcase_result['demo_script'])

                    col1, col2 = st.columns(2)
                    with col1:
                        if showcase_result['key_features']:
                            st.subheader("✨ Key Features")
                            for feature in showcase_result['key_features']:
                                st.write(f"• {feature}")

                    with col2:
                        if showcase_result['lessons_learned']:
                            st.subheader("📚 Lessons Learned")
                            for lesson in showcase_result['lessons_learned']:
                                st.write(f"• {lesson}")
            else:
                st.warning("Please provide project name and details.")

    with tab3:
        st.subheader("📖 Case Study Generator")

        # Case Study Generator
        st.markdown("### 📚 Professional Case Study Creator")

        case_project_info = st.text_area(
            "Project Information for Case Study",
            placeholder="Provide detailed project information including problem, solution, implementation, results...",
            height=150,
            key="case_study_project_info"
        )

        case_target_role = st.text_input(
            "Target Role",
            placeholder="e.g., Software Engineer, Product Manager, Data Scientist",
            key="case_study_target_role"
        )

        if st.button("📚 Generate Case Study", key="generate_case_study"):
            if case_project_info and case_target_role:
                with st.spinner("Creating compelling case study..."):
                    case_result = create_case_study(case_project_info, case_target_role, api_key)

                    st.markdown("### 📖 Generated Case Study")

                    if case_result['case_study_title']:
                        st.subheader("📌 Title")
                        st.write(case_result['case_study_title'])

                    if case_result['problem_statement']:
                        st.subheader("🎯 Problem Statement")
                        st.write(case_result['problem_statement'])

                    if case_result['solution_approach']:
                        st.subheader("💡 Solution Approach")
                        st.write(case_result['solution_approach'])

                    if case_result['implementation']:
                        st.subheader("🛠️ Implementation")
                        st.write(case_result['implementation'])

                    if case_result['results']:
                        st.subheader("📈 Results")
                        st.write(case_result['results'])

                    if case_result['reflection']:
                        st.subheader("🤔 Reflection")
                        st.write(case_result['reflection'])
            else:
                st.warning("Please provide both project information and target role.")

        st.markdown("---")

        # Portfolio Website Content
        st.markdown("### 🌐 Portfolio Website Content")

        col1, col2 = st.columns(2)
        with col1:
            portfolio_style = st.selectbox(
                "Portfolio Style",
                ["Professional", "Creative", "Minimalist", "Technical", "Academic"],
                key="portfolio_style"
            )

        with col2:
            portfolio_focus = st.selectbox(
                "Focus Area",
                ["Full-Stack Development", "Data Science", "UI/UX Design", "Product Management", "DevOps", "Other"],
                key="portfolio_focus"
            )

        about_me = st.text_area(
            "About Me Information",
            placeholder="Tell us about yourself, your background, interests, career goals...",
            height=120,
            key="portfolio_about_me"
        )

        if st.button("🌐 Generate Portfolio Content", key="generate_portfolio_website"):
            if about_me:
                with st.spinner("Creating portfolio website content..."):
                    website_result = generate_portfolio_website_content(about_me, portfolio_style, portfolio_focus, api_key)

                    st.markdown("### 🌐 Portfolio Website Content")

                    if website_result['hero_section']:
                        st.subheader("🎯 Hero Section")
                        st.write(website_result['hero_section'])

                    if website_result['about_section']:
                        st.subheader("👤 About Section")
                        st.write(website_result['about_section'])

                    if website_result['skills_section']:
                        st.subheader("🛠️ Skills Section")
                        st.write(website_result['skills_section'])

                    col1, col2 = st.columns(2)
                    with col1:
                        if website_result['contact_cta']:
                            st.subheader("📞 Contact CTA")
                            st.write(website_result['contact_cta'])

                    with col2:
                        if website_result['testimonial_template']:
                            st.subheader("💬 Testimonial Template")
                            st.write(website_result['testimonial_template'])
            else:
                st.warning("Please provide information about yourself.")

def show_career_coach_page():
    """Display AI career coach page"""
    st.header("🤖 AI Career Coach")

    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key to use AI career coaching.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Career Coach", "💼 Career Strategy", "🏢 Company Research", "🚀 Content Generator"])

    with tab1:
        st.subheader("💬 Ask Your AI Career Coach")

        user_context = st.text_area(
            "Your Background (Optional)",
            placeholder="Brief background: current role, experience level, industry, goals...",
            height=80,
            key="chatbot_context"
        )

        career_question = st.text_area(
            "Your Career Question",
            placeholder="Ask anything: career change advice, interview tips, salary negotiation, skill development...",
            height=100,
            key="career_question"
        )

        if st.button("💬 Get Career Advice", key="get_career_advice"):
            if career_question:
                with st.spinner("Your AI career coach is thinking..."):
                    advice = career_chatbot_response(career_question, user_context or "No background provided", api_key)

                    st.markdown("### 🎯 AI Career Coach Response")
                    st.write(advice)
            else:
                st.warning("Please ask a career question.")

    with tab2:
        st.subheader("💼 Career Strategy & Planning")

        # Personalized Career Advice
        st.markdown("### 🎯 Personalized Career Strategy")

        current_situation = st.text_area(
            "Current Career Situation",
            placeholder="Describe your current role, challenges, what's working, what's not...",
            height=120,
            key="current_situation"
        )

        career_goals = st.text_area(
            "Career Goals",
            placeholder="What do you want to achieve? Short-term and long-term goals...",
            height=120,
            key="career_goals"
        )

        if st.button("🎯 Get Personalized Strategy", key="get_strategy"):
            if current_situation and career_goals:
                with st.spinner("Creating your personalized career strategy..."):
                    strategy_result = provide_career_advice(current_situation, career_goals, api_key)

                    st.markdown("### 🚀 Your Personalized Career Strategy")

                    if strategy_result['immediate_actions']:
                        st.subheader("⚡ Immediate Actions (Next 30 Days)")
                        for action in strategy_result['immediate_actions']:
                            st.write(f"• {action}")

                    if strategy_result['short_term_goals']:
                        st.subheader("📅 Short-term Goals (3-6 Months)")
                        for goal in strategy_result['short_term_goals']:
                            st.write(f"• {goal}")

                    if strategy_result['long_term_strategy']:
                        st.subheader("🎯 Long-term Strategy (1-2 Years)")
                        st.write(strategy_result['long_term_strategy'])

                    col1, col2 = st.columns(2)
                    with col1:
                        if strategy_result['potential_obstacles']:
                            st.subheader("⚠️ Potential Obstacles")
                            for obstacle in strategy_result['potential_obstacles']:
                                st.write(f"• {obstacle}")

                    with col2:
                        if strategy_result['success_metrics']:
                            st.subheader("📊 Success Metrics")
                            for metric in strategy_result['success_metrics']:
                                st.write(f"• {metric}")

                    if strategy_result['resources']:
                        st.subheader("📚 Helpful Resources")
                        for resource in strategy_result['resources']:
                            st.write(f"• {resource}")
            else:
                st.warning("Please provide both current situation and career goals.")

        st.markdown("---")

        # Career Progression Analysis
        st.markdown("### 📈 Career Progression Analysis")

        if st.session_state.get('resume_text'):
            if st.button("📈 Analyze Career Path", key="analyze_career_path"):
                with st.spinner("Analyzing your career progression..."):
                    progression_result = suggest_career_progression(st.session_state.resume_text, api_key)

                    st.markdown("### 🚀 Career Development Plan")

                    col1, col2 = st.columns(2)
                    with col1:
                        if progression_result['current_level']:
                            st.subheader("📊 Current Level")
                            st.write(progression_result['current_level'])

                        if progression_result['timeline']:
                            st.subheader("⏰ Timeline")
                            st.write(progression_result['timeline'])

                    with col2:
                        if progression_result['next_roles']:
                            st.subheader("🎯 Next Role Options")
                            for role in progression_result['next_roles']:
                                st.write(f"• {role}")

                    if progression_result['skill_gaps']:
                        st.subheader("📚 Skills to Develop")
                        for skill in progression_result['skill_gaps']:
                            st.write(f"• {skill}")

                    if progression_result['action_steps']:
                        st.subheader("✅ Action Steps")
                        for step in progression_result['action_steps']:
                            st.write(f"• {step}")

                    if progression_result['certifications']:
                        st.subheader("🏆 Recommended Certifications")
                        for cert in progression_result['certifications']:
                            st.write(f"• {cert}")
        else:
            st.info("💡 Upload your resume first to get personalized career progression analysis.")

    with tab3:
        st.subheader("🏢 Company Research Assistant")

        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Google, Microsoft, Tesla, Netflix",
            key="research_company_name"
        )

        if st.button("🔍 Research Company", key="research_company"):
            if company_name:
                with st.spinner(f"Researching {company_name}..."):
                    research_result = research_company_insights(company_name, api_key)

                    st.markdown(f"### 🏢 {company_name} Research Report")

                    if research_result['company_overview']:
                        st.subheader("🏢 Company Overview")
                        st.write(research_result['company_overview'])

                    if research_result['business_model']:
                        st.subheader("💼 Business Model")
                        st.write(research_result['business_model'])

                    col1, col2 = st.columns(2)
                    with col1:
                        if research_result['company_culture']:
                            st.subheader("🎭 Company Culture")
                            st.write(research_result['company_culture'])

                        if research_result['growth_trajectory']:
                            st.subheader("📈 Growth Trajectory")
                            st.write(research_result['growth_trajectory'])

                    with col2:
                        if research_result['recent_news']:
                            st.subheader("📰 Recent News")
                            for news in research_result['recent_news']:
                                st.write(f"• {news}")

                        if research_result['challenges']:
                            st.subheader("⚠️ Current Challenges")
                            for challenge in research_result['challenges']:
                                st.write(f"• {challenge}")

                    if research_result['interview_tips']:
                        st.subheader("💡 Interview Tips for This Company")
                        for tip in research_result['interview_tips']:
                            st.write(f"• {tip}")
            else:
                st.warning("Please enter a company name.")

    with tab4:
        st.subheader("🚀 Content Generator")

        content_type = st.selectbox(
            "Choose Content Type",
            ["Resume Bullet Points", "Achievement Quantifier", "Email Templates", "Networking Messages", "Thank You Notes"],
            key="content_generator_type"
        )

        if content_type == "Resume Bullet Points":
            st.markdown("### 📝 Resume Bullet Point Generator")
            basic_info = st.text_area(
                "Enter basic work information",
                placeholder="e.g., Worked as software engineer, developed web applications, managed team of 3 people...",
                height=150,
                key="bullet_basic_info"
            )

            if st.button("Generate Bullet Points", key="generate_bullets"):
                if basic_info:
                    with st.spinner("Generating professional bullet points..."):
                        bullets = generate_resume_bullets(basic_info, api_key)

                        st.markdown("### ✨ Generated Bullet Points")
                        for i, bullet in enumerate(bullets, 1):
                            st.write(f"{i}. {bullet}")

                        # Allow editing
                        edited_bullets = st.text_area(
                            "Edit bullets as needed",
                            value="\n".join([f"• {bullet}" for bullet in bullets]),
                            height=200,
                            key="edited_bullets_coach"
                        )
                else:
                    st.warning("Please enter some basic work information.")

        elif content_type == "Achievement Quantifier":
            st.markdown("### 📊 Achievement Quantifier")
            achievements = st.text_area(
                "Enter your achievements (one per line)",
                placeholder="e.g., Improved team productivity\nReduced processing time\nIncreased sales...",
                height=150,
                key="achievements_input"
            )

            if st.button("Quantify Achievements", key="quantify_achievements"):
                if achievements:
                    with st.spinner("Suggesting quantifications..."):
                        result = quantify_achievements(achievements, api_key)

                        st.markdown("### 📈 Quantified Achievements")
                        for achievement in result['achievements']:
                            with st.expander(f"📌 {achievement.get('original', 'Achievement')[:50]}..."):
                                st.write(f"**Original:** {achievement.get('original', 'N/A')}")
                                st.write(f"**Suggested Metrics:** {achievement.get('metrics', 'N/A')}")
                                st.write(f"**Quantified Version:** {achievement.get('quantified', 'N/A')}")
                else:
                    st.warning("Please enter some achievements to quantify.")

        elif content_type == "Email Templates":
            st.markdown("### ✉️ Email Template Generator")

            col1, col2 = st.columns(2)
            with col1:
                email_purpose = st.selectbox(
                    "Email Purpose",
                    ["Follow-up after interview", "Thank you note", "Application inquiry", "Networking outreach", "Salary negotiation"],
                    key="email_purpose"
                )
            with col2:
                company_name_email = st.text_input("Company Name", key="email_company_name")

            if st.button("Generate Email Templates", key="generate_emails"):
                if email_purpose and company_name_email:
                    with st.spinner("Generating email templates..."):
                        templates = generate_email_templates(email_purpose, company_name_email, api_key)

                        st.markdown("### 📧 Email Templates")
                        for template_name, template_content in templates.items():
                            with st.expander(f"📝 {template_name.replace('_', ' ').title()}"):
                                st.text_area("", template_content, height=200, key=f"email_{template_name}")
                else:
                    st.warning("Please select purpose and enter company name.")

        elif content_type == "Networking Messages":
            st.markdown("### 🤝 Networking Message Generator")

            col1, col2 = st.columns(2)
            with col1:
                industry = st.text_input("Industry", placeholder="e.g., Technology, Finance, Healthcare", key="networking_industry")
            with col2:
                purpose = st.selectbox(
                    "Networking Purpose",
                    ["Informational interview", "Job referral", "Industry insights", "Mentorship", "Collaboration"],
                    key="networking_purpose"
                )

            if st.button("Generate Networking Messages", key="generate_networking"):
                if industry and purpose:
                    with st.spinner("Generating networking messages..."):
                        messages = generate_networking_messages(industry, purpose, api_key)

                        st.markdown("### 💬 Networking Messages")
                        for i, message in enumerate(messages, 1):
                            with st.expander(f"Message {i}"):
                                st.write(message)
                else:
                    st.warning("Please enter industry and select purpose.")

        elif content_type == "Thank You Notes":
            st.markdown("### 🙏 Thank You Note Generator")

            col1, col2 = st.columns(2)
            with col1:
                interview_type = st.selectbox(
                    "Interview Type",
                    ["Phone interview", "Video interview", "In-person interview", "Panel interview", "Technical interview"],
                    key="thankyou_interview_type"
                )
            with col2:
                interviewer_name = st.text_input("Interviewer Name", key="interviewer_name_coach")

            if st.button("Generate Thank You Notes", key="generate_thankyou"):
                if interview_type and interviewer_name:
                    with st.spinner("Generating thank you notes..."):
                        notes = generate_thank_you_notes(interview_type, interviewer_name, api_key)

                        st.markdown("### 💌 Thank You Notes")
                        for note_type, note_content in notes.items():
                            with st.expander(f"📝 {note_type.title()} Style"):
                                st.text_area("", note_content, height=150, key=f"thankyou_{note_type}_coach")
                else:
                    st.warning("Please select interview type and enter interviewer name.")

def show_reports_page():
    """Display reports and export options"""
    st.header("📄 Reports & Export")

    if not st.session_state.get('resume_text') or not st.session_state.get('jd_text'):
        st.warning("⚠️ Please upload both resume and job description first.")
        return

    # Generate comprehensive report
    with st.spinner("Generating comprehensive report..."):
        ats_result = calculate_ats_score(st.session_state.resume_text, st.session_state.jd_text)
        match_result = calculate_job_match_percentage(st.session_state.resume_text, st.session_state.jd_text)

    tab1, tab2, tab3 = st.tabs(["📊 Summary Report", "📥 Download Options", "🎯 Readiness Check"])

    with tab1:
        st.subheader("📊 Comprehensive Analysis Report")

        # Executive summary
        st.markdown("### Executive Summary")

        readiness_score = (ats_result['overall_score'] + match_result['match_percentage']) / 2

        if readiness_score >= 80:
            st.success(f"🎉 **Excellent!** Your resume is {readiness_score:.1f}% ready for this position.")
        elif readiness_score >= 60:
            st.warning(f"⚠️ **Good Progress!** Your resume is {readiness_score:.1f}% ready. Some improvements recommended.")
        else:
            st.error(f"❌ **Needs Work!** Your resume is {readiness_score:.1f}% ready. Significant improvements needed.")

        # Detailed metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ATS Analysis")
            st.metric("Overall ATS Score", f"{ats_result['overall_score']:.1f}%")
            st.metric("Keyword Match", f"{ats_result['breakdown']['keyword_match']:.1f}%")
            st.metric("Skills Match", f"{ats_result['breakdown']['skills_match']:.1f}%")

        with col2:
            st.markdown("#### Job Matching")
            st.metric("Job Match Score", f"{match_result['match_percentage']:.1f}%")
            st.metric("Semantic Similarity", f"{ats_result['breakdown']['semantic_similarity']:.1f}%")
            st.metric("Format Score", f"{ats_result['breakdown']['format_structure']:.1f}%")

        # Recommendations summary
        st.markdown("### 🎯 Key Recommendations")

        recommendations = ats_result['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        # Missing elements
        if ats_result['breakdown']['missing_keywords']:
            st.markdown("### ⚠️ Missing Keywords")
            missing_keywords = ats_result['breakdown']['missing_keywords'][:10]
            st.write(", ".join(missing_keywords))

    with tab2:
        st.subheader("📥 Download Options")

        # Generate report content
        report_content = f"""
RESUME ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Overall Readiness Score: {readiness_score:.1f}%
ATS Score: {ats_result['overall_score']:.1f}%
Job Match Score: {match_result['match_percentage']:.1f}%

DETAILED BREAKDOWN
==================
Keyword Match: {ats_result['breakdown']['keyword_match']:.1f}%
Skills Match: {ats_result['breakdown']['skills_match']:.1f}%
Semantic Similarity: {ats_result['breakdown']['semantic_similarity']:.1f}%
Format & Structure: {ats_result['breakdown']['format_structure']:.1f}%

RECOMMENDATIONS
===============
{chr(10).join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])}

MISSING KEYWORDS
================
{', '.join(ats_result['breakdown']['missing_keywords'][:20])}

MATCHED KEYWORDS
================
{', '.join(ats_result['breakdown']['common_keywords'][:20])}
"""

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "📄 Download Report (TXT)",
                report_content,
                file_name=f"resume_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

        with col2:
            if st.session_state.get('optimized_resume'):
                st.download_button(
                    "📄 Download Optimized Resume",
                    st.session_state.optimized_resume,
                    file_name=f"optimized_resume_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

        with col3:
            if st.session_state.get('cover_letter'):
                st.download_button(
                    "✉️ Download Cover Letter",
                    st.session_state.cover_letter,
                    file_name=f"cover_letter_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    with tab3:
        st.subheader("🎯 One-Click Apply Readiness Check")

        # Comprehensive readiness checklist
        checklist_items = [
            ("Resume uploaded and analyzed", bool(st.session_state.get('resume_text'))),
            ("Job description analyzed", bool(st.session_state.get('jd_text'))),
            ("ATS score above 70%", ats_result['overall_score'] > 70),
            ("Job match above 60%", match_result['match_percentage'] > 60),
            ("Resume optimized", bool(st.session_state.get('optimized_resume'))),
            ("Cover letter generated", bool(st.session_state.get('cover_letter'))),
            ("Interview questions prepared", bool(st.session_state.get('interview_questions'))),
            ("Missing keywords addressed", len(ats_result['breakdown']['missing_keywords']) < 5)
        ]

        completed_items = sum(1 for _, status in checklist_items if status)
        completion_percentage = (completed_items / len(checklist_items)) * 100

        st.progress(completion_percentage / 100)
        st.markdown(f"**Readiness: {completion_percentage:.0f}% ({completed_items}/{len(checklist_items)} items complete)**")

        for item, status in checklist_items:
            if status:
                st.success(f"✅ {item}")
            else:
                st.error(f"❌ {item}")

        if completion_percentage >= 80:
            st.balloons()
            st.success("🎉 **You're ready to apply!** Your application package is well-prepared.")
        elif completion_percentage >= 60:
            st.warning("⚠️ **Almost ready!** Complete a few more items for best results.")
        else:
            st.error("❌ **More preparation needed.** Focus on the missing items above.")

def show_settings_page():
    """Display settings and preferences"""
    st.header("⚙️ Settings & Preferences")

    tab1, tab2, tab3 = st.tabs(["🎨 UI Preferences", "🔒 Privacy", "📊 Data Management"])

    with tab1:
        st.subheader("🎨 User Interface")

        col1, col2 = st.columns(2)

        with col1:
            # Theme settings
            st.markdown("**Theme Settings**")
            dark_mode = st.toggle("🌙 Dark Mode", value=True, key="settings_dark_mode")

            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "Blue", "Green", "Purple", "Orange"],
                key="settings_color_scheme"
            )

            # Font size
            font_size = st.selectbox(
                "Font Size",
                ["Small", "Medium", "Large"],
                index=1,
                key="settings_font_size"
            )

        with col2:
            # Layout preferences
            st.markdown("**Layout Preferences**")
            sidebar_expanded = st.toggle("📋 Sidebar Always Expanded", value=True, key="settings_sidebar_expanded")
            show_progress_bars = st.toggle("📊 Show Progress Bars", value=True, key="settings_show_progress_bars")
            show_tooltips = st.toggle("💡 Show Tooltips", value=True, key="settings_show_tooltips")

    with tab2:
        st.subheader("🔒 Privacy & Security")

        st.markdown("**Data Privacy Settings**")

        save_history = st.toggle("💾 Save Analysis History", value=False, key="settings_save_history")
        if save_history:
            st.info("📝 Analysis history will be saved locally for progress tracking.")
        else:
            st.warning("⚠️ Analysis history will not be saved.")

        auto_delete = st.toggle("🗑️ Auto-delete Data After Session", value=True, key="settings_auto_delete")
        if auto_delete:
            st.success("✅ Data will be automatically deleted when you close the app.")

        # Data consent
        st.markdown("**Data Usage Consent**")
        consent_analytics = st.checkbox("📊 Allow anonymous usage analytics", key="settings_consent_analytics")
        consent_improvements = st.checkbox("🔧 Allow data for service improvements", key="settings_consent_improvements")

        if st.button("🗑️ Clear All Data"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['authenticated', 'username']:
                    del st.session_state[key]
            st.success("✅ All data cleared successfully!")

    with tab3:
        st.subheader("📊 Data Management")

        # Export user data
        st.markdown("**Export Your Data**")

        if st.button("📥 Export All Data"):
            user_data = {
                'resume_versions': st.session_state.get('resume_versions', []),
                'job_descriptions': st.session_state.get('job_descriptions', []),
                'ats_history': st.session_state.get('ats_history', []),
                'settings': {
                    'dark_mode': dark_mode,
                    'color_scheme': color_scheme,
                    'font_size': font_size
                },
                'exported_at': datetime.now().isoformat()
            }

            st.download_button(
                "📥 Download Data Export",
                json.dumps(user_data, indent=2, default=str),
                file_name=f"resume_optimizer_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

        # Import data
        st.markdown("**Import Data**")
        uploaded_data = st.file_uploader("Upload data export file", type=['json'])

        if uploaded_data:
            try:
                data = json.load(uploaded_data)

                if st.button("📤 Import Data"):
                    # Import data to session state
                    if 'resume_versions' in data:
                        st.session_state.resume_versions = data['resume_versions']
                    if 'job_descriptions' in data:
                        st.session_state.job_descriptions = data['job_descriptions']
                    if 'ats_history' in data:
                        st.session_state.ats_history = data['ats_history']

                    st.success("✅ Data imported successfully!")
            except Exception as e:
                st.error(f"❌ Error importing data: {e}")

# --- Main Application ---
def main():
    """Main application function"""
    # Initialize database
    init_database()

    # Initialize API key from secrets or session state
    if 'api_key' not in st.session_state:
        try:
            # Try to get API key from Streamlit secrets (for cloud deployment)
            st.session_state.api_key = st.secrets["api_keys"]["OPENROUTER_API_KEY"]
        except (KeyError, FileNotFoundError):
            # Fallback to environment variable or empty string
            st.session_state.api_key = os.getenv("OPENROUTER_API_KEY", "")

    # Apply custom CSS
    apply_custom_css()

    # Create sidebar navigation
    create_sidebar_navigation()

    # Route to appropriate page
    current_page = st.session_state.get('current_page', 'Home')

    if current_page == "Home":
        show_home_page()
    elif current_page == "Upload":
        show_upload_page()
    elif current_page == "Analysis":
        show_analysis_page()
    elif current_page == "ResumeTools":
        show_resume_tools_page()
    elif current_page == "InterviewMastery":
        show_interview_mastery_page()
    elif current_page == "SkillsLearning":
        show_skills_learning_page()
    elif current_page == "Portfolio":
        show_portfolio_projects_page()
    elif current_page == "CareerCoach":
        show_career_coach_page()
    elif current_page == "Comparison":
        show_comparison_page()
    elif current_page == "Analytics":
        show_analytics_page()
    elif current_page == "Reports":
        show_reports_page()
    elif current_page == "Settings":
        show_settings_page()
    else:
        show_home_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 AI-Powered Career Platform</p>
            <p>Built with ❤️ using Streamlit by Vamshi | © 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
