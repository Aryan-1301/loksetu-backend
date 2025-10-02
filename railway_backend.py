# Railway-ready LokSetu Backend
# This is optimized for Railway.app deployment

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import google.generativeai as genai
import speech_recognition as sr
import io
import base64
import json
import asyncio
from datetime import datetime, timedelta
import os
from typing import List, Optional, Dict, Any
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LokSetu AI Assistant API", version="2.0.0")

# Enable CORS for Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
API_SETU_BASE_URL = "https://apisetu.gov.in/dic/myscheme/srv/v3/public"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAgsWiHV-ao8k4zoJYDui0rFE5BybDYBr8")

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic models
class UserQuery(BaseModel):
    question: str
    language: str = "hi"
    user_profile: Optional[Dict[str, Any]] = None

class SchemeResponse(BaseModel):
    scheme_id: str
    title: str
    description: str
    ministry: str
    category: List[str]
    official_link: str

class AIResponse(BaseModel):
    answer: str
    audio_response: Optional[str] = None
    relevant_schemes: List[SchemeResponse] = []
    sources: List[str] = []
    confidence: float = 0.0

class TTSRequest(BaseModel):
    text: str
    language: str = "hi"

# Cache for scheme data
schemes_cache = {}
cache_expiry = {}

class LokSetuAIAssistant:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("LokSetu AI Assistant initialized")

    async def fetch_schemes_from_api_setu(self, language: str = "hi") -> List[Dict]:
        """Fetch latest schemes from API Setu MyScheme API"""
        cache_key = f"schemes_{language}"
        if (cache_key in schemes_cache and
            cache_key in cache_expiry and
            datetime.now() < cache_expiry[cache_key]):
            return schemes_cache[cache_key]

        try:
            url = f"{API_SETU_BASE_URL}/schemes?lang={language}"
            headers = {"accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            schemes = data.get("data", [])

            # Cache the results for 1 hour
            schemes_cache[cache_key] = schemes
            cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)

            logger.info(f"Fetched {len(schemes)} schemes for language {language}")
            return schemes
        except requests.RequestException as e:
            logger.error(f"Error fetching schemes: {e}")
            return []

    def search_relevant_schemes(self, query: str, schemes: List[Dict], top_k: int = 3) -> List[Dict]:
        """Search for schemes relevant to user query"""
        query_lower = query.lower()
        relevant_schemes = []

        for scheme in schemes:
            score = 0
            title = scheme.get("title", "").lower()
            description = scheme.get("description", "").lower()
            categories = " ".join(scheme.get("category", [])).lower()

            # Enhanced scoring
            if query_lower in title:
                score += 10
            if query_lower in description:
                score += 5
            if any(word in categories for word in query_lower.split()):
                score += 3

            # Check for related keywords
            query_words = query_lower.split()
            for word in query_words:
                if word in title or word in description:
                    score += 2

            if score > 0:
                scheme["relevance_score"] = score
                relevant_schemes.append(scheme)

        # Sort by relevance score and return top k
        relevant_schemes.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_schemes[:top_k]

    async def generate_ai_response_with_gemini(self, query: str, relevant_schemes: List[Dict], language: str) -> str:
        """Generate AI-powered response using Gemini AI"""
        # Prepare context with scheme information
        context = "भारत सरकार की योजनाएं:\n\n" if language == "hi" else "Indian Government Schemes:\n\n"

        for scheme in relevant_schemes:
            if language == "hi":
                context += f"योजना: {scheme.get('title')}\n"
                context += f"विवरण: {scheme.get('description', 'N/A')}\n"
                context += f"मंत्रालय: {scheme.get('ministry', 'N/A')}\n\n"
            else:
                context += f"Scheme: {scheme.get('title')}\n"
                context += f"Description: {scheme.get('description', 'N/A')}\n"
                context += f"Ministry: {scheme.get('ministry', 'N/A')}\n\n"

        # Create prompt for Gemini
        if language == "hi":
            prompt = f"""आप एक विशेषज्ञ सरकारी योजना सलाहकार हैं। आप LokSetu AI हैं।

उपयोगकर्ता का प्रश्न: "{query}"

निम्नलिखित योजनाओं के आधार पर विस्तृत उत्तर दें:

{context}

कृपया:
1. सबसे प्रासंगिक योजनाओं के बारे में बताएं
2. पात्रता मानदंड स्पष्ट करें
3. आवेदन प्रक्रिया बताएं
4. सरल हिंदी में उत्तर दें

उत्तर हिंदी में दें।"""
        else:
            prompt = f"""You are an expert government scheme advisor. You are LokSetu AI.

User question: "{query}"

Based on the following scheme information, provide a detailed answer:

{context}

Please:
1. Explain the most relevant schemes
2. Mention eligibility criteria
3. Describe application process
4. Answer in simple English

Respond in English."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            fallback_msg = "क्षमा करें, मैं इस समय उत्तर नहीं दे सकता।" if language == "hi" else "Sorry, I cannot provide an answer right now."
            return fallback_msg

    async def speech_to_text(self, audio_data: bytes, language: str = "hi") -> str:
        """Convert speech to text using speech recognition"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = r.record(source)

            lang_code = "hi-IN" if language == "hi" else "en-IN"
            text = r.recognize_google(audio, language=lang_code)

            os.unlink(temp_audio_path)
            return text
        except Exception as e:
            logger.error(f"Error in speech to text: {e}")
            return ""

    async def text_to_speech(self, text: str, language: str = "hi") -> bytes:
        """Convert text to speech using gTTS"""
        try:
            import gtts

            lang_code = "hi" if language == "hi" else "en"
            tts = gtts.gTTS(text=text, lang=lang_code, slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                tts.save(temp_audio.name)
                temp_audio_path = temp_audio.name

            with open(temp_audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            os.unlink(temp_audio_path)
            return audio_bytes
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return b""

# Initialize AI assistant
ai_assistant = LokSetuAIAssistant()

@app.get("/")
async def root():
    return {
        "message": "LokSetu AI Assistant API is running on Railway!",
        "version": "2.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "LokSetu AI Backend",
        "platform": "Railway"
    }

@app.get("/schemes")
async def get_all_schemes(language: str = "hi"):
    """Get all government schemes from API Setu"""
    schemes = await ai_assistant.fetch_schemes_from_api_setu(language)
    if not schemes:
        raise HTTPException(status_code=500, detail="Unable to fetch schemes")
    return {"schemes": schemes, "count": len(schemes)}

@app.post("/ask")
async def ask_ai_assistant(query: UserQuery) -> AIResponse:
    """Main endpoint for AI-powered query answering using Gemini"""
    try:
        logger.info(f"Received query: {query.question[:50]}... in {query.language}")

        # Fetch latest schemes
        schemes = await ai_assistant.fetch_schemes_from_api_setu(query.language)

        if not schemes:
            return AIResponse(
                answer="क्षमा करें, योजनाओं की जानकारी उपलब्ध नहीं है।" if query.language == "hi" else "Sorry, scheme information is not available.",
                relevant_schemes=[],
                sources=[],
                confidence=0.0
            )

        # Find relevant schemes
        relevant_schemes = ai_assistant.search_relevant_schemes(query.question, schemes)

        # Generate AI response using Gemini
        ai_answer = await ai_assistant.generate_ai_response_with_gemini(
            query.question, relevant_schemes, query.language
        )

        # Format response
        scheme_responses = []
        sources = []

        for scheme in relevant_schemes:
            scheme_response = SchemeResponse(
                scheme_id=scheme.get("schemeId", ""),
                title=scheme.get("title", ""),
                description=scheme.get("description", ""),
                ministry=scheme.get("ministry", ""),
                category=scheme.get("category", []),
                official_link=scheme.get("officialLink", "")
            )
            scheme_responses.append(scheme_response)
            if scheme.get("officialLink"):
                sources.append(scheme.get("officialLink"))

        logger.info(f"Generated response with {len(scheme_responses)} relevant schemes")

        return AIResponse(
            answer=ai_answer,
            relevant_schemes=scheme_responses,
            sources=sources,
            confidence=0.85 if relevant_schemes else 0.3
        )

    except Exception as e:
        logger.error(f"Error in ask_ai_assistant: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ask-voice")
async def ask_voice_assistant(
    audio_file: UploadFile = File(...),
    language: str = Form("hi")
) -> AIResponse:
    """Voice-enabled AI assistant endpoint"""
    try:
        logger.info(f"Received voice query in {language}")

        # Read audio file
        audio_data = await audio_file.read()

        # Convert speech to text
        text_query = await ai_assistant.speech_to_text(audio_data, language)

        if not text_query:
            return AIResponse(
                answer="क्षमा करें, आवाज़ समझ नहीं आई।" if language == "hi" else "Sorry, couldn't understand the voice.",
                audio_response=None,
                relevant_schemes=[],
                sources=[],
                confidence=0.0
            )

        # Process the text query
        query = UserQuery(question=text_query, language=language)
        response = await ask_ai_assistant(query)

        # Generate audio response
        audio_bytes = await ai_assistant.text_to_speech(response.answer, language)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        response.audio_response = audio_base64

        return response

    except Exception as e:
        logger.error(f"Error in ask_voice_assistant: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/text-to-speech")
async def convert_text_to_speech(request: TTSRequest):
    """Convert text to speech endpoint"""
    try:
        audio_bytes = await ai_assistant.text_to_speech(request.text, request.language)

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )

    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
async def search_schemes(query: str, language: str = "hi", limit: int = 5):
    """Search schemes by keyword"""
    schemes = await ai_assistant.fetch_schemes_from_api_setu(language)
    relevant_schemes = ai_assistant.search_relevant_schemes(query, schemes, limit)
    return {"query": query, "schemes": relevant_schemes, "count": len(relevant_schemes)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
