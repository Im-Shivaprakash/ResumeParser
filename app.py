from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
import tempfile
from pathlib import Path

# Import your existing modules
from text_extractor import extract_data
from ResumeParser import (
    State, client, CANDIDATE_PROMPT, JD_PROMPT, GRADING_PROMPT,
    candidate_info_extraction, job_desc_extraction, candidate_job_matching,
    graph, app as resume_app
)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Parser API",
    description="API for parsing resumes and matching them with job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class JobDescription(BaseModel):
    job_description: str

class ResumeParseRequest(BaseModel):
    job_description: str

class MatchResult(BaseModel):
    candidate_info: Dict[str, Any]
    job_info: Dict[str, Any]
    experience_score: float
    education_score: float
    skill_match_score: Dict[str, Any]
    final_match_score: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Resume Parser API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Resume Parser API is operational"}

# Main resume parsing endpoint
@app.post("/parse-resume", response_model=MatchResult)
async def parse_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Parse a resume file and match it against a job description.
    
    - **resume_file**: PDF file containing the resume
    - **job_description**: Text description of the job requirements
    
    Returns detailed matching analysis including scores.
    """
    try:
        # Validate file type
        if not resume_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await resume_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract data from resume
            raw_text, links_info = extract_data(temp_file_path)
            
            # Prepare initial state
            initial_state = {
                "raw_extracted_text": raw_text,
                "links_info": links_info,
                "job_description": job_description
            }
            
            # Process through the graph
            result = resume_app.invoke(initial_state)
            
            # Return structured response
            return MatchResult(
                candidate_info=result["candidate_info_json"],
                job_info=result["job_info_json"],
                experience_score=result.get("experience_score", 0.0),
                education_score=result.get("education_score", 0.0),
                skill_match_score=result.get("skill_match_score", {}),
                final_match_score=result.get("final_match_score", 0.0)
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

# Endpoint to parse job description only
@app.post("/parse-job-description")
async def parse_job_description(job_desc: JobDescription):
    """
    Parse a job description and extract structured information.
    
    - **job_description**: Text description of the job
    
    Returns structured job information.
    """
    try:
        # Create a minimal state with just job description
        state = {
            "job_description": job_desc.job_description
        }
        
        # Extract job info
        job_info = job_desc_extraction(state)
        
        return {
            "job_info": job_info["job_info_json"],
            "message": "Job description parsed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing job description: {str(e)}")

# Endpoint to extract resume text only
@app.post("/extract-resume-text")
async def extract_resume_text(resume_file: UploadFile = File(...)):
    """
    Extract text content from a resume file without processing.
    
    - **resume_file**: PDF file containing the resume
    
    Returns raw extracted text and links.
    """
    try:
        # Validate file type
        if not resume_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await resume_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract data from resume
            raw_text, links_info = extract_data(temp_file_path)
            
            return {
                "raw_text": raw_text,
                "links_info": links_info,
                "message": "Text extracted successfully"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

# Batch processing endpoint
@app.post("/batch-parse-resumes")
async def batch_parse_resumes(
    resume_files: List[UploadFile] = File(...),
    job_description: str = Form(...)
):
    """
    Parse multiple resume files against a single job description.
    
    - **resume_files**: List of PDF files containing resumes
    - **job_description**: Text description of the job requirements
    
    Returns list of matching analyses.
    """
    try:
        results = []
        
        for resume_file in resume_files:
            try:
                # Validate file type
                if not resume_file.filename.endswith('.pdf'):
                    results.append({
                        "filename": resume_file.filename,
                        "error": "Only PDF files are supported",
                        "success": False
                    })
                    continue
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = await resume_file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Extract data from resume
                    raw_text, links_info = extract_data(temp_file_path)
                    
                    # Prepare initial state
                    initial_state = {
                        "raw_extracted_text": raw_text,
                        "links_info": links_info,
                        "job_description": job_description
                    }
                    
                    # Process through the graph
                    result = resume_app.invoke(initial_state)
                    
                    # Add result
                    results.append({
                        "filename": resume_file.filename,
                        "success": True,
                        "candidate_info": result["candidate_info_json"],
                        "job_info": result["job_info_json"],
                        "experience_score": result.get("experience_score", 0.0),
                        "education_score": result.get("education_score", 0.0),
                        "skill_match_score": result.get("skill_match_score", {}),
                        "final_match_score": result.get("final_match_score", 0.0)
                    })
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as e:
                results.append({
                    "filename": resume_file.filename,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "results": results,
            "total_processed": len(resume_files),
            "successful": len([r for r in results if r.get("success", False)]),
            "failed": len([r for r in results if not r.get("success", False)])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")

# Get API documentation
@app.get("/api-info")
async def get_api_info():
    """Get information about available API endpoints."""
    return {
        "endpoints": {
            "/": "Root endpoint - API status",
            "/health": "Health check endpoint",
            "/parse-resume": "Main endpoint to parse resume and match with job description",
            "/parse-job-description": "Parse job description only",
            "/extract-resume-text": "Extract text from resume without processing",
            "/batch-parse-resumes": "Batch process multiple resumes",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "supported_formats": ["PDF"],
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)