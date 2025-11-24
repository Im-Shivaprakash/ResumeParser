# ATS Resume Parser & Scorer

Lightweight ATS-style resume parser and scorer that extracts candidate data from resumes, converts job descriptions to structured JSON, and computes a match score between candidate and job.

Key files
- [ResumeParser.py](ResumeParser.py) — main pipeline and scoring logic. Key symbols: [`candidate_info_extraction`](ResumeParser.py), [`job_desc_extraction`](ResumeParser.py), [`candidate_job_matching`](ResumeParser.py), [`compute_experience_breakdown`](ResumeParser.py), [`degree_match`](ResumeParser.py), [`compute_semantic_score`](ResumeParser.py), [`State`](ResumeParser.py), [`graph`](ResumeParser.py), [`app`](ResumeParser.py).
- [text_extractor.py](text_extractor.py) — resume text & links extraction. Key symbol: [`extract_data`](text_extractor.py).
- [candidate_json.txt](candidate_json.txt), [jd_json.txt](jd_json.txt), [grading.txt](grading.txt) — system prompts / grading templates used by the LLM.
- [jd.txt](jd.txt) — raw job description input.
- [requirements.txt](requirements.txt) — Python dependencies.
- [test2.txt](test2.txt) — sample run output (includes parsed candidate JSON and scoring breakdown).

Overview / Pipeline
1. Extract raw text and links from a resume using [`extract_data`](text_extractor.py).
2. Infer structured candidate JSON via LLM in [`candidate_info_extraction`](ResumeParser.py).
3. Convert job description into structured JSON via LLM in [`job_desc_extraction`](ResumeParser.py).
4. Score match in [`candidate_job_matching`](ResumeParser.py) using:
   - experience (via [`compute_experience_breakdown`](ResumeParser.py) + `experience_match`),
   - education (via [`degree_match`](ResumeParser.py)),
   - skills (LLM-based grading using `grading.txt` prompt and semantic helpers such as [`compute_semantic_score`](ResumeParser.py)).
5. The graph flow is constructed with `langgraph` in [ResumeParser.py](ResumeParser.py) and executed via the compiled [`app`](ResumeParser.py).

Scoring formula
The final score is a weighted combination of experience, education and skill-match.

Inline:
- $FinalScore = 0.20 \times Exp + 0.10 \times Edu + 0.70 \times Skill$

Block:
$$
FinalScore = 0.20 \cdot Exp + 0.10 \cdot Edu + 0.70 \cdot Skill
$$

Where:
- Exp is computed from [`compute_experience_breakdown`](ResumeParser.py) and normalized by `experience_match`.
- Edu is computed by [`degree_match`](ResumeParser.py] (degree level + field relevance).
- Skill is produced by the LLM grading prompt in [grading.txt](grading.txt) and returned as `final_skill_match_score`.

Quick setup
1. Create a virtual environment and install deps:
```bash
# use python3 -m venv venv && source venv/bin/activate (or equivalent on Windows)
pip install -r [requirements.txt](http://_vscodecontentref_/0)