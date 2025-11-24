import os
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from text_extractor import extract_data
from typing_extensions import TypedDict
import json

# -------------------------------------------------
# OpenRouter Client
# -------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-xxx",  
)

# -------------------------------------------------
# System Prompt (unchanged)
# -------------------------------------------------
CANDIDATE_PROMPT = open("candidate_json.txt", "r").read()
JD_PROMPT = open("jd_json.txt", "r").read()
GRADING_PROMPT = open("grading.txt", "r",  encoding="utf-8").read()

# -------------------------------------------------
# STATE
# -------------------------------------------------
class State(TypedDict):
    raw_extracted_text: str
    links_info: dict
    candidate_info_json: dict
    job_description: str
    job_info_json: dict
    match_score: dict
    experience_score: float
    education_score: float
    skill_match_score: dict
    final_match_score: float

# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def safe_get(d, key, default):
    return d[key] if key in d else default


def compute_semantic_score(list1, list2):
    if not list1 or not list2:
        return 0.0

    emb1 = model.encode(list1, convert_to_tensor=True)
    emb2 = model.encode(list2, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return float(sim.mean().item())

def normalize_degree(text: str) -> str:
    if not text:
        return "unknown"

    t = text.lower()
    if any(x in t for x in ["b.tech", "btech", "b tech", "b.e", "be", "bachelor"]):
        return "bachelor"
    if any(x in t for x in ["m.tech", "mtech", "m tech", "m.e", "me", "master", "msc", "ms"]):
        return "master"
    if any(x in t for x in ["phd", "ph.d", "doctor", "doctorate", "doctoral"]):
        return "phd"
    if "diploma" in t:
        return "diploma"

    return "unknown"

def compute_field_relevance(candidate_stream: str, jd_field_req: str) -> float:
    if not jd_field_req or jd_field_req.strip() == "":
        return 1.0

    cand = (candidate_stream or "").lower()
    jd = jd_field_req.lower()
    RELEVANT_FIELDS = [
        "computer", "cs", "cse", "it", "information technology",
        "ai", "ml", "data", "data science", "ds",
        "ece", "eee", "electronics", "csbs"
    ]
    if any(term in cand for term in RELEVANT_FIELDS):
        return 0.9

    return 0.5

def degree_match(candidate_degree, candidate_stream, jd_required):
    if not jd_required:
        return 1.0 

    cand_level = normalize_degree(candidate_degree)
    jd_level = normalize_degree(jd_required)
    if cand_level == jd_level:
        level_score = 1.0
    elif cand_level == "master" and jd_level == "bachelor":
        level_score = 1.0  # higher degree accepted
    elif cand_level == "bachelor" and jd_level == "master":
        level_score = 0.6
    elif cand_level == "diploma" and jd_level == "bachelor":
        level_score = 0.5
    else:
        level_score = 0.4

    field_score = compute_field_relevance(candidate_stream, jd_required)
    final = 0.7 * level_score + 0.3 * field_score
    return round(final, 3)


def experience_match(candidate_exp_years, jd_exp):
    if not jd_exp:
        return 1.0

    try:
        required = int("".join([c for c in jd_exp if c.isdigit()]))
    except:
        return 1.0

    if candidate_exp_years >= required:
        return 1.0
    elif candidate_exp_years >= required - 1:
        return 0.6
    return 0.2

def compute_experience_breakdown(candidate):
    internship_months = 0
    apprentice_months = 0
    industry_months = 0
    part_time_months = 0
    freelance_months = 0
    if isinstance(candidate, dict):
        experiences = candidate.get("experience", [])
    elif isinstance(candidate, list):
        experiences = candidate[0].get("experience", []) if candidate and isinstance(candidate[0], dict) else []
    else:
        experiences = []
    for exp in experiences:
        months = exp.get("duration_months", 0)
        exp_type = (exp.get("type", "") or "").lower()
        if exp_type == "internship":
            internship_months += months
        elif exp_type == "apprentice":
            apprentice_months += months
        elif exp_type == "full time":
            industry_months += months
        elif exp_type == "part time":
            part_time_months += months
        elif exp_type == "free lance":
            freelance_months += months
        else:
            apprentice_months += 0.5 * months
    internship_years = internship_months / 12
    apprentice_years = apprentice_months / 12
    industry_years = industry_months / 12
    part_time_years = part_time_months / 12
    freelance_years = freelance_months / 12
    INTERNSHIP_WEIGHT = 0.5
    APPRENTICE_WEIGHT = 0.6
    PART_TIME_WEIGHT = 0.4
    FREELANCE_WEIGHT = 0.7
    effective_years = (
          industry_years
        + INTERNSHIP_WEIGHT * internship_years
        + APPRENTICE_WEIGHT * apprentice_years
        + PART_TIME_WEIGHT * part_time_years
        + FREELANCE_WEIGHT * freelance_years
    )
    return {
        "internship_years": round(internship_years, 2),
        "apprentice_years": round(apprentice_years, 2),
        "industry_years": round(industry_years, 2),
        "part_time_years": round(part_time_years, 2),
        "freelance_years": round(freelance_years, 2),
        "effective_years": round(effective_years, 2)
    }

# -------------------------------------------------
# NODE: Candidate Data Extraction
# -------------------------------------------------
def candidate_info_extraction(state: State):

    raw_text = state["raw_extracted_text"]
    links = state["links_info"]
    combined_input = (
        "RAW RESUME TEXT:\n\n" + raw_text +
        "\n\nEXTRACTED LINKS JSON:\n" +
        json.dumps(links, indent=2)
    )
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": CANDIDATE_PROMPT},
            {"role": "user", "content": combined_input}
        ],
        response_format={"type": "json_object"}
    )
    assistant_msg = response.choices[0].message.content
    try:
        parsed = json.loads(assistant_msg)
    except:
        parsed = {
            "error": "Invalid JSON returned",
            "raw_output": assistant_msg
        }
    if isinstance(parsed, dict) and "contact" in parsed:
        profile = links.get("profile_info", {})
        parsed["contact"]["email"] = profile.get("mail", "")
        parsed["contact"]["phone"] = profile.get("contact", "")
        parsed["contact"]["linkedin"] = profile.get("linkedin", "")
        projects = links.get("projects", [])
        parsed["contact"]["github"] = projects[0] if len(projects) > 0 else ""
        parsed["contact"]["portfolio"] = projects[1] if len(projects) > 1 else ""
        parsed["contact"]["other_links"] = projects
    return {"candidate_info_json": parsed}

# -------------------------------------------------
# NODE: Job Description Data Extraction
# -------------------------------------------------
def job_desc_extraction(state: State):

    job_desc = state["job_description"]
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": JD_PROMPT},
            {"role": "user", "content": job_desc}
        ],
        response_format={"type": "json_object"}
    )
    assistant_msg = response.choices[0].message.content
    try:
        parsed = json.loads(assistant_msg)
    except:
        parsed = {
            "error": "Invalid JSON returned",
            "raw_output": assistant_msg
        }
    return {"job_info_json": parsed}

# ----------------------------------------------------
# NODE: Matching & Scoring
# ----------------------------------------------------
def candidate_job_matching(state: State):

    candidate = state["candidate_info_json"]
    jd = state["job_info_json"]

    # ------------------ EXPERIENCE ------------------
    exp_data = compute_experience_breakdown(candidate)
    cand_exp_years = exp_data["effective_years"]

    exp_score = experience_match(
        cand_exp_years,
        jd.get("experience_required", "")
    )

    # ------------------ EDUCATION (IMPROVED) ------------------
    if candidate.get("education"):
        degree = candidate["education"][0].get("degree", "")
        stream = candidate["education"][0].get("stream", "")
    else:
        degree = ""
        stream = ""

    edu_score = degree_match(
        candidate_degree=degree,
        candidate_stream=stream,
        jd_required=jd.get("education_required", "")
    )

    # ------------------ TECH SKILLS (LLM-BASED) ------------------
    cand = candidate
    jd_obj = jd
    skills_obj = cand.get("skills", {})
    technical = skills_obj.get("technical", []) or []
    tools = skills_obj.get("tools", []) or []
    payload = {
        "jd_required_skills": jd_obj.get("skills_required", []),
        "jd_optional_skills": jd_obj.get("skills_optional", []),
        "jd_tools": jd_obj.get("tools_and_technologies", []),
        "jd_responsibilities": jd_obj.get("responsibilities", []),

        "candidate_skills": technical + tools,
        "candidate_tools": tools,

        "candidate_projects": cand.get("projects", []),
        "candidate_experience": cand.get("experience", []),
        "candidate_certifications": cand.get("certifications", [])
    }
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ],
        response_format={"type": "json_object"}
    )
    skill_match = json.loads(response.choices[0].message.content)
    if isinstance(skill_match, list):
        skill_match = skill_match[0] if skill_match else {}
    
    skill_score = skill_match.get("final_skill_match_score", 0)

    # ------------------ FINAL SCORE ------------------
    final_score = (
        0.20 * exp_score +
        0.10 * edu_score +
        0.70 * skill_score
    )
    return {
        "experience_score": exp_score,
        "education_score": edu_score,
        "skill_match_score": skill_match,
        "final_match_score": final_score
    }

# -------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------
graph = StateGraph(State)
graph.add_node("candidate_json", candidate_info_extraction)
graph.add_node("jd_json", job_desc_extraction)
graph.add_node("match_score", candidate_job_matching)
graph.add_edge(START, "candidate_json")
graph.add_edge("candidate_json", "jd_json")
graph.add_edge("jd_json", "match_score")
graph.add_edge("match_score", END)
app = graph.compile()

# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
raw_text, links_info = extract_data("PraveenRaj_CreativeDesigner.pdf")
job_desc = open("jd.txt", "r").read()
jd_json = open("jd_json.txt", "r").read()
initial_state = {
    "raw_extracted_text": raw_text,
    "links_info": links_info,
    "job_description": job_desc
}
result = app.invoke(initial_state)
print(json.dumps(result["candidate_info_json"], indent=2))
print(json.dumps(result["job_info_json"], indent=2))
print("Experience Score:", result["experience_score"])
print("Education Score:", result["education_score"])
print("Skill Match Score:", json.dumps(result["skill_match_score"], indent=2))
print("Final Match Score:", result["final_match_score"])
