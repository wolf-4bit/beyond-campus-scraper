from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CategoryDef:
    """Definition for a scraping category."""
    name: str
    classify_hint: str  # short description for URL classification
    structure_prompt: str  # detailed instructions for LLM structuring
    max_urls: int = 1000  # safety ceiling only; the LLM decides relevance below this


CATEGORY_REGISTRY: dict[str, CategoryDef] = {
    "courses": CategoryDef(
        name="courses",
        classify_hint="academic programs, degrees, course catalogs, department pages, majors/minors, curriculum, program requirements, admission requirements for programs, tuition and fees pages, international student requirements, transfer credit policies, graduate program prerequisites, GPA and test score requirements, application deadlines, academic departments, schools and colleges overview pages",
        structure_prompt="""\
Create a comprehensive academic programs reference. For EACH program/course:

## Structure per program:
- Program name and degree type (BS, BA, MS, PhD, Certificate, Minor)
- Department / School it belongs to
- Program description
- **Admission requirements broken down by student type:**
  - Domestic students: GPA, prerequisite courses, test scores, application materials
  - International students: GPA, English proficiency (TOEFL/IELTS scores), credential evaluation, visa requirements, additional documents
  - Transfer students: transfer credit policies, minimum credits, articulation agreements
- Credit hours / duration
- Core courses and electives (list course names/codes if available)
- Concentration or specialization tracks
- Career outcomes / job placement data if available
- Tuition or program-specific fees if mentioned
- Application deadlines per intake (Fall, Spring, Summer)
- Contact info for the department

## Organization:
- Group by School/College first (e.g., School of Engineering, College of Arts & Sciences)
- Then by degree level (Undergraduate, Graduate, Doctoral)
- Then alphabetically by program name

Preserve ALL specific numbers, dates, GPA requirements, test score ranges, and URLs.""",
    ),
    "scholarships": CategoryDef(
        name="scholarships",
        classify_hint="scholarships, financial aid, grants, tuition, fees, merit awards, need-based aid",
        structure_prompt="""\
Create a comprehensive financial aid reference. For each scholarship/aid program include:
- Name, amount (or range), eligibility criteria, deadline, renewal criteria, and how to apply.
- Separate merit-based vs need-based vs program-specific scholarships.
- Include tuition rates if available (domestic vs international).
- Include FAFSA/CSS Profile requirements and codes.
Preserve ALL specific dollar amounts, GPA thresholds, dates, and URLs.""",
    ),
    "staff": CategoryDef(
        name="staff",
        classify_hint="faculty directories, staff pages, department contacts, professor profiles",
        structure_prompt="""\
Create a faculty/staff directory. Organize by department. For each person include:
- Name, title, department, email, phone, office location
- Research interests / specializations if available
- Education background if listed
Preserve ALL contact details and URLs.""",
    ),
    "admissions": CategoryDef(
        name="admissions",
        classify_hint="admission requirements, how to apply, application deadlines, transfer info, international admissions",
        structure_prompt="""\
Create a comprehensive admissions guide. Cover separately for each student type:
- **Freshman domestic**: requirements, GPA, test scores, deadlines, documents
- **Freshman international**: English proficiency, credential evaluation, visa info, additional requirements
- **Transfer domestic**: credit requirements, GPA, articulation
- **Transfer international**: same as transfer + international requirements
- **Graduate**: program-specific requirements, GRE/GMAT, letters of recommendation
- Application process step-by-step
- Key deadlines (Early Decision, Early Action, Regular Decision, Rolling)
- Required documents checklist
- Decision notification timelines
Preserve ALL specific dates, scores, GPA thresholds, and URLs.""",
    ),
    "campus_life": CategoryDef(
        name="campus_life",
        classify_hint="student life, housing, dining, clubs, organizations, athletics, facilities, campus services",
        structure_prompt="""\
Create a campus life reference covering:
- Housing options with costs, room types, meal plan requirements
- Dining locations and meal plan options with pricing
- Student organizations and clubs
- Athletics and recreation facilities
- Health and wellness services
- Transportation and parking
- Campus safety
Preserve ALL specific costs, names, and URLs.""",
    ),
}

DEFAULT_CATEGORIES = list(CATEGORY_REGISTRY.keys())
