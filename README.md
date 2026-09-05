# SIH_ML — AYUSH Clinical Assessment & Health Decision-Support Engine

**SIH_ML** is the core Machine Learning and Artificial Intelligence backend module for an AYUSH-based clinical assessment and health decision-support system. It processes structured questionnaire responses, unstructured clinical text, and transcribed audio input into validated evidence, structured patient cases, 10-parameter AYUSH assessments, explainability reports, evidence-backed recommendations, and longitudinal progress tracking.

---

## Key Features & Architecture

1. **AYUSH 10-Parameter Assessment Engine** (`src/ayush_assessment.py`, `src/ayush_processor.py`)
   - Evaluates patient constitution and status across **exactly 10 active parameters**:
     - `prakriti`, `vikriti`, `sara`, `samhanana`, `pramana`, `satmya`, `sattva`, `aharaShakti`, `vyayamaShakti`, `vaya`.
   - **Agni & Koshtha Exclusion**: Strictly excluded from active assessment, scoring, and recommendation formulas.

2. **Unstructured Clinical Extraction & LLM Fallback** (`src/clinical_extractor.py`, `src/llm_extractor.py`)
   - Natural language processing for clinical notes, doctor-patient dialogues, and audio transcripts.
   - Extracts demographics, chief complaints, HPI, PMH, medications, allergies, family/personal history, ROS, and AYUSH-relevant evidence.
   - `LLMClinicalExtractor` provides API wrapper support with seamless deterministic rule-based fallback when offline.

3. **Multi-Source Patient Case Builder** (`src/patient_case_builder.py`)
   - Merges questionnaire responses and clinical text/audio extraction into a unified schema ([`schemas/patient_case.json`](file:///c:/Users/avina/OneDrive/Desktop/SIH_ML/ml/schemas/patient_case.json)).
   - Preserves source provenance (`questionnaire`, `text`, `audio_transcript`, `prescription`) and confidence scores for every finding.

4. **AYUSH Explainability Engine** (`src/ayush_explainer.py`)
   - Generates transparent, human-readable rationales linking every assessment parameter to specific patient evidence and mapping rules.

5. **Clinical Interpretation & Recommendation Engine** (`src/recommendation_engine.py`, `data/ayush_recommendation_rules.json`)
   - Data-driven recommendation layer producing structured JSON for dietary, lifestyle, herbal/supportive, and exercise guidance.
   - Embeds exact evidence references (`questionId`, `answer`, `source`, `confidence`) and rationales for every suggestion.
   - Implements safety cautions and non-diagnostic legal disclaimers.

6. **Longitudinal Progress Tracking & Risk Stratification** (`src/longitudinal_risk_engine.py`)
   - Computes quantitative **AYUSH Health Resilience Score (0–100)** and **Imbalance Severity Score (0–100)**.
   - Categorizes patients into risk tiers (`Low`, `Moderate`, `High`, `Critical`).
   - Tracks longitudinal progress across sequential visits, calculates Vikriti resolution rates (%), and tracks a 10-parameter evolution matrix.

7. **REST API & Python SDK Integration** (`src/api.py`, `src/__init__.py`)
   - Zero-dependency HTTP REST API (`http.server` on port 8000):
     - `GET /health`
     - `POST /process-case`
     - `POST /extract`
     - `POST /recommendations`
     - `POST /risk-stratification`
     - `POST /longitudinal-track`
   - Clean Python SDK library exports for seamless integration into larger backend applications.

---

## Directory Structure

```text
SIH_ML/
├── data/                             # Data-driven AYUSH parameter mappings & rules
│   ├── ayush_questions.json
│   ├── ayush_recommendation_rules.json
│   ├── prakriti_mapping.json
│   ├── sara_mapping.json
│   ├── sattva_mapping.json
│   └── ... (mappings for 10 active parameters)
├── models/                           # Model checkpoints & offline NLP assets
├── output/                           # Demonstration output artifacts
│   ├── ayush_explainability_report.json
│   ├── clinical_recommendation_output.json
│   ├── longitudinal_risk_report.json
│   └── patient_case_output.json
├── prompts/                          # LLM extraction prompt configurations
│   └── clinical_extraction_prompt.json
├── schemas/                          # Structured JSON Schemas
│   └── patient_case.json
├── src/                              # Core Python source modules
│   ├── __init__.py                   # SDK exports
│   ├── api.py                        # REST API Server
│   ├── ayush_assessment.py           # 10-Parameter Assessment Engine
│   ├── ayush_explainer.py            # Explainability Engine
│   ├── ayush_processor.py            # Answer Validation & Evidence Mapping
│   ├── clinical_extractor.py         # Clinical Entity Extractor
│   ├── llm_extractor.py              # LLM Wrapper with Fallback
│   ├── longitudinal_risk_engine.py   # Health Resilience & Progress Tracking Engine
│   ├── main.py                       # CLI Pipeline Runner
│   ├── patient_case_builder.py       # Patient Case Builder
│   └── recommendation_engine.py     # Recommendation & Interpretation Engine
└── tests/                            # Automated Unit & Integration Tests
    ├── sample_patient_cases.json
    ├── test_longitudinal_risk.py
    ├── test_pipeline.py
    └── test_recommendations.py
```

---

## Getting Started & Running

### Requirements
- Python 3.9+ (Built using standard library; 0 mandatory pip dependencies for basic execution and API server).

### Running Automated Tests

Run the full test suite:
```bash
python tests/test_pipeline.py
python tests/test_recommendations.py
python tests/test_longitudinal_risk.py
```

### Running the End-to-End Pipeline CLI

Run default pipeline with sample data:
```bash
python src/main.py
```

Process custom clinical text or transcript:
```bash
python src/main.py --text "Patient reports severe headache and fatigue for 3 days." --source "audio_transcript"
```

### Running the REST API Server

Start the REST API server on port 8000:
```bash
python src/api.py
```

Health check request:
```bash
curl http://localhost:8000/health
```

---

## Clinical & Safety Disclaimer

This system is designed strictly for educational, research, and clinical decision-support purposes. All generated interpretations, recommendations, and risk scores are decision-support outputs and do **NOT** constitute a medical diagnosis, prescription, or direct therapeutic directive. Consult a certified Ayurvedic practitioner or medical professional for clinical diagnosis and treatment planning.
