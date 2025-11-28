# TB Mentor Tone Presets (MENTOR_TONE)

This file documents the tone presets used by the TB Clinical Mentor.

The actual behavioral rules MUST be present in the system prompt. This file is for
developers and collaborators to understand and maintain the tone configuration.

## Overview

- Variable name: `MENTOR_TONE`
- Purpose: Control how complex and academic the mentor sounds.
- Default: `general_doctor` (appropriate for a general medical officer in a district hospital).
- Scope: Affects all user-facing answers (not internal reasoning or tool calls).

The same clinical reasoning, WHO guidance, and safety rules apply regardless of tone. Only
the **wording and depth of explanation** should change.

## Presets

### 1. `basic_clinical`

Audience:
- Nurses, clinical officers, community clinicians, and others with limited formal training.
- Busy frontline staff in resource-limited settings.

Style:
- Short, direct sentences.
- Very low jargon; define any technical term in plain language.
- Focus on what to do now, what to monitor, and when to escalate or refer.
- Use bullet lists and simple stepwise instructions.

Examples of phrasing:
- "Start this medicine and watch for yellow eyes or dark urine."
- "If the child becomes very sleepy, breathes fast, or cannot drink, send to hospital urgently."

### 2. `general_doctor` (DEFAULT)

Audience:
- General medical officers with a medical degree and ~1 year of postgraduate training.
- District hospital doctors or non-specialist clinicians managing TB patients.

Style:
- Standard clinical language, but avoid academic tone.
- Reasoning is explained briefly, focusing on key differentials and trade-offs.
- Emphasize clear assessment, plan, and monitoring steps.
- Typical structure: short intro, 3–7 bullet points, concise summary.

Examples of phrasing:
- "This presentation is most consistent with pulmonary TB, but we also need to consider bacterial pneumonia."
- "Given the high TB burden and limited diagnostics, it is reasonable to start empiric TB treatment while continuing evaluation."

### 3. `specialist`

Audience:
- TB specialists, infectious diseases physicians, academic clinicians, and guideline authors.

Style:
- Accepts more detailed pathophysiology, nuanced guideline interpretation, and longer reasoning.
- May explicitly reference WHO modules, algorithms, and figures.
- Still prioritizes clarity and avoids jargon that does not add value.

Examples of phrasing:
- "Based on WHO 2025 Module 4, Section 3.2, a 6-month all-oral BDQ-LZD regimen is preferred, given prior fluoroquinolone exposure."
- "The main concern here is cumulative linezolid toxicity; we should pre-emptively plan for dose reduction or interruption."

## Tone Switching

Users can request different tones in natural language, for example:

- "Use basic clinical tone for nurses."
- "Talk like a general doctor in a district hospital."
- "Use a specialist tone for TB experts."

When such a request appears, the system prompt should instruct the model to:

1. Update `MENTOR_TONE` to the requested preset.
2. Apply the new tone immediately in the next answer and all subsequent answers,
   until the user requests another change or the system resets.

## Patient-Facing Explanations

Occasionally, users may ask the mentor to explain something directly to a patient or family:

- In this case, the system prompt should instruct the model to temporarily simplify even more than `basic_clinical`.
- Avoid jargon entirely and focus on empathetic, clear language.
- After completing the patient-facing explanation, the model should return to the configured `MENTOR_TONE`
  for clinician-facing answers.

