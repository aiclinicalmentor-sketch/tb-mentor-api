import fs from "fs";
import path from "path";
import OpenAI from "openai";

/**
 * TB Mentor API route
 *
 * Drop this file into your Next.js `pages/api` folder as `tb-mentor.js`.
 * Then POST { "message": "your case description" } to /api/tb-mentor.
 *
 * You MUST set OPENAI_API_KEY in your environment.
 * You ALSO need your existing /api/tb-rag-query and /api/tb_peds_tda routes working.
 */

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// vvv System prompt comes from my last custom GPT instructions
const SYSTEM_PROMPT = fs.readFileSync(
  path.join(process.cwd(), "prompts/mentor-system.txt"),
  "utf8"
);

// Tools derived from your tb-openapi-schema.yaml (RAG) and tb_peds_tda.js (TDA)
const tools = [
  {
    type: "function",
    function: {
      name: "fetchRelevantTbGuidance",
      description: `
Embed a clinician TB question and return the top-matching WHO TB guideline passages from the TB RAG store.

You MUST use this tool before giving any case-specific TB explanation, reasoning, or regimen/treatment recommendation,
unless the user is explicitly asking a purely factual/non-clinical question about TB (e.g., epidemiology, biology).

For real patients, always construct a CASE SUMMARY, TASK, and RAG FOCUS in your message,
then call this tool with a compact, embedding-friendly question string that encodes patient, problem, task, and context.

Use this before giving any case-specific TB explanation or reasoning.
`,
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["question"],
        properties: {
          question: {
            type: "string",
            description:
              "Free-form clinician question about a TB patient or scenario. " +
              "Include key details (age, HIV status, drug-resistance, comorbidities, prior treatment, and resource setting) " +
              "to retrieve relevant guidance.",
            minLength: 1,
            examples: [
              "How should I adjust the regimen for a TB/HIV co-infected patient with renal impairment?"
            ]
          },
          top_k: {
            type: "integer",
            description:
              "Desired number of passages to return. Defaults to 8 and supports between 1 and 8 results.",
            minimum: 1,
            maximum: 8,
            default: 8,
            examples: [8]
          },
          include_table_rows: {
            type: "boolean",
            description:
              "When true, include raw table_rows in results for table chunks. " +
              "Defaults to false in the API schema to keep responses small, " +
              "but your mentor can set this to true when it needs detailed table content.",
            default: false
          },
          table_row_limit: {
            type: "integer",
            description:
              "Maximum number of table rows to include when include_table_rows is true. " +
              "Defaults to 150, capped at 500.",
            minimum: 1,
            maximum: 500,
            default: 150
          },
          scope: {
            type: "string",
            description:
              "Optional topic hint to focus retrieval on part of the TB care cascade. " +
              'Most nuance (e.g., TB/HIV, renal disease, pregnancy) should be expressed in the question text, not as scope values.',
            enum: ["prevention", "screening", "diagnosis", "treatment"],
            examples: ["diagnosis"]
          }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "computePediatricTbTdaScore",
      description:
        "Compute the WHO Pediatric TB Disease Algorithm (TDA) score for children <10 years " +
        "using Algorithm A (with chest X-ray) or Algorithm B (without chest X-ray). " +
        "Returns the total score, whether it meets the WHO treatment-threshold, " +
        "and a human-readable explanation of how each point was assigned.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["algorithm", "age_band"],
        properties: {
          algorithm: {
            type: "string",
            description:
              'Which WHO pediatric TB algorithm to use. "A" when CXR is available, "B" when CXR is not available.',
            enum: ["A", "B"]
          },
          age_band: {
            type: "string",
            description:
              "Age band used for age-specific tachycardia and tachypnoea thresholds.",
            enum: ["<2m", "2-12m", "1-5y", ">5y"]
          },
          symptoms: {
            type: "object",
            description:
              "Symptom flags used in the WHO algorithm. If tachycardia or tachypnoea are omitted here, " +
              "they may be inferred from vitals when provided.",
            additionalProperties: false,
            properties: {
              cough_gt_2w: {
                type: "boolean",
                description: "Cough lasting more than 2 weeks."
              },
              fever_gt_2w: {
                type: "boolean",
                description: "Fever lasting more than 2 weeks."
              },
              lethargy: {
                type: "boolean",
                description: "Lethargy or reduced activity."
              },
              weight_loss_or_ftt: {
                type: "boolean",
                description: "Weight loss or failure to thrive."
              },
              haemoptysis: {
                type: "boolean",
                description: "Haemoptysis (coughing up blood)."
              },
              night_sweats: {
                type: "boolean",
                description: "Night sweats."
              },
              swollen_nodes: {
                type: "boolean",
                description: "Swollen lymph nodes suggestive of TB."
              },
              tachycardia: {
                type: "boolean",
                description:
                  "Clinically identified tachycardia. If omitted, may be inferred from vitals.hr using age-specific thresholds."
              },
              tachypnoea: {
                type: "boolean",
                description:
                  "Clinically identified tachypnoea. If omitted, may be inferred from vitals.rr using age-specific thresholds."
              }
            }
          },
          vitals: {
            type: "object",
            description:
              "Optional vital signs used to infer tachycardia and tachypnoea when those flags are not explicitly set in symptoms.",
            additionalProperties: false,
            properties: {
              hr: {
                type: "number",
                description: "Heart rate in beats per minute."
              },
              rr: {
                type: "number",
                description: "Respiratory rate in breaths per minute."
              }
            }
          },
          cxr: {
            type: "object",
            description:
              "Chest X-ray findings (used only for Algorithm A). " +
              "Ignored when algorithm is B (no CXR).",
            additionalProperties: false,
            properties: {
              cavities: {
                type: "boolean",
                description: "Pulmonary cavities on chest X-ray."
              },
              enlarged_nodes: {
                type: "boolean",
                description: "Enlarged intrathoracic lymph nodes."
              },
              opacities: {
                type: "boolean",
                description: "Parenchymal opacities on chest X-ray."
              },
              miliary: {
                type: "boolean",
                description: "Miliary pattern on chest X-ray."
              },
              effusion: {
                type: "boolean",
                description: "Pleural effusion on chest X-ray."
              }
            }
          }
        }
      }
    }
  }
];


function extractToolsUsed(messages) {
  const used = new Set();
  for (const m of messages) {
    if (m.tool_calls && Array.isArray(m.tool_calls)) {
      for (const c of m.tool_calls) {
        if (c && c.function && c.function.name) {
          used.add(c.function.name);
        }
      }
    }
  }
  return Array.from(used);
}

async function getReasoningSummary(client, assistantContent) {

  try {
    const completion = await client.chat.completions.create({
      model: "gpt-5.1",
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            "You summarize clinical reasoning. Produce a short 3–6 bullet chain-of-thought summary. Do not repeat the full answer; do not add disclaimers."
        },
        { role: "assistant", content: assistantContent }
      ]
    });
    return completion.choices?.[0]?.message?.content || null;
  } catch (err) {
    console.warn("Reasoning summary error:", err?.message || err);
    return null;
  }
}

async function buildCaseSnapshot(client, caseMessages) {
  try {
    const completion = await client.chat.completions.create({
      model: "gpt-5.1",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "You are TB Mentor's case summarizer. Based ONLY on the conversation below between a clinician and TB Mentor, extract a brief snapshot of the CURRENT patient. " +
            "Return a JSON object with exactly two top-level keys: 'caseSummary' (string) and 'snapshot' (object). " +
            "The 'snapshot' object must contain the keys: ageSex, weight, hiv, pregnancy, tbProfile, comorbidities, regimen, danger. " +
            "Use short, clinician-style phrases (e.g., '2 y boy', 'HIV-negative', 'no TB contacts reported'). " +
            "If information is not provided, use an empty string ''. " +
            "If danger signs or TB contacts are clearly absent, say so explicitly (e.g., 'no danger signs reported'). " +
            "Do not include any additional keys or commentary."
        },
        ...caseMessages
      ]
    });

    const content = completion.choices?.[0]?.message?.content || "{}";
    const parsed = JSON.parse(content);
    const snap = parsed.snapshot || {};

    return {
      caseSummary: typeof parsed.caseSummary === "string" ? parsed.caseSummary : "",
      snapshot: {
        ageSex: typeof snap.ageSex === "string" ? snap.ageSex : "",
        weight: typeof snap.weight === "string" ? snap.weight : "",
        hiv: typeof snap.hiv === "string" ? snap.hiv : "",
        pregnancy: typeof snap.pregnancy === "string" ? snap.pregnancy : "",
        tbProfile: typeof snap.tbProfile === "string" ? snap.tbProfile : "",
        comorbidities: typeof snap.comorbidities === "string" ? snap.comorbidities : "",
        regimen: typeof snap.regimen === "string" ? snap.regimen : "",
        danger: typeof snap.danger === "string" ? snap.danger : ""
      }
    };
  } catch (err) {
    console.warn("Case snapshot error:", err?.message || err);
    return { caseSummary: null, snapshot: null };
  }
}

async function buildMentorStatus(client, convoMessages) {
  try {
    const completion = await client.chat.completions.create({
      model: "gpt-5.1",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "You are TB Mentor's status summarizer. Based on the conversation so far between a clinician and TB Mentor, infer the mentor's current status. " +
            "Return a JSON object with exactly these keys: mode, phase, confidence, epiContext, siteProfile, resourceSetting, riskPosture. " +
            "Use concise labels that are comfortable for clinicians to scan quickly. " +
            "The value of mode MUST be one of: 'Clarify', 'Synthesize', 'Plan', 'Teach', or 'Summarize' (no other options). " +
            "The value of phase MUST be one of: 'Intake', 'Identify', 'Workup', 'Treat', or 'Follow-up' (no other options). " +
            "The value of resourceSetting MUST be one of: 'Resource-limited', 'Intermediate', or 'Tertiary / resource-rich' (no other options). " +
            "The value of riskPosture MUST be either 'Conservative' or 'Pragmatic' (capitalize the first letter, no other options). " +
            "If you are uncertain, choose the best-fitting label rather than leaving it blank."
        },
        ...convoMessages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "tool")
      ]
    });

    const content = completion.choices?.[0]?.message?.content || "{}";
    const parsed = JSON.parse(content);

    return {
      mode: typeof parsed.mode === "string" ? parsed.mode : "",
      phase: typeof parsed.phase === "string" ? parsed.phase : "",
      confidence: typeof parsed.confidence === "string" ? parsed.confidence : "",
      epiContext: typeof parsed.epiContext === "string" ? parsed.epiContext : "",
      siteProfile: typeof parsed.siteProfile === "string" ? parsed.siteProfile : "",
      resourceSetting: typeof parsed.resourceSetting === "string" ? parsed.resourceSetting : "",
      riskPosture: typeof parsed.riskPosture === "string" ? parsed.riskPosture : ""
    };
  } catch (err) {
    console.warn("Mentor status error:", err?.message || err);
    return null;
  }
}

async function callRag(args) {
  const ragBase =
    process.env.TB_RAG_BASE_URL ||
    process.env.TB_MENTOR_BASE_URL ||
    "";
  const ragKey =
    process.env.TB_RAG_API_KEY || process.env.TB_RAG_AUTH || null;

  const headers = { "Content-Type": "application/json" };
  if (ragKey) headers.Authorization = `Bearer ${ragKey}`;

  const res = await fetch(`${ragBase}/api/tb-rag-query`, {
    method: "POST",
    headers,
    body: JSON.stringify(args)
  });

  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body?.error ? ` - ${body.error}` : "";
    } catch (_) {}
    throw new Error(`tb-rag-query error: ${res.status} ${res.statusText}${detail}`);
  }

  return await res.json();
}

async function callTda(args) {
  const tdaBase =
    process.env.TB_TDA_BASE_URL ||
    process.env.TB_MENTOR_BASE_URL ||
    "";
  const tdaKey =
    process.env.TB_TDA_API_KEY ||
    process.env.TB_PEDS_TDA_API_KEY ||
    null;

  const headers = { "Content-Type": "application/json" };
  if (tdaKey) headers.Authorization = `Bearer ${tdaKey}`;

  const res = await fetch(`${tdaBase}/api/tb_peds_tda`, {
    method: "POST",
    headers,
    body: JSON.stringify(args)
  });

  if (!res.ok) {
    throw new Error(`tb_peds_tda error: ${res.status} ${res.statusText}`);
  }

  return await res.json();
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { message, history } = req.body || {};

  if (!message || typeof message !== "string") {
    return res.status(400).json({ error: "Missing 'message' in request body" });
  }

  if (history && !Array.isArray(history)) {
    return res.status(400).json({ error: "'history' must be an array" });
  }

  // Keep a small, validated history so conversation context can be reused.
  const sanitizedHistory = (Array.isArray(history) ? history : [])
    .filter(
      (h) =>
        h &&
        typeof h === "object" &&
        (h.role === "user" || h.role === "assistant") &&
        typeof h.content === "string" &&
        h.content.trim()
    )
    .slice(-12) // cap history depth
    .map((h) => ({ role: h.role, content: h.content.trim() }));

  try {
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      ...sanitizedHistory,
      { role: "user", content: message }
    ];

    while (true) {
      const completion = await client.chat.completions.create({
        model: "gpt-5.1",
        messages,
        tools,
        tool_choice: "auto"
      });

      const msg = completion.choices[0].message;

      // If the model wants to call tools
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        messages.push(msg);

        for (const toolCall of msg.tool_calls) {
          const name = toolCall.function.name;
          const args = JSON.parse(toolCall.function.arguments || "{}");

          let result;
          if (name === "fetchRelevantTbGuidance") {
            if (args.top_k == null) args.top_k = 8;
            if (args.include_table_rows == null) args.include_table_rows = true;
            if (args.table_row_limit == null) args.table_row_limit = 150;

            result = await callRag(args);
          } else if (name === "computePediatricTbTdaScore") {
            const usedTools = extractToolsUsed(messages);
            if (!usedTools.includes("fetchRelevantTbGuidance")) {
              result = {
                error: "TDA_PRECONDITION_FAILED",
                detail:
                  "You attempted to call computePediatricTbTdaScore before using fetchRelevantTbGuidance. " +
                  "For real pediatric cases, you must first perform intake, construct CASE SUMMARY / TASK / RAG FOCUS, " +
                  "and call fetchRelevantTbGuidance at least once to retrieve WHO guidance."
              };
            } else {
              result = await callTda(args);
            }
          } else {
            result = { error: `Unknown tool: ${name}` };
          }

          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify(result)
          });
        }

        // Loop again so the model can read tool results and synthesize
        continue;
      }

      // No more tool calls => final answer
      messages.push(msg);

      const toolsUsed = extractToolsUsed(messages);
      const caseMessages = sanitizedHistory.concat({ role: "user", content: message.trim() });

      const [reasoning, snapshotResult, mentorStatus] = await Promise.all([
        getReasoningSummary(client, msg.content),
        buildCaseSnapshot(client, caseMessages),
        buildMentorStatus(client, messages)
      ]);

      return res.status(200).json({
        output: msg.content,
        toolsUsed,
        reasoning,
        caseSummary: snapshotResult.caseSummary,
        snapshot: snapshotResult.snapshot,
        mentorStatus
      });
    }
  } catch (err) {
    console.error("TB Mentor error:", err);
    return res.status(500).json({
      error: "Internal server error",
      detail: err?.message || null
    });
  }
}
