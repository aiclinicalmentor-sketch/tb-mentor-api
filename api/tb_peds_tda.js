// api/tb_peds_tda.js
const B_POINTS = {
  cough_gt_2w: 5,
  fever_gt_2w: 10,
  lethargy: 4,
  weight_loss_or_ftt: 5,
  haemoptysis: 9,
  night_sweats: 6,
  swollen_nodes: 7,
  tachycardia: 4,
  tachypnoea: 2
};
const A_SYMPTOMS = {
  cough_gt_2w: 2,
  fever_gt_2w: 5,
  lethargy: 3,
  weight_loss_or_ftt: 3,
  haemoptysis: 4,
  night_sweats: 2,
  swollen_nodes: 4,
  tachycardia: 2,
  tachypnoea: -1
};
const A_CXR = {
  cavities: 6,
  enlarged_nodes: 17,
  opacities: 5,
  miliary: 15,
  effusion: 8
};
const TREAT_THRESHOLD = 11; // >10 meets threshold

export function computePediatricTbTdaScore(input = {}) {
  const { algorithm, age_band, symptoms = {}, vitals = {}, cxr = {} } = input;

  if (!algorithm || !age_band) {
    throw new Error("Missing algorithm or age_band");
  }
  if (algorithm !== "A" && algorithm !== "B") {
    throw new Error("Algorithm must be 'A' or 'B'");
  }

  // Derive tachy flags if not provided explicitly
  const tachycardia =
    typeof symptoms.tachycardia === "boolean"
      ? symptoms.tachycardia
      : isTachycardic(age_band, vitals.hr);
  const tachypnoea =
    typeof symptoms.tachypnoea === "boolean"
      ? symptoms.tachypnoea
      : isTachypnoeic(age_band, vitals.rr);

  let score = 0;
  const explanation = [];

  if (algorithm === "B") {
    const entries = { ...symptoms, tachycardia, tachypnoea };
    for (const [k, v] of Object.entries(entries)) {
      if (v && B_POINTS[k] != null) {
        score += B_POINTS[k];
        explanation.push(`${human(k)} (${formatPoints(B_POINTS[k])})`);
      }
    }
  } else {
    const entriesA = { ...symptoms, tachycardia, tachypnoea };
    for (const [k, v] of Object.entries(entriesA)) {
      if (v && A_SYMPTOMS[k] != null) {
        score += A_SYMPTOMS[k];
        explanation.push(`${human(k)} (${formatPoints(A_SYMPTOMS[k])})`);
      }
    }
    for (const [k, v] of Object.entries(cxr)) {
      if (v && A_CXR[k] != null) {
        score += A_CXR[k];
        explanation.push(`CXR ${human(k)} (${formatPoints(A_CXR[k])})`);
      }
    }
  }

  return {
    algorithm,
    score,
    meets_threshold: score >= TREAT_THRESHOLD,
    explanation,
    derived: { tachycardia, tachypnoea }
  };
}

export default async function handler(req, res) {
  const API_KEY = process.env.TB_PEDS_TDA_API_KEY || "";

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  if (!API_KEY) {
    res.status(500).json({ error: "Server misconfigured: missing API key" });
    return;
  }
  if ((req.headers.authorization || "") !== `Bearer ${API_KEY}`) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  let body = {};
  try {
    body =
      typeof req.body === "object" && req.body ? req.body : JSON.parse(await readBody(req));
  } catch {
    res.status(400).json({ error: "Invalid JSON" });
    return;
  }

  const algorithm = body.algorithm;
  const age_band = body.age_band;
  const symptoms = body.symptoms || {};
  const vitals = body.vitals || {};
  const cxr = body.cxr || {};

  try {
    const { derived, ...result } = computePediatricTbTdaScore({
      algorithm,
      age_band,
      symptoms,
      vitals,
      cxr
    });

    console.log(
      "[tb_peds_tda] request",
      JSON.stringify({
        method: req.method,
        algorithm,
        age_band,
        symptoms,
        vitals,
        cxr,
        derived
      })
    );

    console.log("[tb_peds_tda] response", JSON.stringify(result));
    res.status(200).json(result);
  } catch (err) {
    res.status(400).json({ error: err?.message || "Invalid request" });
  }
}

// ---- helpers ----
function isTachypnoeic(age, rr) {
  if (rr == null) return false;
  if (age === "<2m") return rr >= 60;
  if (age === "2-12m") return rr >= 50;
  if (age === "1-5y") return rr >= 40;
  return rr >= 30; // >5y
}
function isTachycardic(age, hr) {
  if (hr == null) return false;
  if (age === "<2m") return hr >= 170;
  if (age === "2-12m") return hr >= 160;
  if (age === "1-5y") return hr >= 140;
  return hr >= 120; // >5y
}
function human(k) {
  return k
    .replace(/_/g, " ")
    .replace("gt", ">")
    .replace("ftt", "FTT")
    .replace(/\b([a-z])/g, (_, c) => c.toUpperCase());
}
function formatPoints(n) {
  return n >= 0 ? `+${n}` : `${n}`;
}
function readBody(req) {
  return new Promise((resolve) => {
    let d = "";
    req.on("data", (c) => (d += c));
    req.on("end", () => resolve(d || "{}"));
  });
}
