import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Enable JSON bodies up to 15mb to handle base64 skin canvas image uploads
app.use(express.json({ limit: "15mb" }));

// Helper to get GoogleGenAI client lazily & securely
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    throw new Error("GEMINI_API_KEY is not configured. Please add it via the Secrets panel in Settings.");
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// 1. HEALTHCHECK ENDPOINT
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

// 2. SKIN ANALYSIS VISION ENDPOINT
app.post("/api/analyze", async (req, res) => {
  try {
    const { image, mimeType } = req.body;
    if (!image) {
      return res.status(400).json({ error: "Missing skin image data (base64 is required)." });
    }

    const ai = getGeminiClient();

    // Standard baseline base64 cleanup
    const base64Data = image.replace(/^data:image\/\w+;base64,/, "");

    const imagePart = {
      inlineData: {
        mimeType: mimeType || "image/jpeg",
        data: base64Data,
      },
    };

    const promptPart = {
      text: `Analyze this skin close-up image for dermatology indications. 
Determine if there are signs of acne, pigmentation, dark spots, dryness, oiliess, irritation/redness, or possible infections.
Return a structured skin assessment.
IMPORTANT MEDICAL SAFETY CONSTRAINTS:
- Do NOT prescribe or list any dangerous custom/prescription-only medicines (e.g., isotretinoin, oral antibiotics, prescription tretinoin).
- Recommend only safe over-the-counter (OTC) ingredients and routines (e.g., salicylic acid cleansers, niacinamide serum, broad spectrum SPF 50 sunscreen, aloe vera gel, centella, ceramides).
- For each recommendation, suggest 3 highly matched OTC skincare products detailing exactly what they are, when to apply them (Morning/Night/Both), step-by-step physical application guides (exactly how to use on skin), and how they offset the symptoms.
- If severity is Severe or infection risk is high, set 'dermatologistWarning' to true, and explain kindly that professional medical consultation is urgent.
- Avoid claiming 100% absolute diagnostic certainty; frame it as an educational, non-invasive assessment.`,
    };

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: { parts: [imagePart, promptPart] },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            detectedCondition: {
              type: Type.STRING,
              description: "Name of the primary detected skin state or condition, e.g. 'Mild Acne and Oily Areas', 'Healthy Dry Skin', 'Localized Redness/Irritation'."
            },
            confidence: {
              type: Type.INTEGER,
              description: "Confidence scoring between 0 and 100."
            },
            severity: {
              type: Type.STRING,
              description: "Overall severity indicator: 'Healthy', 'Mild', 'Moderate', 'Severe'."
            },
            metrics: {
              type: Type.OBJECT,
              properties: {
                acne: { type: Type.INTEGER, description: "Intensity score 0-100" },
                pigmentation: { type: Type.INTEGER, description: "Intensity score 0-100" },
                darkSpots: { type: Type.INTEGER, description: "Intensity score 0-100" },
                dryness: { type: Type.INTEGER, description: "Intensity score 0-100" },
                oily: { type: Type.INTEGER, description: "Intensity score 0-100" },
                irritation: { type: Type.INTEGER, description: "Intensity score 0-100" },
                redness: { type: Type.INTEGER, description: "Intensity score 0-100" },
                infectionRisk: { type: Type.INTEGER, description: "Infection risk percentage 0-100" },
              },
              required: ["acne", "pigmentation", "darkSpots", "dryness", "oily", "irritation", "redness", "infectionRisk"]
            },
            recommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "List of general non-prescriptive daily routine and OTC product types to seek (e.g. gentle foaming cleanser, SPF 50 sunscreen)."
            },
            suggestedIngredients: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Skincare ingredients that are highly compatible with this state (e.g. Hyaluronic Acid, Centella Asiatica, Niacinamide)."
            },
            precautions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Things to avoid right now (e.g. avoid high-percentage physical scrubs, limit direct midday sun exposure)."
            },
            hydrationAdvice: {
              type: Type.STRING,
              description: "Specific tailored hydration guidance (e.g. increase ambient humidity, drink 2.5L and apply hyaluronic acid on damp skin)."
            },
            dermatologistWarning: {
              type: Type.BOOLEAN,
              description: "Must be true if condition is severe or has an infection risk."
            },
            dermatologistDetail: {
              type: Type.STRING,
              description: "A kind suggestion on when to see a clinical dermatologist in person."
            },
            recommendedProducts: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING, description: "Unique short alphanumeric string e.g., 'p1', 'p2'" },
                  name: { type: Type.STRING, description: "Name of the target skincare product." },
                  category: { type: Type.STRING, description: "Category (e.g., 'Cleanser', 'Serum', 'Moisturizer', 'Sunscreen')" },
                  whenToUse: { type: Type.STRING, description: "Exactly when to use: 'Morning', 'Night', 'Both', or 'Morning & Night'" },
                  howToApply: { type: Type.STRING, description: "Explicit manual instructions explaining how to apply it, recommended skin condition, and dwell time." },
                  benefits: { type: Type.STRING, description: "How exactly this product improves the specific skin condition." }
                },
                required: ["id", "name", "category", "whenToUse", "howToApply", "benefits"]
              },
              description: "An array of tailored OTC skincare product suggestions detailing what, when, how to apply, and why."
            }
          },
          required: [
            "detectedCondition",
            "confidence",
            "severity",
            "metrics",
            "recommendations",
            "suggestedIngredients",
            "precautions",
            "hydrationAdvice",
            "dermatologistWarning",
            "dermatologistDetail",
            "recommendedProducts"
          ]
        }
      }
    });

    const parsedData = JSON.parse(response.text || "{}");
    return res.json(parsedData);

  } catch (error: any) {
    console.error("Skin analysis error:", error);
    // Provide a premium high-quality fallback analysis if API key is missing, for demonstration and user review
    if (error.message && error.message.includes("GEMINI_API_KEY")) {
      return res.status(200).json({
        isDemoFallback: true,
        detectedCondition: "Slight Epidermal Dehydration & Mild Redness (Demo Mode)",
        confidence: 94,
        severity: "Mild",
        metrics: {
          acne: 24,
          pigmentation: 18,
          darkSpots: 12,
          dryness: 45,
          oily: 35,
          irritation: 15,
          redness: 10,
          infectionRisk: 5
        },
        recommendations: [
          "Use a gentle non-stripping moisturizing foaming cleanser daily",
          "Apply an ultra-lightweight Niacinamide (2-4%) evening serum",
          "Always apply broad-spectrum sunscreen SPF 50 before light exposure",
          "Apply Ceramide-rich soothing moisturizers to lock barrier health"
        ],
        suggestedIngredients: [
          "Niacinamide (Vitamin B3) for moisture barrier",
          "Ceramides NP/AP for skin lipid reinforcement",
          "Hyaluronic Acid for multi-layer hydration",
          "Aloe Vera Exosome or Centella for redness soothing"
        ],
        precautions: [
          "Avoid direct intensive scrubbing or harsh coarse face exfoliants",
          "Limit immediate application of low-grade drying alcohols",
          "Keep skin protected under intense UV levels"
        ],
        hydrationAdvice: "Your epidermal moisture level is slightly low. We recommend hydrating inside-out (2.2L water daily) and locking hydration using a lightweight humectant serum followed by a lipid-sealing barrier moisturizer.",
        dermatologistWarning: false,
        dermatologistDetail: "While your assessment shows only mild signs of dryness and oily areas, we recommend scheduling an annual in-person routine screen with a certified dermatologist.",
        recommendedProducts: [
          {
            id: "demop1",
            name: "Ceramide Gentle Hydration Wash",
            category: "Cleanser",
            whenToUse: "Both",
            howToApply: "Massage 1-2 pumps onto damp skin with lukewarm water for 45-60 seconds daily. Pat dry gently with a face towel.",
            benefits: "Cleanses surface bacteria and grid without dissolving the dry epidermis moisture stratum layer."
          },
          {
            id: "demop2",
            name: "Duo Hyaluronic & 3% Niacinamide Ampoule",
            category: "Serum",
            whenToUse: "Both",
            howToApply: "Apply 3 droplets onto your clean, slightly damp skin. Spread outwards gently and pat to absorb.",
            benefits: "Retains hydration tightly and speeds up epidermal restoration, fading dryness spots."
          },
          {
            id: "demop3",
            name: "Barrier Restoring Ceramide Gel-Cream",
            category: "Moisturizer",
            whenToUse: "Night",
            howToApply: "Apply a light, pea-sized layer to your face every night as the final step to lock in active water retention.",
            benefits: "Re-bonds vulnerable lipids and acts as an overnight skin barrier protector."
          }
        ]
      });
    }

    return res.status(500).json({ error: error.message || "Failed to analyze skin image." });
  }
});

// 3. HEALTHCARE CHATBOT ENDPOINT
app.post("/api/chat", async (req, res) => {
  const { message, history } = req.body || {};
  try {
    if (!message) {
      return res.status(400).json({ error: "Missing message payload." });
    }

    const ai = getGeminiClient();

    // Map the user history format to standard structures for Gemini
    // Max last 10 messages for lightweight performance
    const conversationHistory = (history || []).slice(-10).map((msg: any) => {
      return {
        role: msg.sender === "user" ? "user" : "model",
        parts: [{ text: msg.text }]
      };
    });

    // Add immediate incoming query
    conversationHistory.push({
      role: "user",
      parts: [{ text: message }]
    });

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: conversationHistory,
      config: {
        systemInstruction: `You are the premium, highly intelligent AI Dermatology & Skincare Companion named DermaVision AI.
Your purpose is to answer user queries with elegant, highly compassionate, professional, and science-backed skincare advice.
SAFETY MANDATES:
- You are an educational AI assistant, NOT a doctor.
- You MUST NEVER prescribe or endorse dangerous prescription-only pharmaceuticals (e.g. oral isotretinoin/Accutane, prescription oral clindamycin, high-dose steroids, spironolactone).
- Recommend highly safe, elegant OTC skincare ingredients (e.g. Salicylic Acid, Glycerin, Ceramides, Zinc Oxide, Niacinamide, Squalane, Centella Asiatica, Aloe Vera, Bakuchiol).
- Strictly warn they must consult a board-certified dermatologist for clinical skin infections or deep cystic conditions.
- Keep recommendations clear, elegant, scannable, and reassuring. Always output beautifully-spaced markdown with premium lists or text.`
      }
    });

    return res.json({ text: response.text });
  } catch (error: any) {
    console.error("Healthcare chatbot error:", error);

    // Provide premium fallback responses for demonstration if key is missing
    if (error.message && error.message.includes("GEMINI_API_KEY")) {
      const lowerMsg = message.toLowerCase();
      let replyText = "Thank you for reaching out to DermaVision AI's skin concierge. ";

      if (lowerMsg.includes("acne") || lowerMsg.includes("pimple")) {
        replyText += `\n\nFor mild skin blemishes and acne-prone skin, a balanced and non-stripping approach is key:\n\n` +
          `- **Morning:** Use a gentle, salicylic acid (0.5% - 2%) cleanser to clear pore blockages, followed by a light humectant gel and SPF 50 Broad-Spectrum sunscreen.\n` +
          `- **Evening:** Cleanse with a sensitive-skin wash, then apply a low-strength Niacinamide or Bakuchiol serum. Lock hydration with a non-comedogenic gel moisturizer.\n` +
          `- **Avoid:** Popping blemishes, touch trigger-points, or applying concentrated alcohol-based astringents.`;
      } else if (lowerMsg.includes("dry") || lowerMsg.includes("hydrate") || lowerMsg.includes("hydration")) {
        replyText += `\n\nTo restore balance to dry or dehydrated skin barriers, seek intensive lipid restoration:\n\n` +
          `- **Humectants:** Apply multi-molecular Hyaluronic Acid or Panthenol formulas to damp skin.\n` +
          `- **Emollients:** Look for creams containing Ceramides NP/AP/EOP and squalane to replicate your skin's natural moisture barrier.\n` +
          `- **Lifestyle:** Keep showers lukewarm, drink sufficient structured water (2-2.5L), and consider using a bedtime ultrasonic room humidifier.`;
      } else if (lowerMsg.includes("routine") || lowerMsg.includes("morning") || lowerMsg.includes("night")) {
        replyText += `\n\nA solid skincare routine relies on consistency and premium ingredient layering:\n\n` +
          `1. **Cleanse:** A pH-balanced (approx. 5.5) gentle wash that doesn't leave skin tight.\n` +
          `2. **Treat:** Apply specialized target solutions (e.g., Niacinamide for spots/redness, Salicylic for pore clearance, Vitamin C for radiance in mornings).\n` +
          `3. **Moisturize:** A lipid-sealing barrier formulation appropriate for your skin type.\n` +
          `4. **Protect:** Broad-spectrum SPF 50 is non-negotiable every morning.`;
      } else {
        replyText += `\n\nI can assist you with customized routine tips, detailing safe OTC skincare ingredients (such as Hyaluronic acid, Niacinamide, Ceramides, or Salicylic acid), and setting up healthy hydration protocols.\n\n*Reminder: This informational analysis doesn't replace structured clinical dermatology diagnostics.*`;
      }

      return res.json({ text: replyText, isDemoFallback: true });
    }

    return res.status(500).json({ error: error.message || "Failed to process chat response." });
  }
});

// 4. EMBED VITE MIDDLEWARE (DEVELOPMENT) OR STATIC FILE SERVING (PRODUCTION)
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Vite dev middleware loaded continuously.");
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log("Serving statically compiled build from dist/");
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`DermaVision AI server running on port ${PORT}`);
  });
}

startServer();
