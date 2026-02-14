# Three-LLM-agents
## 以下是三個 LLM 代理協作分別的prompt
### 1. 生成代理
~~~You are an assistant that must produce ONLY valid JSON.
You are given image tags (internal). Use them to write natural {DIALOGUE_LANGUAGE} dialogues about a shrimp's **activity and behavior**, BUT NEVER mention tags/labels/metadata directly.

Output format:
- A single JSON array of dialogue arrays: [ [{{...}}, {{...}}], [{{...}}, {{...}}] ]
- Each dialogue object: {{ "from": "human"|"gpt", "value": "text" }}

Rules:
1. Generate {GEN_DIALOGUE_MIN}-{GEN_DIALOGUE_MAX} INDEPENDENT dialogues.
2. Each dialogue: 1-5 turns.
3.Every dialogue MUST start with a human message whose value begins exactly with "<image>\\n".
4. Speakers must alternate strictly: human, gpt, human, gpt... (no consecutive same speaker).
5. Content: Questions about **what the shrimp is doing (behavior)**, its posture, appendages movement, or position.
6. RESTRICTION: Do NOT mention "tags", "labels", "metadata", "dataset".
7. OBSERVATION: Phrase answers conservatively from a single frame: use "appears/seems/likely" and avoid time-continuous claims.
 E.g., "It appears to be moving its swimmerets." instead of "The tag says swimming".
8. Internal Tag: shrimp_state: {state}
9. Each message object must contain ONLY two keys: "from" and "value". No extra keys.

Background Knowledge (Use these to describe behavior naturally):
- **Swimming**:
  - Shrimp uses pleopods (swimmerets) under the abdomen to propel itself.
  - Body is often extended or slightly curved.
  - Tail fan (uropods) may be spread for steering.
  - Antennae trail backwards or sweep the water.
  - Movement may look forward-directed or mid-motion in the frame.
- **Feeding**:
  - Shrimp uses pereiopods (walking legs) and maxillipeds (feeding appendages) to manipulate food.
  - Often stationary or moving slowly along the bottom (substrate).
  - Head is tilted down towards the substrate.
  - Appendages near the mouth are moving rapidly.
  - May be picking at particles on the floor.
- **Stationary/Resting**:
  - Sitting still on the bottom.
  - Minimal movement of appendages.

Example Q&A (Do not copy verbatim):
- Q: "<image>\\nWhat is the shrimp doing in this video?"
  A: "The shrimp is swimming actively through the water."
- Q: "<image>\\nIs the shrimp eating?"
  A: "Yes, it appears to be feeding near the bottom, using its legs to pick up particles."
- Q: "<image>\\nDescribe the movement of the shrimp."
  A: "It is propelling itself forward using its swimmerets."

Additional constraints from previous check:
{extra_notice}

Return ONLY JSON.
~~~

### 2. 檢核結果
~~~"""
You are given a JSON array of dialogues about shrimp activity and condition.

Your task is to determine whether the dialogues are acceptable (PASS) or need refinement (FAIL).

#### dialogues
{conversations_json_text}

#### task_note (you MUST follow all rules strictly)

The entire set is acceptable (PASS) only if ALL answers:

1. Are grounded in plausible visual observation 
   (e.g., posture, body orientation, limb movement, relative position in frame).
2. Do NOT mention or rely on non-visual metadata such as:
   "tags", "labels", "annotations", "dataset", "provided list", or similar.
3. Do NOT claim medical diagnosis, lab confirmation, or external knowledge.
4. If describing activity (e.g., swimming, feeding), 
   use conservative phrasing such as:
   - "appears to be"
   - "seems to be"
   - "likely"
5. Do NOT introduce details that cannot reasonably be inferred from a still image.
6. Strictly follow dialogue structure:
   - First message must be from "human"
   - It must begin exactly with "<image>\n"
   - Speakers must alternate human/gpt
   - 1–5 turns per dialogue
   - Only keys allowed: "from", "value"

If ANY rule is violated, mark the whole set as FAIL.

#### Output format

Return ONLY this JSON:

{{
  "reason": "short justification",
  "result": "PASS" | "FAIL"
}}

No additional text.
~~~

### 3. 修正代理
~~~"""
You are given dialogues that failed validation and the reason for failure.

Fail reason:
{fail_reason}

Failed generation (snippet):
{last_gen_text[:800]}


Your task:
1. Identify the main structural or content problems.
2. Write a short, structured NOTICE block that will be appended to the next generation prompt.
3. The notice must:
   - Enforce JSON-only output
   - Enforce "<image>\n" at the start
   - Enforce strict alternation of speakers
   - Prohibit mention of tags/metadata
   - Require conservative activity phrasing ("appears", "seems")

Return ONLY the NOTICE text.
No JSON.
No explanations.
~~~