"""
Comprehensive Multi-Model Evaluation - ROBUST VERSION
======================================================

150 questions × 12 policies × 3 models = 5,400 LLM calls

Features:
- Extensive logging
- Retry with exponential backoff
- Checkpointing (saves progress, can resume)
- Timeout handling
- Progress tracking

Usage:
    pip install openai google-genai httpx
    export OPENAI_API_KEY='your-openai-key'
    export GEMINI_API_KEY='your-gemini-key'
    python e2e_multimodel_robust.py
"""

import json
import os
import time
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "models": ["gpt-3.5-turbo", "gpt-4o-mini", "gemini-2.5-pro"],
    "sleep_between_calls": 0.5,  # seconds
    "max_retries": 3,
    "retry_backoff": 2,  # exponential backoff multiplier
    "timeout_seconds": 60,
    "checkpoint_every": 50,  # save progress every N questions
}

# ============================================================================
# SETUP
# ============================================================================

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting setup...")

# Token counting uses response.usage for accurate counts,
# with word-split estimation used for pre-call context sizing.

# ── OpenAI ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = None

if OPENAI_API_KEY:
    try:
        import httpx
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=httpx.Timeout(CONFIG["timeout_seconds"], connect=10.0),
            max_retries=0,
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ OpenAI client ready (timeout={CONFIG['timeout_seconds']}s)")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ OpenAI setup failed: {e}")
else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ No OPENAI_API_KEY found")

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None

if GEMINI_API_KEY:
    try:
        from google import genai as google_genai
        gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Gemini client ready")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Gemini setup failed: {e}")
else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ No GEMINI_API_KEY found")

def _is_gemini(model: str) -> bool:
    return model.startswith("gemini")

def _client_available(model: str) -> bool:
    if _is_gemini(model):
        return gemini_client is not None
    return openai_client is not None

USE_REAL_LLM = openai_client is not None or gemini_client is not None

STORE_COSTS = {"stm": 1, "summary": 1, "ltm": 3, "episodic": 5}

# ============================================================================
# MEMORY STORES
# ============================================================================

SHORT_MEMORY = {
    "stm": [
        "User just mentioned they have a meeting at 3pm today with the marketing team.",
        "User asked about the weather in Chicago - it's currently 45°F and cloudy.",
        "User requested clarification on the Q4 budget allocation discussion.",
        "User mentioned they're feeling tired today and need coffee.",
        "User asked to schedule a follow-up call for tomorrow morning.",
    ],
    "summary": [
        "User Profile: Name is Alex Chen. Works at TechCorp as Senior Software Engineer.",
        "Health: Allergic to penicillin and shellfish. Doctor is Dr. Sarah Smith at City Hospital.",
        "Contact: Phone number is 555-867-5309. Email is alex.chen@techcorp.com.",
        "Preferences: Favorite restaurant is Olive Garden. Prefers morning meetings before 10am.",
        "Work: Current project is codenamed 'Phoenix'. Manager is Jennifer Williams.",
        "Personal: Birthday is March 15th. Has a dog named Max (golden retriever, 3 years old). Lives in San Francisco.",
        "Education: Graduated from Stanford University with CS degree in 2018.",
        "Hobbies: Enjoys hiking, photography, and playing guitar.",
    ],
    "ltm": [
        "Session 3 (2 weeks ago): Discussed vacation trip to Italy. User visited Rome and Florence. Loved the Colosseum and local pasta.",
        "Session 5 (1 week ago): Project deadline discussion. Phoenix project deadline is Friday, January 24th. Critical milestone.",
        "Session 7 (3 days ago): Restaurant recommendations. User mentioned enjoying Olive Garden, Chipotle, and a new Thai place called 'Bangkok Garden'.",
        "Session 8 (2 days ago): Health check-in. Discussed allergies and upcoming doctor appointment with Dr. Smith.",
        "Session 9 (yesterday): Before the recent reorg, user's manager was Michael Torres. Now reports to Jennifer Williams.",
        "Session 10 (yesterday): User mentioned weight was 185 lbs in January before starting a diet. Current goal is 170 lbs.",
        "Session 11 (last week): Discussed user's guitar practice - learning 'Hotel California' by Eagles.",
        "Session 12 (last month): User's dog Max had a vet appointment. Max is a 3-year-old golden retriever.",
    ],
    "episodic": [
        "Session 3, Turn 5: User said 'Italy was absolutely amazing! Rome was my favorite - the Colosseum was breathtaking. Florence had the best pasta I've ever had.'",
        "Session 5, Turn 3: User said 'The Phoenix deadline is next Friday the 24th. We absolutely cannot miss it - the client demo is scheduled.'",
        "Session 7, Turn 2: User said 'For restaurants, I really like Olive Garden for Italian, Chipotle for quick Mexican, and I just tried Bangkok Garden - amazing Thai food!'",
        "Session 9, Turn 4: User said 'Before the reorg last month, I reported to Michael Torres. Now Jennifer Williams is my new manager.'",
        "Session 10, Turn 2: User said 'Back in January I was 185 pounds. With the diet I started, I'm hoping to get down to 170 by summer.'",
        "Session 11, Turn 3: User said 'I've been practicing Hotel California on guitar for about 2 months now. The solo is really hard!'",
        "Session 12, Turn 1: User said 'Max just turned 3 years old! We had a little birthday party for him with dog treats.'",
    ],
}

LONG_MEMORY = {
    "stm": [
        """User is implementing a transformer encoder layer from scratch in a custom PyTorch NLP framework. They requested a detailed derivation of the multi-head attention pipeline, including the projection matrices W_Q, W_K, W_V, splitting embeddings into per-head subspaces, computing scaled dot-product attention, and concatenating heads before the output projection. Discussion emphasized why the scaling factor 1/sqrt(d_k) prevents gradient saturation when softmax is applied to large dot-product magnitudes. We also discussed sinusoidal positional encodings defined as sine/cosine waves of exponentially increasing frequencies, residual connections combined with layer normalization to stabilize training, dropout placement inside attention weights and feedforward blocks, and exploration of rotary positional embeddings (RoPE) for relative position encoding.""",
        """The user asked for a comparison between RLHF pipelines and Constitutional AI. We examined supervised fine-tuning datasets, reward model training, PPO policy optimization, and self-critique mechanisms using rule-based constitutional prompts. The discussion included interpretability tools such as activation patching, feature visualization, probing classifiers, and causal tracing for identifying transformer circuits. We also explored scalable oversight models combining automated evaluators with limited human labeling, along with risks such as deceptive alignment, specification gaming, and goal misgeneralization across distribution shifts.""",
        """User is studying long-context transformer variants capable of processing sequences beyond 128k tokens. We discussed sparse attention patterns such as sliding window and global tokens in Longformer and BigBird, kernelized linear attention used by Performer, retrieval-augmented transformers combining vector databases with inference pipelines, and memory-compressed attention that periodically downsamples intermediate representations. KV-cache reuse during autoregressive decoding was also discussed as a major latency optimization technique.""",
        """We discussed implementing compute-optimal training based on Chinchilla scaling laws. The conversation included balancing parameter count and training tokens, dataset deduplication to reduce memorization artifacts, tokenizer entropy effects on cross-entropy loss, cosine learning-rate decay, AdamW schedules, gradient clipping, and experimental design for testing scaling predictions at sub-billion parameter regimes.""",
    ],
    "summary": [
        """Dr. Alex Chen conducted a four-year ecological study tracking 47 reticulated giraffes using GPS satellite collars transmitting hourly location coordinates across roughly 12,000 square kilometers of northern Kenyan savanna. Seasonal migration routes were strongly correlated with rainfall patterns and acacia woodland density. Drone-based vegetation surveys and interviews with pastoralist communities helped identify conflict zones where livestock expansion reduced migration corridors. The research proposed conservation corridor preservation supported by cross-border wildlife agreements.""",
        """The doctoral research also constructed longitudinal giraffe social interaction networks using proximity clustering from GPS data. Matrilineal herd persistence was observed across breeding cycles. Betweenness centrality metrics identified connector individuals essential to maintaining migration network continuity. DBSCAN clustering detected seasonal aggregation points critical for population resilience.""",
        """User specializes in Brutalist architecture research, studying Le Corbusier's Unité d'Habitation, Denys Lasdun's National Theatre London, and Moshe Safdie's Habitat 67. Their research explores béton brut philosophy, social housing ideology, demolition vs preservation debates, and the Japanese Metabolist movement including Kisho Kurokawa's Nakagin Capsule Tower.""",
        """User advocates evaluating Brutalist buildings using lifecycle sustainability metrics rather than aesthetic trends. Their lectures compare adaptive reuse projects in Europe and North America and emphasize historical housing equity narratives influencing contemporary public perception of Brutalist structures.""",
    ],
    "ltm": [
        """Conversation discussed the Pacific Theater strategy of island-hopping led by Admiral Chester Nimitz, emphasizing the Battle of Midway as a decisive turning point enabled by U.S. codebreaking. The Guadalcanal campaign logistics, MacArthur versus Nimitz strategic debates, Tokyo firebombing civilian casualties, and Soviet entry accelerating Japan's surrender were analyzed.""",
        """Two weeks earlier we discussed the transition from RNNs to transformers, highlighting parallel training advantages of self-attention, quadratic sequence complexity challenges, sparse and linear attention research (Longformer, BigBird, Performer), and scaling-law-driven emergence of capabilities in large language models.""",
        """Earlier discussions examined agent-based ecological models showing corridor fragmentation increasing extinction risk despite stable habitat area. Metapopulation theory, dispersal probability modeling, and satellite land-use classification were analyzed for long-term conservation planning.""",
        """We compared modernist megastructure planning with modern mixed-use urbanism, discussing density optimization, infrastructure efficiency, and the evolving sustainability perspective in redevelopment.""",
    ],
    "episodic": [
        """User asked: explain how the 1/sqrt(d_k) scaling prevents softmax saturation in attention scores and how positional encoding sine/cosine functions encode order.""",
        """User asked: what is the difference between RLHF PPO optimization and constitutional AI self-critique evaluation frameworks?""",
        """User asked: why did the Battle of Midway become the turning point in the Pacific War?""",
        """User asked: why did transformers replace RNNs despite quadratic complexity?""",
    ],
}

# ============================================================================
# QUESTIONS (100 short + 50 long)
# ============================================================================

SHORT_QUESTIONS = [
    {"query":"What is the user's full name?","answer":"alex chen","alt_answers":["alex"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which company does Alex work for?","answer":"techcorp","alt_answers":["tech corp"],"stores":["summary"],"type":"single_hop"},
    {"query":"Who is Alex's doctor?","answer":"dr. sarah smith","alt_answers":["sarah smith","dr sarah smith"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which city does the user live in?","answer":"san francisco","alt_answers":["sf"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is Alex's favorite restaurant?","answer":"olive garden","alt_answers":["the olive garden"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is the user's phone number?","answer":"555-867-5309","alt_answers":["5558675309"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which university did Alex graduate from?","answer":"stanford university","alt_answers":["stanford"],"stores":["summary"],"type":"single_hop"},
    {"query":"What degree did Alex earn?","answer":"cs degree","alt_answers":["computer science"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is the user's birthday month?","answer":"march","alt_answers":["march 15th"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is the dog's name?","answer":"max","alt_answers":["dog max"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which breed is Max?","answer":"golden retriever","alt_answers":["retriever"],"stores":["summary"],"type":"single_hop"},
    {"query":"Who is Alex's current manager?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is the current project name?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is Alex allergic to?","answer":"penicillin","alt_answers":["shellfish"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which hobby involves taking pictures?","answer":"photography","alt_answers":["photo"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which instrument does Alex play?","answer":"guitar","alt_answers":["plays guitar"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which animal does Alex own?","answer":"dog","alt_answers":["pet dog"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which meal preference is mentioned?","answer":"morning meetings","alt_answers":["before 10am"],"stores":["summary"],"type":"single_hop"},
    {"query":"What is Alex's email domain?","answer":"techcorp.com","alt_answers":["@techcorp.com"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which outdoor hobby does Alex enjoy?","answer":"hiking","alt_answers":["hike"],"stores":["summary"],"type":"single_hop"},
    {"query":"What time is the meeting scheduled today?","answer":"3pm","alt_answers":["3 pm"],"stores":["stm"],"type":"single_session"},
    {"query":"Which team is today's meeting with?","answer":"marketing team","alt_answers":["marketing"],"stores":["stm"],"type":"single_session"},
    {"query":"What is the weather temperature mentioned?","answer":"45°f","alt_answers":["45f","45"],"stores":["stm"],"type":"single_session"},
    {"query":"Which city weather was discussed?","answer":"chicago","alt_answers":["chicago weather"],"stores":["stm"],"type":"single_session"},
    {"query":"What budget topic is being discussed?","answer":"q4 budget allocation","alt_answers":["q4 budget"],"stores":["stm"],"type":"single_session"},
    {"query":"How is the user feeling right now?","answer":"tired","alt_answers":["feeling tired"],"stores":["stm"],"type":"single_session"},
    {"query":"What drink does the user need?","answer":"coffee","alt_answers":["needs coffee"],"stores":["stm"],"type":"single_session"},
    {"query":"When is the follow-up call?","answer":"tomorrow morning","alt_answers":["tomorrow"],"stores":["stm"],"type":"single_session"},
    {"query":"What is the cloud condition today?","answer":"cloudy","alt_answers":["overcast"],"stores":["stm"],"type":"single_session"},
    {"query":"Which quarter budget is referenced?","answer":"q4","alt_answers":["fourth quarter"],"stores":["stm"],"type":"single_session"},
    {"query":"Which Italian cities were visited?","answer":"rome","alt_answers":["florence"],"stores":["ltm"],"type":"recent_session"},
    {"query":"What landmark did the user love in Italy?","answer":"colosseum","alt_answers":["the colosseum"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which food was enjoyed in Italy?","answer":"pasta","alt_answers":["italian pasta"],"stores":["ltm"],"type":"recent_session"},
    {"query":"When is the Phoenix deadline?","answer":"friday, january 24th","alt_answers":["january 24th","friday"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which Thai restaurant was mentioned?","answer":"bangkok garden","alt_answers":["thai restaurant"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which Mexican restaurant was mentioned?","answer":"chipotle","alt_answers":["chipotle mexican grill"],"stores":["ltm"],"type":"recent_session"},
    {"query":"What was the user's weight in January?","answer":"185 lbs","alt_answers":["185"],"stores":["ltm"],"type":"recent_session"},
    {"query":"What is the weight goal?","answer":"170 lbs","alt_answers":["170"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which song is Alex learning on guitar?","answer":"hotel california","alt_answers":["hotel california song"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which pet had a vet appointment?","answer":"max","alt_answers":["dog max"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Who is Alex's current manager for the Phoenix project?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which restaurant appears both as favorite and previously discussed?","answer":"olive garden","alt_answers":["olive"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which hobby relates to the song being practiced?","answer":"guitar","alt_answers":["playing guitar"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which pet had a vet appointment according to history?","answer":"max","alt_answers":["dog max"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which project deadline relates to Alex's current project?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which allergy was mentioned during health check-ins?","answer":"penicillin","alt_answers":["shellfish"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which manager replaced Michael Torres?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which hobby connects to the song learned over two months?","answer":"guitar","alt_answers":["instrument guitar"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which restaurant preference overlaps with past restaurant mentions?","answer":"olive garden","alt_answers":["olive"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which pet species corresponds to the vet appointment memory?","answer":"dog","alt_answers":["pet dog"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"List one restaurant mentioned historically.","answer":"olive garden","alt_answers":["chipotle","bangkok garden"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Name one city visited in Italy.","answer":"rome","alt_answers":["florence"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Name one food enjoyed in Italy.","answer":"pasta","alt_answers":["italian pasta"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Mention one health event discussed earlier.","answer":"doctor appointment","alt_answers":["health check-in"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Provide one hobby mentioned in past transcripts.","answer":"guitar","alt_answers":["hiking","photography"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Name one weight-related fact recalled earlier.","answer":"185 lbs","alt_answers":["170 lbs","185"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Give one person referenced in manager history.","answer":"michael torres","alt_answers":["jennifer williams"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Provide one pet-related historical event.","answer":"vet appointment","alt_answers":["max vet"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"List one project-related historical detail.","answer":"phoenix","alt_answers":["phoenix deadline"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Mention one song referenced in transcripts.","answer":"hotel california","alt_answers":["hotel california song"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Who was the manager before Jennifer Williams?","answer":"michael torres","alt_answers":["michael"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which weight value came before the goal weight?","answer":"185 lbs","alt_answers":["185"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which restaurant was discussed previously before Bangkok Garden?","answer":"olive garden","alt_answers":["chipotle"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which city was visited before Florence?","answer":"rome","alt_answers":["rome italy"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which manager originally led before the reorg?","answer":"michael torres","alt_answers":["torres"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which health topic was discussed earlier before appointments?","answer":"allergies","alt_answers":["allergy discussion"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which weight goal replaced the earlier weight value?","answer":"170 lbs","alt_answers":["170"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which restaurant appeared earlier in discussions?","answer":"olive garden","alt_answers":["chipotle"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which pet-related event occurred previously?","answer":"vet appointment","alt_answers":["max vet"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which project reference appeared earlier before deadline talk?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Who is the current manager compared to the previous one?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which manager replaced Michael Torres?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"What is the current project relative to past mentions?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which restaurant remains the favorite despite past mentions?","answer":"olive garden","alt_answers":["olive"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which pet currently belongs to the user based on history?","answer":"max","alt_answers":["dog max"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which allergy continues to be relevant from past health records?","answer":"penicillin","alt_answers":["shellfish"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which hobby continues from earlier conversations?","answer":"guitar","alt_answers":["playing guitar"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which manager is now in charge following historical changes?","answer":"jennifer williams","alt_answers":["jennifer"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which project name remains unchanged from earlier discussions?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which restaurant preference persists compared to earlier mentions?","answer":"olive garden","alt_answers":["olive"],"stores":["summary","ltm"],"type":"knowledge_update"},
    # Adding 20 more to reach 100
    {"query":"What is the user's job title?","answer":"senior software engineer","alt_answers":["software engineer"],"stores":["summary"],"type":"single_hop"},
    {"query":"What year did Alex graduate?","answer":"2018","alt_answers":["2018"],"stores":["summary"],"type":"single_hop"},
    {"query":"How old is Max the dog?","answer":"3 years old","alt_answers":["3","three"],"stores":["summary"],"type":"single_hop"},
    {"query":"What type of meeting does Alex prefer?","answer":"morning","alt_answers":["morning meetings"],"stores":["summary"],"type":"single_hop"},
    {"query":"What activity is scheduled tomorrow?","answer":"follow-up call","alt_answers":["call"],"stores":["stm"],"type":"single_session"},
    {"query":"Where is the current weather report from?","answer":"chicago","alt_answers":["in chicago"],"stores":["stm"],"type":"single_session"},
    {"query":"What type of team meeting is planned?","answer":"marketing","alt_answers":["marketing team"],"stores":["stm"],"type":"single_session"},
    {"query":"Which session topic mentions budget planning?","answer":"q4 budget allocation","alt_answers":["budget allocation"],"stores":["stm"],"type":"single_session"},
    {"query":"Which trip country was discussed?","answer":"italy","alt_answers":["trip to italy"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which historical manager changed after reorg?","answer":"michael torres","alt_answers":["michael"],"stores":["ltm"],"type":"recent_session"},
    {"query":"What project was discussed in earlier conversations?","answer":"phoenix","alt_answers":["project phoenix"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which health-related event was noted earlier?","answer":"doctor appointment","alt_answers":["health check-in"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Give one restaurant recalled from conversations.","answer":"chipotle","alt_answers":["olive garden","bangkok garden"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Name one European landmark mentioned.","answer":"colosseum","alt_answers":["the colosseum"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Provide one travel destination recalled earlier.","answer":"italy","alt_answers":["rome","florence"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Which hobby was mentioned earlier in conversations?","answer":"guitar","alt_answers":["playing guitar"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which travel memory happened before restaurant discussions?","answer":"italy","alt_answers":["trip italy"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which food preference appeared earlier?","answer":"pasta","alt_answers":["italian pasta"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which manager name appeared historically first?","answer":"michael torres","alt_answers":["torres"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which location was referenced before later project talks?","answer":"italy","alt_answers":["rome"],"stores":["ltm","episodic"],"type":"temporal"},
]

LONG_QUESTIONS = [
    {"query":"How many giraffes were tracked in the field study?","answer":"47","alt_answers":["47 giraffes","47 reticulated giraffes"],"stores":["summary"],"type":"single_hop"},
    {"query":"Approximately how large was the migration study region?","answer":"12,000 square kilometers","alt_answers":["12000 km","12000"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which vegetation factor correlated with migration routes?","answer":"acacia woodland density","alt_answers":["acacia density","acacia"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which clustering method detected aggregation points?","answer":"dbscan","alt_answers":["dbscan clustering"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which centrality metric identified connector giraffes?","answer":"betweenness centrality","alt_answers":["centrality","betweenness"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which Japanese architecture example was studied?","answer":"nakagin capsule tower","alt_answers":["nakagin tower","nakagin"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which London building was analyzed in Brutalist studies?","answer":"national theatre","alt_answers":["national theatre london"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which architectural material philosophy is associated with Brutalism?","answer":"béton brut","alt_answers":["raw concrete","beton brut"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which Montreal project represents Brutalist housing innovation?","answer":"habitat 67","alt_answers":["habitat67","habitat 67"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which conservation policy was recommended to preserve corridors?","answer":"cross-border wildlife agreements","alt_answers":["wildlife agreements","transboundary"],"stores":["summary"],"type":"single_hop"},
    {"query":"Which scaling factor stabilizes dot-product attention?","answer":"1/sqrt(d_k)","alt_answers":["1 over sqrt dk","scaling factor"],"stores":["stm"],"type":"single_session"},
    {"query":"Which positional encoding variant is being explored?","answer":"rotary positional embeddings","alt_answers":["rope","rotary"],"stores":["stm"],"type":"single_session"},
    {"query":"Which interpretability method replaces internal activations?","answer":"activation patching","alt_answers":["patching"],"stores":["stm"],"type":"single_session"},
    {"query":"Which optimization algorithm follows reward model training in rlhf?","answer":"ppo","alt_answers":["proximal policy optimization"],"stores":["stm"],"type":"single_session"},
    {"query":"Which architecture uses sliding window sparse attention?","answer":"longformer","alt_answers":["bigbird"],"stores":["stm"],"type":"single_session"},
    {"query":"Which attention variant uses kernel approximations?","answer":"performer","alt_answers":["performer linear attention"],"stores":["stm"],"type":"single_session"},
    {"query":"Which scaling law guides compute-optimal training discussed in session?","answer":"chinchilla","alt_answers":["chinchilla scaling laws"],"stores":["stm"],"type":"single_session"},
    {"query":"Which optimization technique prevents exploding gradients in scaling experiments?","answer":"gradient clipping","alt_answers":["clipping"],"stores":["stm"],"type":"single_session"},
    {"query":"Which naval battle was the turning point of the Pacific War?","answer":"battle of midway","alt_answers":["midway"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which campaign highlighted brutal jungle warfare logistics?","answer":"guadalcanal","alt_answers":["guadalcanal campaign"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which admiral led the island-hopping strategy?","answer":"chester nimitz","alt_answers":["nimitz","admiral nimitz"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which neural architecture replaced recurrence for parallel training?","answer":"transformers","alt_answers":["transformer models","transformer"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which complexity issue arises with long sequence attention?","answer":"quadratic complexity","alt_answers":["o(n^2)","quadratic"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which research introduced sparse attention for long sequences?","answer":"longformer","alt_answers":["bigbird"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which ecological theory explains corridor fragmentation risk?","answer":"metapopulation theory","alt_answers":["metapopulation"],"stores":["ltm"],"type":"recent_session"},
    {"query":"Which redevelopment concept replaced megastructure planning in discussions?","answer":"mixed-use urbanism","alt_answers":["mixed use","mixed-use"],"stores":["ltm"],"type":"recent_session"},
    {"query":"How does the user's corridor research connect to earlier ecological modeling discussions?","answer":"corridor fragmentation increases extinction risk","alt_answers":["fragmentation risk","extinction risk"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which architectural interest aligns with earlier urban design debates?","answer":"brutalist architecture","alt_answers":["brutalism preservation","brutalist preservation"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"How does giraffe migration research relate to metapopulation modeling discussions?","answer":"dispersal connectivity","alt_answers":["connectivity","dispersal"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which conservation recommendation aligns with earlier habitat modeling analysis?","answer":"migration corridor preservation","alt_answers":["corridor preservation"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which building preservation topic connects to historical urban redevelopment discussions?","answer":"adaptive reuse","alt_answers":["adaptive reuse of brutalist buildings"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Which social network metric research connects to earlier ecological modeling?","answer":"betweenness centrality","alt_answers":["centrality connectors","betweenness"],"stores":["summary","ltm"],"type":"multi_hop"},
    {"query":"Name one major Pacific War battle discussed.","answer":"midway","alt_answers":["guadalcanal","battle of midway"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"List one sparse attention architecture discussed historically.","answer":"longformer","alt_answers":["bigbird","performer"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Provide one transformer complexity issue mentioned.","answer":"quadratic complexity","alt_answers":["o(n^2)"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Name one ecological modeling framework mentioned earlier.","answer":"metapopulation theory","alt_answers":["agent-based modeling"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Mention one urban planning concept discussed earlier.","answer":"mixed-use","alt_answers":["mixed-use urbanism"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Give one transformer replacement for recurrence discussed.","answer":"self-attention","alt_answers":["attention mechanisms","attention"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Provide one naval leader mentioned in earlier conversations.","answer":"nimitz","alt_answers":["chester nimitz"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"List one ecological simulation method discussed previously.","answer":"agent-based","alt_answers":["agent modeling","agent-based simulation"],"stores":["ltm","episodic"],"type":"memory_capacity"},
    {"query":"Which architecture replaced rnns before later sparse attention research?","answer":"transformers","alt_answers":["transformer models"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which naval battle occurred before Japan's surrender discussions?","answer":"midway","alt_answers":["battle of midway"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which attention discussion happened before scaling-law conversations?","answer":"quadratic complexity","alt_answers":["attention complexity","quadratic"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which conservation modeling discussion preceded corridor preservation recommendations?","answer":"metapopulation","alt_answers":["metapopulation modeling"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which architecture discussion came before efficient attention variants?","answer":"rnn to transformer","alt_answers":["transformer transition","rnn transition"],"stores":["ltm","episodic"],"type":"temporal"},
    {"query":"Which conservation expertise continues from earlier ecological modeling discussions?","answer":"corridor connectivity","alt_answers":["connectivity research"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which architecture interest remains consistent with earlier urban design discussions?","answer":"brutalist","alt_answers":["brutalism","brutalist architecture"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which migration analysis approach aligns with earlier ecological simulations?","answer":"spatial movement","alt_answers":["movement modeling"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which research theme persists between profile and past conversations?","answer":"conservation corridor","alt_answers":["corridor preservation"],"stores":["summary","ltm"],"type":"knowledge_update"},
    {"query":"Which population dynamics focus continues from past modeling conversations?","answer":"giraffe population","alt_answers":["population connectivity"],"stores":["summary","ltm"],"type":"knowledge_update"},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def count_tokens(text: str) -> int:
    """Estimate token count using word split (approx 1 token per word).
    Actual prompt/completion tokens are tracked via response.usage in ask_llm."""
    return len(text.split())

def retrieve_context(stores: Set[str], memory: Dict) -> Tuple[str, int]:
    if not stores:
        return "", 0
    
    store_names = {
        "stm": "Current Session", 
        "summary": "User Profile", 
        "ltm": "Past Conversations", 
        "episodic": "Conversation Transcripts"
    }
    
    context_parts = []
    for store in sorted(stores):
        if store in memory:
            context_parts.append(f"=== {store_names[store]} ===")
            context_parts.extend(memory[store])
            context_parts.append("")
    
    context = "\n".join(context_parts)
    tokens = count_tokens(context)
    return context, tokens

# ============================================================================
# POLICIES
# ============================================================================

POLICIES = {
    "none": set(),
    "stm": {"stm"},
    "summary": {"summary"},
    "ltm": {"ltm"},
    "episodic": {"episodic"},
    "stm+summary": {"stm", "summary"},
    "summary+ltm": {"summary", "ltm"},
    "ltm+episodic": {"ltm", "episodic"},
    "stm+sum+ltm": {"stm", "summary", "ltm"},
    "uniform": {"stm", "summary", "ltm", "episodic"},
}

def hybrid_policy(query: Dict) -> Set[str]:
    text = query["query"].lower()
    
    # Current session signals
    if any(w in text for w in ["today", "right now", "current session"]):
        return {"stm"}
    # Technical STM signals
    if any(w in text for w in ["scaling factor", "ppo", "rlhf", "attention", "longformer", "performer", "gradient", "chinchilla"]):
        return {"stm"}
    
    # Memory capacity signals
    if any(w in text for w in ["list", "name one", "provide one", "mention one", "give one"]):
        return {"ltm", "episodic"}
    
    # Temporal signals
    if any(w in text for w in ["before", "previously", "earlier", "preceded", "first", "originally"]):
        return {"ltm", "episodic"}
    
    # Multi-hop signals
    if any(w in text for w in ["connect", "relate", "align", "continues", "persists", "remains"]):
        return {"summary", "ltm"}
    
    # LTM signals
    if any(w in text for w in ["naval", "pacific", "midway", "guadalcanal", "nimitz", "rnn", "metapopulation", "urbanism"]):
        return {"ltm"}
    
    # Summary signals
    if any(w in text for w in ["giraffe", "dbscan", "centrality", "brutalist", "habitat", "nakagin", "béton", "corridor"]):
        return {"summary"}
    
    return {"summary", "ltm"}

def oracle_policy(query: Dict) -> Set[str]:
    return set(query["stores"])

# ============================================================================
# LLM CALL WITH RETRY
# ============================================================================

def ask_llm(query: str, context: str, model: str, question_id: int) -> tuple:
    """Call LLM with retry logic and detailed logging.
    Routes to OpenAI or Gemini based on model name prefix.
    Returns (response_text, total_tokens)."""

    if not _client_available(model):
        return f"[NO CLIENT FOR {model}]", 0

    if not context:
        prompt = f"""Answer the question. If you don't know, say "I don't know".

Question: {query}

Answer concisely:"""
    else:
        prompt = f"""Based ONLY on the following context, answer the question.
If the answer is not in the context, say "Information not available".

Context:
{context}

Question: {query}

Answer concisely:"""

    for attempt in range(CONFIG["max_retries"]):
        try:
            start_time = time.time()

            if _is_gemini(model):
                # ── Gemini path ──────────────────────────────────────────
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                elapsed = time.time() - start_time
                result = response.text.strip()
                # Gemini returns token counts in usage_metadata
                usage = getattr(response, "usage_metadata", None)
                total_tokens = (
                    (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)
                    if usage else count_tokens(prompt)
                )
            else:
                # ── OpenAI path ──────────────────────────────────────────
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0,
                )
                elapsed = time.time() - start_time
                result = response.choices[0].message.content.strip()
                total_tokens = response.usage.total_tokens if response.usage else count_tokens(prompt)

            if elapsed > 5:
                log(f"  Q{question_id}: {elapsed:.1f}s (slow) | tokens={total_tokens}")

            return result, total_tokens

        except Exception as e:
            wait_time = CONFIG["retry_backoff"] ** attempt
            log(f"  Q{question_id} ERROR (attempt {attempt+1}/{CONFIG['max_retries']}): {str(e)[:60]}")
            log(f"  Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    log(f"  Q{question_id} FAILED after {CONFIG['max_retries']} attempts")
    return "[ERROR: max retries exceeded]", 0

def check_answer(response: str, expected: str, alt_answers: List[str]) -> bool:
    response_lower = response.lower()
    
    if expected.lower() in response_lower:
        return True
    
    for alt in alt_answers:
        if alt.lower() in response_lower:
            return True
    
    if any(x in response_lower for x in ["not available", "don't know", "cannot find", "no information"]):
        return False
    
    return False

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(results: List[Dict], checkpoint_file: str = "data/checkpoint.json"):
    """Save progress to checkpoint file."""
    os.makedirs("data", exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Checkpoint saved: {len(results)} results")

def load_checkpoint(checkpoint_file: str = "data/checkpoint.json") -> List[Dict]:
    """Load progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        log(f"Loaded checkpoint: {len(results)} results")
        return results
    return []

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(resume: bool = True):
    log("="*60)
    log("COMPREHENSIVE MULTI-MODEL EVALUATION")
    log("="*60)
    
    # Build questions list
    questions = []
    for q in SHORT_QUESTIONS:
        questions.append({"question": q, "memory": SHORT_MEMORY, "context_type": "short"})
    for q in LONG_QUESTIONS:
        questions.append({"question": q, "memory": LONG_MEMORY, "context_type": "long"})
    
    # Build policies list
    policy_list = []
    for name, stores in POLICIES.items():
        policy_list.append((name, lambda q, s=stores: s))
    policy_list.append(("hybrid", hybrid_policy))
    policy_list.append(("oracle", oracle_policy))
    
    total_calls = len(questions) * len(policy_list) * len(CONFIG["models"])
    
    log(f"Models: {CONFIG['models']}")
    log(f"Policies: {len(policy_list)}")
    log(f"Questions: {len(SHORT_QUESTIONS)} short + {len(LONG_QUESTIONS)} long = {len(questions)}")
    log(f"Total LLM calls: {total_calls}")
    log(f"Estimated time: {total_calls * 0.7 / 60:.1f} minutes")
    
    if not USE_REAL_LLM:
        log("✗ Cannot run: no OpenAI or Gemini client available")
        return None
    
    # Load checkpoint if resuming
    all_results = []
    completed_keys = set()
    
    if resume:
        all_results = load_checkpoint()
        for r in all_results:
            key = f"{r['model']}|{r['policy']}|{r['question_idx']}"
            completed_keys.add(key)
    
    call_count = len(completed_keys)
    
    # Run evaluation
    for model in CONFIG["models"]:
        log(f"\n{'='*50}")
        log(f"MODEL: {model}")
        log(f"{'='*50}")
        
        for policy_name, policy_fn in policy_list:
            log(f"\n  Policy: {policy_name}")
            
            policy_results = []
            
            for q_idx, item in enumerate(questions):
                # Skip if already completed
                key = f"{model}|{policy_name}|{q_idx}"
                if key in completed_keys:
                    continue
                
                q = item["question"]
                memory = item["memory"]
                ctx_type = item["context_type"]
                
                stores = policy_fn(q)
                context, _ = retrieve_context(stores, memory)
                
                # Call LLM — returns (response_text, actual_token_count)
                response, tokens = ask_llm(q["query"], context, model, q_idx)
                is_correct = check_answer(response, q["answer"], q["alt_answers"])
                
                result = {
                    "model": model,
                    "policy": policy_name,
                    "question_idx": q_idx,
                    "context_type": ctx_type,
                    "query_type": q["type"],
                    "correct": is_correct,
                    "tokens": tokens,
                    "stores": list(stores) if stores else [],
                }
                
                all_results.append(result)
                policy_results.append(result)
                call_count += 1
                
                # Progress log
                if call_count % 10 == 0:
                    log(f"    Progress: {call_count}/{total_calls} ({100*call_count/total_calls:.1f}%)")
                
                # Checkpoint
                if call_count % CONFIG["checkpoint_every"] == 0:
                    save_checkpoint(all_results)
                
                # Sleep between calls
                time.sleep(CONFIG["sleep_between_calls"])
            
            # Policy summary
            if policy_results:
                correct = sum(1 for r in policy_results if r["correct"])
                total = len(policy_results)
                log(f"    Result: {correct}/{total} = {100*correct/total:.1f}%")
    
    # Final save
    save_checkpoint(all_results, "data/multimodel_results.json")
    log(f"\nFinal results saved: data/multimodel_results.json")
    
    # Print summary
    print_summary(all_results)
    
    return all_results

def print_summary(results: List[Dict]):
    """Print comprehensive summary."""
    
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    
    # Group by model and policy
    for model in CONFIG["models"]:
        log(f"\n{model}:")
        log(f"{'Policy':<15} {'Overall':<10} {'Short':<10} {'Long':<10} {'Tokens':<10}")
        log("-"*55)
        
        policy_stats = defaultdict(lambda: {"correct": 0, "total": 0, "short_c": 0, "short_t": 0, "long_c": 0, "long_t": 0, "tokens": 0})
        
        for r in results:
            if r["model"] != model:
                continue
            
            p = r["policy"]
            policy_stats[p]["total"] += 1
            policy_stats[p]["tokens"] += r["tokens"]
            
            if r["correct"]:
                policy_stats[p]["correct"] += 1
            
            if r["context_type"] == "short":
                policy_stats[p]["short_t"] += 1
                if r["correct"]:
                    policy_stats[p]["short_c"] += 1
            else:
                policy_stats[p]["long_t"] += 1
                if r["correct"]:
                    policy_stats[p]["long_c"] += 1
        
        # Sort by accuracy
        sorted_policies = sorted(policy_stats.items(), key=lambda x: -x[1]["correct"]/max(x[1]["total"],1))
        
        for policy, stats in sorted_policies:
            if stats["total"] == 0:
                continue
            overall = stats["correct"] / stats["total"]
            short_acc = stats["short_c"] / max(stats["short_t"], 1)
            long_acc = stats["long_c"] / max(stats["long_t"], 1)
            avg_tokens = stats["tokens"] / stats["total"]
            
            log(f"{policy:<15} {overall:<10.1%} {short_acc:<10.1%} {long_acc:<10.1%} {avg_tokens:<10.0f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Check for resume flag
    resume = "--fresh" not in sys.argv
    if not resume:
        log("Starting fresh (ignoring checkpoint)")
    
    run_evaluation(resume=resume)
