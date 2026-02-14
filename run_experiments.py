"""
Multi-Store Routing Experiments
===============================

Complete experiment code for the paper:
"Learning Where to Retrieve: Multi-Store Routing for Agent Memory"

Usage:
    python run_experiments.py --all          # Run everything
    python run_experiments.py --generate     # Generate dataset only
    python run_experiments.py --train        # Train policies only
    python run_experiments.py --ablation     # Run ablation studies only
    python run_experiments.py --visualize    # Generate figures only

"""

import json
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

###############################################################################
# PART 1: DATA STRUCTURES
###############################################################################

@dataclass
class Fact:
    topic: str
    fact: str
    criticality: str
    mentioned_in: List[str] = field(default_factory=list)

@dataclass
class Turn:
    role: str
    text: str
    turn_id: str
    entities: List[str] = field(default_factory=list)
    is_key_turn: bool = False

@dataclass
class Session:
    session_id: str
    user_id: str
    date: str
    turns: List[Turn] = field(default_factory=list)
    facts_mentioned: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)

@dataclass
class UserProfile:
    user_id: str
    name: str
    facts: List[Fact] = field(default_factory=list)
    critical_topics: List[str] = field(default_factory=list)

@dataclass 
class Query:
    query_id: str
    user_id: str
    session_id: str
    turn_index: int
    text: str
    query_type: str
    ground_truth_stores: List[str]
    ground_truth_answer: str
    relevant_memory_ids: Dict[str, List[str]]
    criticality: str
    has_anaphora: bool
    has_possessive: bool
    references_past: bool
    entities_mentioned: List[str]

###############################################################################
# PART 2: DATA GENERATION
###############################################################################

# Templates and data for generation
FIRST_NAMES = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "James", "Ashley",
    "Robert", "Amanda", "William", "Stephanie", "Daniel", "Nicole", "Christopher",
    "Elizabeth", "Matthew", "Jennifer", "Anthony", "Melissa", "Joseph", "Rebecca",
    "Andrew", "Laura", "Ryan", "Michelle", "Kevin", "Kimberly", "Brian", "Lisa",
    "Mark", "Angela", "Steven", "Heather", "Paul", "Amy", "George", "Anna",
    "Edward", "Rachel", "Peter", "Katherine", "Thomas", "Christine", "Charles",
    "Samantha", "Frank", "Deborah", "Henry", "Catherine"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
    "Martin", "Lee", "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson",
    "Walker", "Hall", "Young", "King", "Wright", "Scott", "Green", "Baker",
    "Adams", "Nelson", "Hill", "Campbell", "Mitchell", "Roberts", "Carter"]

DOCTORS = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Chen", "Dr. Patel",
    "Dr. Kim", "Dr. Garcia", "Dr. Brown", "Dr. Lee", "Dr. Wilson"]

COMPANIES = ["Acme Corp", "TechStart Inc", "Global Solutions", "Innovate Labs", 
    "DataDrive", "CloudFirst", "MegaSoft", "StartupX", "Enterprise Co"]

ALLERGIES = ["penicillin", "peanuts", "shellfish", "latex", "bee stings",
    "sulfa drugs", "dairy", "gluten", "eggs", "soy"]

HEALTH_CONDITIONS = ["knee pain", "back pain", "high blood pressure", "diabetes",
    "anxiety", "insomnia", "migraines", "arthritis", "asthma"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9am", "10am", "11am", "2pm", "3pm", "4pm"]
HOBBIES = ["hiking", "photography", "cooking", "reading", "gardening",
    "painting", "yoga", "cycling", "swimming", "chess"]

PETS = [("dog", ["Max", "Buddy", "Charlie", "Cooper", "Rocky"]),
        ("cat", ["Luna", "Milo", "Oliver", "Leo", "Bella"])]

# Conversation templates
CONVERSATION_TEMPLATES = {
    "health_appointment": [
        ("user", "I need to schedule a doctor's appointment"),
        ("assistant", "I can help with that. What's this regarding?"),
        ("user", "My {condition} has been acting up again"),
        ("assistant", "I see you've been seeing {doctor} for {condition}. Would you like to schedule with them?"),
        ("user", "Yes, that would be great"),
        ("assistant", "I have availability on {day} at {time}. Does that work?"),
        ("user", "That works for me"),
        ("assistant", "Great, I've scheduled your appointment with {doctor} for {day} at {time}"),
    ],
    "health_medication": [
        ("user", "I have a question about my medication"),
        ("assistant", "Of course. I see you're taking {medication}. What's your question?"),
        ("user", "I've been having some side effects"),
        ("assistant", "I'm sorry to hear that. What side effects are you experiencing?"),
        ("user", "Some dizziness and nausea"),
        ("assistant", "Those can be common with {medication}. You should discuss this with {doctor}"),
        ("user", "When is my next appointment?"),
        ("assistant", "Your next appointment is on {day} at {time}"),
    ],
    "work_schedule": [
        ("user", "I need to check my work schedule"),
        ("assistant", "Sure. You work at {company}, right?"),
        ("user", "Yes, I need to know if I have any conflicts this week"),
        ("assistant", "Looking at your schedule, you have a {appt_type} appointment on {day}"),
        ("user", "That might conflict with a meeting"),
        ("assistant", "Would you like me to help reschedule?"),
    ],
    "personal_chat": [
        ("user", "How's your day going?"),
        ("assistant", "I'm doing well! How about you?"),
        ("user", "Pretty good. Just got back from {hobby}"),
        ("assistant", "That sounds nice! You mentioned you enjoy {hobby}. How was it?"),
        ("user", "It was great, very relaxing"),
        ("assistant", "Glad to hear it!"),
    ],
}

# Query templates
QUERY_TEMPLATES = {
    "no_retrieval": [
        {"text": "Hello!", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "Hi there!", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "Thanks!", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "Got it, thanks", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "Sounds good", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "What's the weather like?", "has_anaphora": False, "has_possessive": False, "refs_past": False},
    ],
    "stm": [
        {"text": "What did you just say?", "has_anaphora": True, "has_possessive": False, "refs_past": False},
        {"text": "Can you repeat that?", "has_anaphora": True, "has_possessive": False, "refs_past": False},
        {"text": "Tell me more about that", "has_anaphora": True, "has_possessive": False, "refs_past": False},
        {"text": "Why?", "has_anaphora": False, "has_possessive": False, "refs_past": False},
        {"text": "And then what?", "has_anaphora": False, "has_possessive": False, "refs_past": False},
    ],
    "summary": [
        {"text": "What's my doctor's name?", "has_anaphora": False, "has_possessive": True, "refs_past": False, "fact_topic": "health"},
        {"text": "Am I allergic to anything?", "has_anaphora": False, "has_possessive": False, "refs_past": False, "fact_topic": "health"},
        {"text": "Where do I work?", "has_anaphora": False, "has_possessive": False, "refs_past": False, "fact_topic": "work"},
        {"text": "What's my pet's name?", "has_anaphora": False, "has_possessive": True, "refs_past": False, "fact_topic": "personal"},
        {"text": "When is my next appointment?", "has_anaphora": False, "has_possessive": True, "refs_past": False, "fact_topic": "schedule"},
    ],
    "stm_summary": [
        {"text": "Tell me more about that appointment", "has_anaphora": True, "has_possessive": False, "refs_past": False},
        {"text": "Is that the same doctor I usually see?", "has_anaphora": True, "has_possessive": False, "refs_past": False},
        {"text": "Does that conflict with my schedule?", "has_anaphora": True, "has_possessive": True, "refs_past": False},
    ],
    "ltm": [
        {"text": "What did we discuss last time?", "has_anaphora": False, "has_possessive": False, "refs_past": True},
        {"text": "What was the outcome of my previous visit?", "has_anaphora": False, "has_possessive": True, "refs_past": True},
        {"text": "Did I mention any issues last time?", "has_anaphora": False, "has_possessive": False, "refs_past": True},
        {"text": "What did you recommend before?", "has_anaphora": False, "has_possessive": False, "refs_past": True},
    ],
    "summary_ltm": [
        {"text": "Has my condition changed since last visit?", "has_anaphora": False, "has_possessive": True, "refs_past": True},
        {"text": "Is this the same issue I had before?", "has_anaphora": False, "has_possessive": False, "refs_past": True},
        {"text": "Compare my current symptoms to last time", "has_anaphora": False, "has_possessive": True, "refs_past": True},
    ],
    "all": [
        {"text": "Give me a complete summary of my health situation", "has_anaphora": False, "has_possessive": True, "refs_past": True},
        {"text": "What do you know about me?", "has_anaphora": False, "has_possessive": False, "refs_past": True},
        {"text": "Summarize all my information", "has_anaphora": False, "has_possessive": True, "refs_past": True},
    ],
}

QUERY_TYPE_TO_STORES = {
    "no_retrieval": [],
    "stm": ["stm"],
    "summary": ["summary"],
    "stm_summary": ["stm", "summary"],
    "ltm": ["ltm"],
    "summary_ltm": ["summary", "ltm"],
    "all": ["stm", "summary", "ltm", "episodic"],
}

QUERY_TYPE_DISTRIBUTION = {
    "no_retrieval": 0.15,
    "stm": 0.15,
    "summary": 0.25,
    "stm_summary": 0.15,
    "ltm": 0.15,
    "summary_ltm": 0.10,
    "all": 0.05,
}


def generate_user_profile(user_idx: int) -> UserProfile:
    """Generate a user profile with facts."""
    user_id = f"user_{user_idx:03d}"
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    facts = []
    critical_topics = []
    
    # Health facts
    if random.random() < 0.7:
        facts.append(Fact("health", f"Allergic to {random.choice(ALLERGIES)}", "HIGH"))
        critical_topics.append("health")
    
    if random.random() < 0.8:
        doctor = random.choice(DOCTORS)
        condition = random.choice(HEALTH_CONDITIONS)
        facts.append(Fact("health", f"Sees {doctor} for {condition}", "HIGH"))
    
    # Schedule facts
    if random.random() < 0.8:
        day, time = random.choice(WEEKDAYS), random.choice(TIMES)
        facts.append(Fact("schedule", f"Doctor appointment on {day} at {time}", "HIGH"))
        critical_topics.append("schedule")
    
    # Work facts
    company = random.choice(COMPANIES)
    facts.append(Fact("work", f"Works at {company}", "MEDIUM"))
    
    # Personal facts
    if random.random() < 0.6:
        pet_type, pet_names = random.choice(PETS)
        facts.append(Fact("personal", f"Has a {pet_type} named {random.choice(pet_names)}", "LOW"))
    
    if random.random() < 0.5:
        facts.append(Fact("personal", f"Enjoys {random.choice(HOBBIES)}", "LOW"))
    
    return UserProfile(user_id, name, facts, critical_topics)


def fill_template(template: List[tuple], user: UserProfile) -> List[Turn]:
    """Fill conversation template with user facts."""
    facts_dict = {}
    for fact in user.facts:
        if "Sees " in fact.fact:
            parts = fact.fact.split(" for ")
            facts_dict["doctor"] = parts[0].replace("Sees ", "")
            facts_dict["condition"] = parts[1] if len(parts) > 1 else "checkup"
        elif "Works at" in fact.fact:
            facts_dict["company"] = fact.fact.split(" at ")[-1]
        elif "Has a " in fact.fact and "named" in fact.fact:
            facts_dict["pet_name"] = fact.fact.split(" named ")[-1]
        elif "Enjoys " in fact.fact:
            facts_dict["hobby"] = fact.fact.replace("Enjoys ", "")
        elif "appointment on" in fact.fact:
            parts = fact.fact.split(" on ")[-1].split(" at ")
            facts_dict["day"] = parts[0]
            facts_dict["time"] = parts[1] if len(parts) > 1 else "10am"
            facts_dict["appt_type"] = "doctor"
    
    # Defaults
    defaults = {"doctor": random.choice(DOCTORS), "condition": random.choice(HEALTH_CONDITIONS),
                "medication": "ibuprofen", "company": random.choice(COMPANIES), "pet_name": "Buddy",
                "hobby": random.choice(HOBBIES), "day": random.choice(WEEKDAYS), 
                "time": random.choice(TIMES), "appt_type": "doctor"}
    for k, v in defaults.items():
        facts_dict.setdefault(k, v)
    
    turns = []
    for i, (role, text) in enumerate(template):
        filled = text.format(**facts_dict)
        turns.append(Turn(role, filled, f"turn_{i:02d}", [], i in [3, 4, 5]))
    return turns


def generate_session(user: UserProfile, session_idx: int, base_date: datetime) -> Session:
    """Generate a conversation session."""
    session_id = f"{user.user_id}_session_{session_idx:02d}"
    date = (base_date + timedelta(days=session_idx * random.randint(3, 14))).strftime("%Y-%m-%d")
    
    templates = random.sample(list(CONVERSATION_TEMPLATES.keys()), random.randint(1, 2))
    turns = []
    for tname in templates:
        t = fill_template(CONVERSATION_TEMPLATES[tname], user)
        for turn in t:
            turn.turn_id = f"{session_id}_{turn.turn_id}"
        turns.extend(t)
    
    return Session(session_id, user.user_id, date, turns, [], [t.split("_")[0] for t in templates])


def populate_stores(user: UserProfile, sessions: List[Session], current_idx: int) -> Dict:
    """Populate memory stores for a user at a point in time."""
    current = sessions[current_idx]
    past = sessions[:current_idx]
    
    stm = [{"id": t.turn_id, "role": t.role, "text": t.text} for t in current.turns[-5:]]
    summary = [{"topic": f.topic, "fact": f.fact, "criticality": f.criticality} for f in user.facts]
    ltm = [{"id": t.turn_id, "session_id": s.session_id, "date": s.date, "text": t.text}
           for s in past for t in s.turns if t.is_key_turn]
    episodic = [{"session_id": s.session_id, "date": s.date, 
                 "turns": [{"role": t.role, "text": t.text} for t in s.turns]} for s in past]
    
    return {"stm": stm, "summary": summary, "ltm": ltm, "episodic": episodic}


def generate_queries(user: UserProfile, sessions: List[Session], stores: Dict, n: int = 30) -> List[Query]:
    """Generate test queries for a user."""
    queries = []
    idx = 0
    
    for qtype, prob in QUERY_TYPE_DISTRIBUTION.items():
        count = max(1, int(n * prob))
        templates = QUERY_TEMPLATES[qtype]
        
        for _ in range(count):
            t = random.choice(templates)
            crit = "HIGH" if qtype in ["summary", "summary_ltm", "all"] else "MEDIUM" if qtype == "ltm" else "LOW"
            
            queries.append(Query(
                f"{user.user_id}_query_{idx:03d}", user.user_id, sessions[-1].session_id,
                len(sessions[-1].turns), t["text"], qtype, QUERY_TYPE_TO_STORES[qtype],
                "Answer placeholder", {s: [] for s in ["stm", "summary", "ltm", "episodic"]},
                crit, t["has_anaphora"], t["has_possessive"], t["refs_past"], []
            ))
            idx += 1
    return queries


def generate_dataset(num_users: int = 50, sessions_per_user: int = 4, 
                     queries_per_user: int = 30, output_dir: str = "data"):
    """Generate complete dataset."""
    os.makedirs(output_dir, exist_ok=True)
    for sub in ["users", "sessions", "stores", "queries"]:
        os.makedirs(f"{output_dir}/{sub}", exist_ok=True)
    
    all_users, all_sessions, all_queries, all_stores = [], [], [], []
    base_date = datetime(2024, 1, 1)
    
    print(f"Generating dataset: {num_users} users...")
    for i in range(num_users):
        user = generate_user_profile(i)
        all_users.append(user)
        
        sessions = [generate_session(user, j, base_date) for j in range(sessions_per_user)]
        all_sessions.extend(sessions)
        
        stores = populate_stores(user, sessions, len(sessions) - 1)
        all_stores.append({"user_id": user.user_id, "stores": stores})
        
        queries = generate_queries(user, sessions, stores, queries_per_user)
        all_queries.extend(queries)
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_users} users")
    
    # Save data
    with open(f"{output_dir}/users/all_users.json", "w") as f:
        json.dump([asdict(u) for u in all_users], f, indent=2)
    with open(f"{output_dir}/sessions/all_sessions.json", "w") as f:
        json.dump([asdict(s) for s in all_sessions], f, indent=2)
    with open(f"{output_dir}/stores/all_stores.json", "w") as f:
        json.dump(all_stores, f, indent=2)
    with open(f"{output_dir}/queries/all_queries.json", "w") as f:
        json.dump([asdict(q) for q in all_queries], f, indent=2)
    
    # Create splits
    random.shuffle(all_users)
    train_users = all_users[:30]
    val_users = all_users[30:40]
    test_users = all_users[40:]
    
    def get_queries(users):
        ids = {u.user_id for u in users}
        return [asdict(q) for q in all_queries if q.user_id in ids]
    
    splits = {
        "train": {"users": [u.user_id for u in train_users], "queries": get_queries(train_users)},
        "val": {"users": [u.user_id for u in val_users], "queries": get_queries(val_users)},
        "test": {"users": [u.user_id for u in test_users], "queries": get_queries(test_users)},
    }
    with open(f"{output_dir}/splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}/")
    print(f"  Users: {len(all_users)}")
    print(f"  Sessions: {len(all_sessions)}")
    print(f"  Queries: {len(all_queries)} (train: {len(splits['train']['queries'])}, "
          f"val: {len(splits['val']['queries'])}, test: {len(splits['test']['queries'])})")
    
    return {"users": all_users, "sessions": all_sessions, "stores": all_stores, 
            "queries": all_queries, "splits": splits}


###############################################################################
# PART 3: ROUTING POLICIES
###############################################################################

# Action space
ACTIONS = {
    0: {"stores": [], "k": 0, "name": "no_retrieval"},
    1: {"stores": ["stm"], "k": 2, "name": "stm_only"},
    2: {"stores": ["summary"], "k": 3, "name": "summary_only"},
    3: {"stores": ["stm", "summary"], "k": 5, "name": "stm_summary"},
    4: {"stores": ["ltm"], "k": 5, "name": "ltm_only"},
    5: {"stores": ["summary", "ltm"], "k": 8, "name": "summary_ltm"},
    6: {"stores": ["stm", "summary", "ltm", "episodic"], "k": 10, "name": "all_stores"},
}
NUM_ACTIONS = len(ACTIONS)
LAMBDA = 0.05  # Efficiency penalty

# Feature extraction patterns
ANAPHORA = [r"\bthat\b", r"\bthis\b", r"\bit\b", r"\bthose\b", r"\bthese\b"]
POSSESSIVE = [r"\bmy\b", r"\bmine\b", r"\bour\b"]
PAST_REF = [r"\blast time\b", r"\bbefore\b", r"\bprevious\b", r"\bearlier\b", r"\blast\b"]
GREETING = [r"^hello\b", r"^hi\b", r"^hey\b", r"^thanks\b", r"^ok\b", r"^okay\b"]


def extract_features(query: Dict, user_ctx: Dict = None, session_ctx: List = None) -> np.ndarray:
    """Extract feature vector from query and context."""
    text = query.get("text", "").lower()
    f = []
    
    # Query features
    f.append(1 if any(re.search(p, text) for p in ANAPHORA) else 0)
    f.append(1 if any(re.search(p, text) for p in POSSESSIVE) else 0)
    f.append(1 if any(re.search(p, text) for p in PAST_REF) else 0)
    f.append(1 if any(re.search(p, text, re.I) for p in GREETING) else 0)
    f.append(1 if "?" in text else 0)
    f.append(len(text.split()))
    f.append(len(query.get("entities_mentioned", [])))
    
    # User context
    if user_ctx:
        summary = user_ctx.get("summary", [])
        f.append(len(summary))
        f.append(sum(1 for s in summary if s.get("criticality") == "HIGH"))
        f.append(1 if any(w in text for s in summary for w in s.get("fact", "").lower().split() if len(w) > 3) else 0)
    else:
        f.extend([0, 0, 0])
    
    # Session context
    if session_ctx:
        f.append(len(session_ctx))
        f.append(1 if any(w in t.get("text", "").lower() for t in session_ctx[-3:] for w in text.split() if len(w) > 3) else 0)
    else:
        f.extend([0, 0])
    
    return np.array(f, dtype=np.float32)


class UniformPolicy:
    """Always retrieve from all stores."""
    name = "uniform"
    def select_action(self, *args): return 6
    def update(self, *args): pass


class RuleBasedPolicy:
    """Hand-crafted routing rules."""
    name = "rule_based"
    
    def select_action(self, query: Dict, *args) -> int:
        text = query.get("text", "").lower()
        
        if any(re.search(p, text, re.I) for p in GREETING):
            return 0
        
        has_past = any(re.search(p, text) for p in PAST_REF)
        has_poss = any(re.search(p, text) for p in POSSESSIVE)
        has_ana = any(re.search(p, text) for p in ANAPHORA)
        
        if has_past and has_poss: return 5
        if has_past: return 4
        if has_ana and has_poss: return 3
        if has_ana: return 1
        if has_poss: return 2
        return 3
    
    def update(self, *args): pass


class OraclePolicy:
    """Uses ground truth (upper bound)."""
    name = "oracle"
    _map = {"no_retrieval": 0, "stm": 1, "summary": 2, "stm_summary": 3, 
            "ltm": 4, "summary_ltm": 5, "all": 6}
    
    def select_action(self, query: Dict, *args) -> int:
        return self._map.get(query.get("query_type", "stm_summary"), 3)
    
    def update(self, *args): pass


class UCBPolicy:
    """Upper Confidence Bound policy."""
    name = "ucb"
    
    def __init__(self, n_features: int = 12, c: float = 2.0, lr: float = 0.01):
        self.c, self.lr = c, lr
        self.weights = np.zeros((NUM_ACTIONS, n_features))
        self.bias = np.zeros(NUM_ACTIONS)
        self.counts = np.ones(NUM_ACTIONS)
        self.total = NUM_ACTIONS
    
    def select_action(self, query: Dict, user_ctx: Dict = None, session_ctx: List = None) -> int:
        f = extract_features(query, user_ctx, session_ctx)
        q = np.dot(self.weights, f) + self.bias
        ucb = self.c * np.sqrt(np.log(self.total) / self.counts)
        return int(np.argmax(q + ucb))
    
    def update(self, query: Dict, action: int, reward: float, user_ctx: Dict = None, session_ctx: List = None):
        f = extract_features(query, user_ctx, session_ctx)
        error = reward - (np.dot(self.weights[action], f) + self.bias[action])
        self.weights[action] += self.lr * error * f
        self.bias[action] += self.lr * error
        self.counts[action] += 1
        self.total += 1


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy."""
    name = "epsilon_greedy"
    
    def __init__(self, n_features: int = 12, epsilon: float = 0.1, lr: float = 0.01):
        self.epsilon, self.lr = epsilon, lr
        self.weights = np.zeros((NUM_ACTIONS, n_features))
        self.bias = np.zeros(NUM_ACTIONS)
    
    def select_action(self, query: Dict, user_ctx: Dict = None, session_ctx: List = None) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        f = extract_features(query, user_ctx, session_ctx)
        return int(np.argmax(np.dot(self.weights, f) + self.bias))
    
    def update(self, query: Dict, action: int, reward: float, user_ctx: Dict = None, session_ctx: List = None):
        f = extract_features(query, user_ctx, session_ctx)
        error = reward - (np.dot(self.weights[action], f) + self.bias[action])
        self.weights[action] += self.lr * error * f
        self.bias[action] += self.lr * error


class ThompsonSamplingPolicy:
    """Thompson Sampling policy."""
    name = "thompson_sampling"
    
    def __init__(self, n_features: int = 12, prior_var: float = 1.0):
        self.mu = np.zeros((NUM_ACTIONS, n_features))
        self.sigma = np.ones((NUM_ACTIONS, n_features)) * prior_var
        self.obs_var = 0.1
    
    def select_action(self, query: Dict, user_ctx: Dict = None, session_ctx: List = None) -> int:
        f = extract_features(query, user_ctx, session_ctx)
        q = [np.dot(np.random.normal(self.mu[a], np.sqrt(self.sigma[a])), f) for a in range(NUM_ACTIONS)]
        return int(np.argmax(q))
    
    def update(self, query: Dict, action: int, reward: float, user_ctx: Dict = None, session_ctx: List = None):
        f = extract_features(query, user_ctx, session_ctx)
        for i in range(len(f)):
            if abs(f[i]) > 1e-6:
                prec = 1.0 / self.sigma[action, i]
                new_prec = prec + (f[i] ** 2) / self.obs_var
                self.sigma[action, i] = 1.0 / new_prec
                self.mu[action, i] = self.sigma[action, i] * (prec * self.mu[action, i] + f[i] * reward / self.obs_var)


###############################################################################
# PART 4: TRAINING AND EVALUATION
###############################################################################

def compute_reward(action: int, query: Dict, correct: bool, lam: float = LAMBDA) -> float:
    """Compute reward for an action."""
    k = ACTIONS[action]["k"]
    r_c = 1.0 if correct else -1.0
    weight = {"HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}.get(query.get("criticality", "MEDIUM"), 1.0)
    return weight * r_c - lam * k


def check_correctness(action: int, query: Dict) -> bool:
    """Check if action retrieves from correct stores."""
    action_stores = set(ACTIONS[action]["stores"])
    gt_stores = set(query.get("ground_truth_stores", []))
    return gt_stores.issubset(action_stores) if gt_stores else len(action_stores) == 0


def evaluate_policy(policy, queries: List[Dict], stores_by_user: Dict, lam: float = LAMBDA) -> Dict:
    """Evaluate a routing policy."""
    results = {"total": 0, "correct": 0, "total_k": 0, "reward": 0.0, 
               "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
               "action_dist": defaultdict(int)}
    
    for q in queries:
        user_stores = stores_by_user.get(q["user_id"], {})
        user_ctx = {"summary": user_stores.get("stores", {}).get("summary", [])}
        session_ctx = user_stores.get("stores", {}).get("stm", [])
        
        action = policy.select_action(q, user_ctx, session_ctx)
        correct = check_correctness(action, q)
        reward = compute_reward(action, q, correct, lam)
        
        results["total"] += 1
        results["correct"] += int(correct)
        results["total_k"] += ACTIONS[action]["k"]
        results["reward"] += reward
        results["action_dist"][action] += 1
        results["by_type"][q.get("query_type", "unknown")]["total"] += 1
        results["by_type"][q.get("query_type", "unknown")]["correct"] += int(correct)
    
    n = results["total"]
    if n > 0:
        results["accuracy"] = results["correct"] / n
        results["avg_k"] = results["total_k"] / n
        results["avg_reward"] = results["reward"] / n
    
    for qt in results["by_type"]:
        t, c = results["by_type"][qt]["total"], results["by_type"][qt]["correct"]
        results["by_type"][qt]["accuracy"] = c / t if t > 0 else 0.0
    
    return results


def train_policy(policy, queries: List[Dict], stores_by_user: Dict, 
                 n_epochs: int = 5, lam: float = LAMBDA, verbose: bool = True):
    """Train a policy using contextual bandit updates."""
    for epoch in range(n_epochs):
        random.shuffle(queries)
        total_reward = 0.0
        
        for q in queries:
            user_stores = stores_by_user.get(q["user_id"], {})
            user_ctx = {"summary": user_stores.get("stores", {}).get("summary", [])}
            session_ctx = user_stores.get("stores", {}).get("stm", [])
            
            action = policy.select_action(q, user_ctx, session_ctx)
            correct = check_correctness(action, q)
            reward = compute_reward(action, q, correct, lam)
            policy.update(q, action, reward, user_ctx, session_ctx)
            total_reward += reward
        
        if verbose:
            print(f"  Epoch {epoch+1}/{n_epochs}: avg_reward = {total_reward/len(queries):.3f}")


###############################################################################
# PART 5: EXPERIMENTS
###############################################################################

def run_main_experiment(data_dir: str = "data"):
    """Run main experiment comparing all policies."""
    print("\n" + "="*60)
    print("MAIN EXPERIMENT: Policy Comparison")
    print("="*60)
    
    # Load data
    with open(f"{data_dir}/stores/all_stores.json") as f:
        stores_by_user = {s["user_id"]: s for s in json.load(f)}
    with open(f"{data_dir}/splits.json") as f:
        splits = json.load(f)
    
    train_q = splits["train"]["queries"]
    test_q = splits["test"]["queries"]
    print(f"Train: {len(train_q)}, Test: {len(test_q)}")
    
    # Initialize policies
    policies = {
        "uniform": UniformPolicy(),
        "rule_based": RuleBasedPolicy(),
        "oracle": OraclePolicy(),
        "epsilon_greedy": EpsilonGreedyPolicy(epsilon=0.1),
        "ucb": UCBPolicy(c=0.5),
        "thompson": ThompsonSamplingPolicy(),
    }
    
    # Train learned policies
    print("\nTraining learned policies...")
    for name in ["epsilon_greedy", "ucb", "thompson"]:
        print(f"\n{name}:")
        train_policy(policies[name], train_q.copy(), stores_by_user, n_epochs=5)
    
    # Evaluate all
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"{'Policy':<20} {'Accuracy':<12} {'Avg k':<10} {'Reward':<12} {'Efficiency':<10}")
    print("-"*60)
    
    results = {}
    for name, policy in policies.items():
        r = evaluate_policy(policy, test_q, stores_by_user)
        results[name] = r
        eff = 10.0 / r["avg_k"] if r["avg_k"] > 0 else 0
        print(f"{name:<20} {r['accuracy']:<12.3f} {r['avg_k']:<10.2f} {r['avg_reward']:<12.3f} {eff:<10.2f}x")
    
    # Save results
    with open(f"{data_dir}/experiment_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "by_type"} 
                   for k, v in results.items()}, f, indent=2, default=str)
    
    return results


def run_ablation_lambda(data_dir: str = "data"):
    """Ablation study on lambda (efficiency penalty)."""
    print("\n" + "="*60)
    print("ABLATION: Lambda (Efficiency Penalty)")
    print("="*60)
    
    with open(f"{data_dir}/stores/all_stores.json") as f:
        stores_by_user = {s["user_id"]: s for s in json.load(f)}
    with open(f"{data_dir}/splits.json") as f:
        splits = json.load(f)
    
    train_q, test_q = splits["train"]["queries"], splits["test"]["queries"]
    
    results = []
    for lam in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        policy = UCBPolicy(c=0.5)
        train_policy(policy, train_q.copy(), stores_by_user, n_epochs=5, lam=lam, verbose=False)
        r = evaluate_policy(policy, test_q, stores_by_user, lam=lam)
        results.append({"lambda": lam, "accuracy": r["accuracy"], "avg_k": r["avg_k"]})
        print(f"λ={lam:.2f}: acc={r['accuracy']:.3f}, avg_k={r['avg_k']:.2f}, eff={10/r['avg_k']:.2f}x")
    
    return results


def run_ablation_ucb_c(data_dir: str = "data"):
    """Ablation study on UCB exploration parameter."""
    print("\n" + "="*60)
    print("ABLATION: UCB Exploration (c)")
    print("="*60)
    
    with open(f"{data_dir}/stores/all_stores.json") as f:
        stores_by_user = {s["user_id"]: s for s in json.load(f)}
    with open(f"{data_dir}/splits.json") as f:
        splits = json.load(f)
    
    train_q, test_q = splits["train"]["queries"], splits["test"]["queries"]
    
    results = []
    for c in [0.1, 0.5, 1.0, 2.0, 4.0]:
        policy = UCBPolicy(c=c)
        train_policy(policy, train_q.copy(), stores_by_user, n_epochs=5, verbose=False)
        r = evaluate_policy(policy, test_q, stores_by_user)
        results.append({"c": c, "accuracy": r["accuracy"], "avg_k": r["avg_k"]})
        print(f"c={c:.1f}: acc={r['accuracy']:.3f}, avg_k={r['avg_k']:.2f}")
    
    return results


###############################################################################
# PART 6: VISUALIZATION
###############################################################################

def generate_figures(data_dir: str = "data", output_dir: str = "figures"):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(f"{data_dir}/experiment_results.json") as f:
        results = json.load(f)
    
    # Figure 1: Accuracy vs Efficiency
    fig, ax = plt.subplots(figsize=(8, 6))
    policies = ["uniform", "rule_based", "epsilon_greedy", "ucb", "thompson", "oracle"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["s", "^", "o", "D", "v", "*"]
    
    for i, p in enumerate(policies):
        if p in results:
            acc, k = results[p]["accuracy"], results[p]["avg_k"]
            eff = 10.0 / k
            ax.scatter(eff, acc, s=200, c=colors[i], marker=markers[i], label=p, zorder=5)
    
    ax.set_xlabel("Efficiency (10 / avg_k)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Routing Policy: Accuracy vs Efficiency", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_efficiency.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/accuracy_efficiency.png")
    
    # Figure 2: Reward comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    rewards = [results[p]["avg_reward"] for p in policies if p in results]
    colors = ["#ff7f0e" if p in ["uniform", "rule_based"] else "#1f77b4" if p in ["epsilon_greedy", "ucb", "thompson"] else "#2ca02c" for p in policies]
    ax.bar([p for p in policies if p in results], rewards, color=colors)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title("Reward Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/reward_comparison.png")


def generate_latex_table(data_dir: str = "data") -> str:
    """Generate LaTeX table for paper."""
    with open(f"{data_dir}/experiment_results.json") as f:
        results = json.load(f)
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Routing policy comparison. Efficiency relative to uniform (k=10).}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
Policy & Accuracy & Avg $k$ & Reward & Efficiency \\
\midrule
"""
    
    for p, name in [("uniform", "Uniform"), ("rule_based", "Rule-based"), 
                    ("epsilon_greedy", "$\\epsilon$-Greedy"), ("ucb", "UCB"),
                    ("thompson", "Thompson"), ("oracle", "Oracle")]:
        if p in results:
            r = results[p]
            eff = 10.0 / r["avg_k"]
            latex += f"{name} & {r['accuracy']:.3f} & {r['avg_k']:.2f} & {r['avg_reward']:.3f} & {eff:.2f}$\\times$ \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Multi-Store Routing Experiments")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--generate", action="store_true", help="Generate dataset")
    parser.add_argument("--train", action="store_true", help="Train and evaluate policies")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    parser.add_argument("--visualize", action="store_true", help="Generate figures")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--fig-dir", default="figures", help="Figures directory")
    args = parser.parse_args()
    
    if args.all or args.generate:
        generate_dataset(num_users=50, sessions_per_user=4, queries_per_user=30, output_dir=args.data_dir)
    
    if args.all or args.train:
        run_main_experiment(args.data_dir)
    
    if args.all or args.ablation:
        run_ablation_lambda(args.data_dir)
        run_ablation_ucb_c(args.data_dir)
    
    if args.all or args.visualize:
        generate_figures(args.data_dir, args.fig_dir)
        print("\nLaTeX Table:")
        print(generate_latex_table(args.data_dir))


if __name__ == "__main__":
    main()
