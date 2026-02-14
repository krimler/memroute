"""
Comprehensive Benchmark Experiments with Improved Features
==========================================================

Fixes:
1. Enhanced features (quantity, temporal, comparison signals)
2. Embedding-based features (query-store similarity)
3. Hybrid approach (rules + learned)
4. More robust evaluation

Benchmarks: LoCoMo, LongMemEval (synthetic versions)
"""

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import re
import hashlib

random.seed(42)
np.random.seed(42)

#==============================================================================
# STORE AND ACTION DEFINITIONS
#==============================================================================

STORES = ["stm", "summary", "ltm", "episodic"]
STORE_COSTS = {"stm": 1, "summary": 1, "ltm": 3, "episodic": 5}

ACTIONS = {
    0: {"stores": set(), "k": 0, "name": "none"},
    1: {"stores": {"stm"}, "k": 2, "name": "stm"},
    2: {"stores": {"summary"}, "k": 3, "name": "summary"},
    3: {"stores": {"stm", "summary"}, "k": 5, "name": "stm+summary"},
    4: {"stores": {"ltm"}, "k": 5, "name": "ltm"},
    5: {"stores": {"summary", "ltm"}, "k": 8, "name": "summary+ltm"},
    6: {"stores": {"stm", "summary", "ltm", "episodic"}, "k": 10, "name": "all"},
}

NUM_ACTIONS = len(ACTIONS)

#==============================================================================
# ENHANCED FEATURE EXTRACTION
#==============================================================================

class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction with:
    1. Linguistic features (original)
    2. Semantic signals (quantity, temporal, comparison)
    3. Embedding-based features
    """
    
    # Pattern groups
    ANAPHORA = [r"\bthat\b", r"\bthis\b", r"\bit\b", r"\bthose\b", r"\bthese\b"]
    POSSESSIVE = [r"\bmy\b", r"\bmine\b", r"\bour\b", r"\bours\b"]
    PAST_REF = [r"\blast time\b", r"\bbefore\b", r"\bprevious\b", r"\bearlier\b", r"\blast\b", r"\bago\b"]
    GREETING = [r"^hello\b", r"^hi\b", r"^hey\b", r"^thanks\b", r"^ok\b", r"^okay\b"]
    
    # NEW: Quantity signals (memory_capacity queries)
    QUANTITY = [r"\ball\b", r"\blist\b", r"\bevery\b", r"\beach\b", r"\bsummarize\b", 
                r"\boverview\b", r"\bcomplete\b", r"\bentire\b", r"\bwhole\b"]
    
    # NEW: Temporal signals
    TEMPORAL = [r"\bbefore\b", r"\bafter\b", r"\bwhen\b", r"\blast week\b", r"\blast month\b",
                r"\byesterday\b", r"\brecently\b", r"\bhistory\b", r"\bover time\b", r"\bchanged\b"]
    
    # NEW: Multi-hop / comparison signals
    MULTI_HOP = [r"\bcompare\b", r"\bcombine\b", r"\brelate\b", r"\bconnect\b", r"\bboth\b",
                 r"\btogether\b", r"\bacross\b", r"\bbetween\b"]
    
    # NEW: Current/update signals
    CURRENT = [r"\bcurrent\b", r"\bnow\b", r"\btoday\b", r"\blatest\b", r"\bupdated\b",
               r"\bpresent\b", r"\bstill\b"]
    
    # NEW: Single fact signals
    SINGLE_FACT = [r"\bwhat is\b", r"\bwhat's\b", r"\bwho is\b", r"\bwhere is\b",
                   r"\bwhen is\b", r"\btell me\b"]
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        
        # Store centroids for embedding-based features
        self.store_centroids = self._init_store_centroids()
        
        # Query type centroids (learned from training)
        self.query_type_centroids = {}
    
    def _init_store_centroids(self) -> Dict[str, np.ndarray]:
        """Initialize store centroids with semantic priors."""
        centroids = {}
        
        # Create deterministic centroids based on store semantics
        np.random.seed(100)
        centroids["stm"] = np.random.randn(self.embedding_dim)
        np.random.seed(200)
        centroids["summary"] = np.random.randn(self.embedding_dim)
        np.random.seed(300)
        centroids["ltm"] = np.random.randn(self.embedding_dim)
        np.random.seed(400)
        centroids["episodic"] = np.random.randn(self.embedding_dim)
        
        # Normalize
        for store in centroids:
            centroids[store] /= np.linalg.norm(centroids[store])
        
        np.random.seed(42)  # Reset
        return centroids
    
    def _embed_query(self, text: str) -> np.ndarray:
        """Create query embedding from text."""
        embedding = np.zeros(self.embedding_dim)
        
        # Word-based features
        words = text.lower().split()
        for i, word in enumerate(words):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.embedding_dim
            embedding[idx] += 1.0
            
            # Bigrams
            if i > 0:
                bigram = words[i-1] + "_" + word
                idx2 = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % self.embedding_dim
                embedding[idx2] += 0.5
        
        # Pattern-based boosts
        text_lower = text.lower()
        
        if any(re.search(p, text_lower) for p in self.QUANTITY):
            embedding[0:4] += 2.0  # Quantity signal
        if any(re.search(p, text_lower) for p in self.TEMPORAL):
            embedding[4:8] += 2.0  # Temporal signal
        if any(re.search(p, text_lower) for p in self.MULTI_HOP):
            embedding[8:12] += 2.0  # Multi-hop signal
        if any(re.search(p, text_lower) for p in self.CURRENT):
            embedding[12:16] += 2.0  # Current/update signal
        if any(re.search(p, text_lower) for p in self.SINGLE_FACT):
            embedding[16:20] += 2.0  # Single fact signal
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding
    
    def fit(self, queries: List[Dict], labels: List[int]):
        """Learn query type centroids from training data."""
        type_embeddings = defaultdict(list)
        
        for query, label in zip(queries, labels):
            emb = self._embed_query(query.get("text", ""))
            query_type = query.get("query_type", "unknown")
            type_embeddings[query_type].append(emb)
        
        for qtype, embeddings in type_embeddings.items():
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                self.query_type_centroids[qtype] = centroid
    
    def extract(self, query: Dict) -> np.ndarray:
        """Extract enhanced feature vector."""
        text = query.get("text", "").lower()
        features = []
        
        # === Original linguistic features ===
        features.append(1 if any(re.search(p, text) for p in self.ANAPHORA) else 0)
        features.append(1 if any(re.search(p, text) for p in self.POSSESSIVE) else 0)
        features.append(1 if any(re.search(p, text) for p in self.PAST_REF) else 0)
        features.append(1 if any(re.search(p, text, re.I) for p in self.GREETING) else 0)
        features.append(1 if "?" in text else 0)
        features.append(len(text.split()) / 10.0)
        
        # === NEW: Semantic signal features ===
        features.append(1 if any(re.search(p, text) for p in self.QUANTITY) else 0)
        features.append(1 if any(re.search(p, text) for p in self.TEMPORAL) else 0)
        features.append(1 if any(re.search(p, text) for p in self.MULTI_HOP) else 0)
        features.append(1 if any(re.search(p, text) for p in self.CURRENT) else 0)
        features.append(1 if any(re.search(p, text) for p in self.SINGLE_FACT) else 0)
        
        # === NEW: Embedding-based features ===
        query_emb = self._embed_query(text)
        
        # Similarity to store centroids
        for store in ["stm", "summary", "ltm", "episodic"]:
            sim = np.dot(query_emb, self.store_centroids[store])
            features.append(sim)
        
        # Similarity to learned query type centroids (if available)
        if self.query_type_centroids:
            # Top-3 similarities
            sims = []
            for qtype, centroid in self.query_type_centroids.items():
                sims.append((qtype, np.dot(query_emb, centroid)))
            sims.sort(key=lambda x: -x[1])
            
            for i in range(min(3, len(sims))):
                features.append(sims[i][1])
            while len(features) < 20:  # Pad if needed
                features.append(0)
        else:
            features.extend([0, 0, 0])
        
        return np.array(features[:20], dtype=np.float32)


#==============================================================================
# IMPROVED POLICIES
#==============================================================================

class ImprovedUCBPolicy:
    """UCB policy with enhanced features."""
    
    name = "ucb_enhanced"
    
    def __init__(self, n_features: int = 20, c: float = 1.0, lr: float = 0.01):
        self.n_features = n_features
        self.c = c
        self.lr = lr
        
        self.weights = np.zeros((NUM_ACTIONS, n_features))
        self.bias = np.zeros(NUM_ACTIONS)
        self.counts = np.ones(NUM_ACTIONS)
        self.total = NUM_ACTIONS
        
        self.feature_extractor = EnhancedFeatureExtractor()
    
    def fit_features(self, queries: List[Dict], labels: List[int]):
        """Pre-train feature extractor."""
        self.feature_extractor.fit(queries, labels)
    
    def select_action(self, query: Dict, *args) -> int:
        features = self.feature_extractor.extract(query)
        q_values = np.dot(self.weights, features) + self.bias
        ucb_bonus = self.c * np.sqrt(np.log(self.total) / self.counts)
        return int(np.argmax(q_values + ucb_bonus))
    
    def update(self, query: Dict, action: int, reward: float, *args):
        features = self.feature_extractor.extract(query)
        pred = np.dot(self.weights[action], features) + self.bias[action]
        error = reward - pred
        self.weights[action] += self.lr * error * features
        self.bias[action] += self.lr * error
        self.counts[action] += 1
        self.total += 1


class HybridPolicy:
    """
    Hybrid policy: Rules for clear cases, learned for ambiguous.
    """
    
    name = "hybrid"
    
    def __init__(self, n_features: int = 20):
        self.learned_policy = ImprovedUCBPolicy(n_features=n_features, c=0.5)
        self.feature_extractor = EnhancedFeatureExtractor()
    
    def fit_features(self, queries: List[Dict], labels: List[int]):
        self.feature_extractor.fit(queries, labels)
        self.learned_policy.fit_features(queries, labels)
    
    def select_action(self, query: Dict, *args) -> int:
        text = query.get("text", "").lower()
        
        # Rule 1: Quantity/list queries → LTM + Episodic (all stores)
        if any(re.search(p, text) for p in self.feature_extractor.QUANTITY):
            return 6  # all stores
        
        # Rule 2: Temporal queries → LTM + Episodic
        if any(re.search(p, text) for p in self.feature_extractor.TEMPORAL):
            # Check if also needs summary
            if any(re.search(p, text) for p in self.feature_extractor.POSSESSIVE):
                return 6  # all stores
            return 6  # ltm + episodic (mapped to all for safety)
        
        # Rule 3: Multi-hop queries → Summary + LTM
        if any(re.search(p, text) for p in self.feature_extractor.MULTI_HOP):
            return 5  # summary + ltm
        
        # Rule 4: Current/update queries → Summary
        if any(re.search(p, text) for p in self.feature_extractor.CURRENT):
            return 2  # summary
        
        # Rule 5: Simple fact queries (no past, no quantity) → Summary
        if any(re.search(p, text) for p in self.feature_extractor.SINGLE_FACT):
            if not any(re.search(p, text) for p in self.feature_extractor.PAST_REF):
                return 2  # summary
        
        # Rule 6: Anaphora only → STM
        if any(re.search(p, text) for p in self.feature_extractor.ANAPHORA):
            if not any(re.search(p, text) for p in self.feature_extractor.POSSESSIVE):
                return 1  # stm
        
        # Rule 7: Past reference → LTM
        if any(re.search(p, text) for p in self.feature_extractor.PAST_REF):
            if any(re.search(p, text) for p in self.feature_extractor.POSSESSIVE):
                return 5  # summary + ltm
            return 4  # ltm
        
        # Fallback: Use learned policy
        return self.learned_policy.select_action(query)
    
    def update(self, query: Dict, action: int, reward: float, *args):
        self.learned_policy.update(query, action, reward)


class EmbeddingOnlyPolicy:
    """Pure embedding-based routing."""
    
    name = "embedding_only"
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.feature_extractor = EnhancedFeatureExtractor(embedding_dim)
        
        # Action centroids
        self.action_centroids = {a: np.random.randn(embedding_dim) for a in range(NUM_ACTIONS)}
        for a in self.action_centroids:
            self.action_centroids[a] /= np.linalg.norm(self.action_centroids[a])
    
    def fit_features(self, queries: List[Dict], labels: List[int]):
        """Learn action centroids from data."""
        action_embeddings = defaultdict(list)
        
        for query, label in zip(queries, labels):
            emb = self.feature_extractor._embed_query(query.get("text", ""))
            action_embeddings[label].append(emb)
        
        for action, embeddings in action_embeddings.items():
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                self.action_centroids[action] = centroid
    
    def select_action(self, query: Dict, *args) -> int:
        emb = self.feature_extractor._embed_query(query.get("text", ""))
        
        best_action = 0
        best_sim = -1
        
        for action, centroid in self.action_centroids.items():
            sim = np.dot(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_action = action
        
        return best_action
    
    def update(self, *args):
        pass  # No online update for this version


class UniformPolicy:
    name = "uniform"
    def select_action(self, *args): return 6
    def update(self, *args): pass
    def fit_features(self, *args): pass


class RuleBasedPolicy:
    """Original rule-based policy."""
    name = "rule_based"
    
    ANAPHORA = [r"\bthat\b", r"\bthis\b", r"\bit\b"]
    POSSESSIVE = [r"\bmy\b", r"\bmine\b", r"\bour\b"]
    PAST_REF = [r"\blast time\b", r"\bbefore\b", r"\bprevious\b", r"\blast\b"]
    GREETING = [r"^hello\b", r"^hi\b", r"^thanks\b"]
    
    def select_action(self, query: Dict, *args) -> int:
        text = query.get("text", "").lower()
        
        if any(re.search(p, text, re.I) for p in self.GREETING):
            return 0
        
        has_past = any(re.search(p, text) for p in self.PAST_REF)
        has_poss = any(re.search(p, text) for p in self.POSSESSIVE)
        has_ana = any(re.search(p, text) for p in self.ANAPHORA)
        
        if has_past and has_poss: return 5
        if has_past: return 4
        if has_ana and has_poss: return 3
        if has_ana: return 1
        if has_poss: return 2
        return 3
    
    def update(self, *args): pass
    def fit_features(self, *args): pass


#==============================================================================
# BENCHMARK ADAPTERS (same as before but with more data)
#==============================================================================

@dataclass
class LoCoMoSample:
    conversation_id: str
    session_id: int
    turn_id: int
    query: str
    answer: str
    evidence_sessions: List[int]
    evidence_turns: List[Tuple[int, int]]
    query_type: str


class LoCoMoAdapter:
    def __init__(self):
        self.samples: List[LoCoMoSample] = []
    
    def load_or_generate(self, n_per_type: int = 100):
        """Generate more diverse LoCoMo-style data."""
        
        query_types = {
            "single_session": {
                "queries": [
                    "What did you just mention about the project?",
                    "Can you clarify what you said earlier?",
                    "What was the main point of our discussion?",
                    "Repeat what you just told me.",
                    "What did you say about that?",
                    "Explain that last part again.",
                ],
                "target_stores": {"stm"},
                "action": 1,
            },
            "recent_session": {
                "queries": [
                    "What did we discuss in our last meeting?",
                    "What was decided yesterday?",
                    "Remind me of our recent conversation about the budget.",
                    "What happened in our previous call?",
                    "Tell me about last week's discussion.",
                ],
                "target_stores": {"ltm"},
                "action": 4,
            },
            "multi_session": {
                "queries": [
                    "Summarize all our discussions about the marketing strategy.",
                    "What have we talked about regarding customer feedback?",
                    "Give me an overview of our project conversations.",
                    "List all the topics we've covered.",
                    "Combine everything we discussed about sales.",
                ],
                "target_stores": {"ltm", "episodic"},
                "action": 6,
            },
            "temporal": {
                "queries": [
                    "What did I say about the deadline last week?",
                    "How has my opinion on this changed over time?",
                    "What was the situation before the update?",
                    "Compare my views from last month to now.",
                    "What was different before we made the change?",
                ],
                "target_stores": {"ltm", "episodic"},
                "action": 6,
            },
            "user_fact": {
                "queries": [
                    "What's my preferred meeting time?",
                    "What allergies do I have?",
                    "Where do I work?",
                    "What is my phone number?",
                    "Who is my manager?",
                    "What's my email address?",
                ],
                "target_stores": {"summary"},
                "action": 2,
            },
            "knowledge_update": {
                "queries": [
                    "What's my current address?",
                    "Who is my manager now?",
                    "What's the latest status of the project?",
                    "What's my current role?",
                    "What are my current priorities?",
                ],
                "target_stores": {"summary"},
                "action": 2,
            },
        }
        
        for qtype, config in query_types.items():
            for i in range(n_per_type):
                query = random.choice(config["queries"])
                
                # Add some variation
                if random.random() < 0.3:
                    query = query.replace("my", "our")
                if random.random() < 0.2:
                    query = "Please " + query.lower()
                
                sample = LoCoMoSample(
                    conversation_id=f"conv_{qtype}_{i}",
                    session_id=0,
                    turn_id=i,
                    query=query,
                    answer=f"Answer for {qtype}",
                    evidence_sessions=[],
                    evidence_turns=[],
                    query_type=qtype,
                )
                self.samples.append(sample)
        
        random.shuffle(self.samples)
        print(f"Generated {len(self.samples)} LoCoMo-style samples")
    
    def get_routing_labels(self) -> List[Tuple[Dict, Set[str], int]]:
        action_map = {
            "single_session": 1,
            "recent_session": 4,
            "multi_session": 6,
            "temporal": 6,
            "user_fact": 2,
            "knowledge_update": 2,
        }
        
        store_map = {
            "single_session": {"stm"},
            "recent_session": {"ltm"},
            "multi_session": {"ltm", "episodic"},
            "temporal": {"ltm", "episodic"},
            "user_fact": {"summary"},
            "knowledge_update": {"summary"},
        }
        
        labeled = []
        for sample in self.samples:
            labeled.append((
                {"text": sample.query, "query_type": sample.query_type,
                 "criticality": "HIGH" if "fact" in sample.query_type or "update" in sample.query_type else "MEDIUM"},
                store_map.get(sample.query_type, {"summary"}),
                action_map.get(sample.query_type, 2),
            ))
        return labeled


@dataclass
class LongMemEvalSample:
    query_id: str
    query: str
    answer: str
    category: str
    required_memories: List[str]
    temporal_constraint: Optional[str]


class LongMemEvalAdapter:
    def __init__(self):
        self.samples: List[LongMemEvalSample] = []
    
    def load_or_generate(self, n_per_type: int = 80):
        """Generate more diverse LongMemEval-style data."""
        
        categories = {
            "single_hop": {
                "queries": [
                    "What is my favorite restaurant?",
                    "When is my birthday?",
                    "What car do I drive?",
                    "What's my phone number?",
                    "Where do I live?",
                    "What is my job title?",
                    "Who is my best friend?",
                    "What's my favorite color?",
                ],
                "target_stores": {"summary"},
                "action": 2,
            },
            "multi_hop": {
                "queries": [
                    "Who introduced me to the restaurant I went to last week?",
                    "What did my friend recommend after I told them about my allergy?",
                    "Combine the feedback from my last three meetings.",
                    "Connect my travel plans with my work schedule.",
                    "How does my budget relate to my vacation plans?",
                ],
                "target_stores": {"summary", "ltm"},
                "action": 5,
            },
            "temporal": {
                "queries": [
                    "What was my weight before I started the diet?",
                    "What did I say about the project before the deadline changed?",
                    "Show me messages from before July.",
                    "What was my salary before the raise?",
                    "How did things change after the meeting?",
                ],
                "target_stores": {"ltm", "episodic"},
                "action": 6,
            },
            "knowledge_update": {
                "queries": [
                    "What's my current job title?",
                    "Who is my current landlord?",
                    "What's the latest project status?",
                    "What's my current address?",
                    "What are my current medications?",
                ],
                "target_stores": {"summary"},
                "action": 2,
            },
            "memory_capacity": {
                "queries": [
                    "List all the books I've mentioned reading.",
                    "Summarize all our discussions about travel.",
                    "What are all the restaurants I've mentioned?",
                    "Give me every project we discussed.",
                    "List all my past addresses.",
                    "Show me all the people I've mentioned.",
                ],
                "target_stores": {"ltm", "episodic"},
                "action": 6,
            },
        }
        
        for category, config in categories.items():
            for i in range(n_per_type):
                query = random.choice(config["queries"])
                
                sample = LongMemEvalSample(
                    query_id=f"{category}_{i}",
                    query=query,
                    answer=f"Answer for {category}",
                    category=category,
                    required_memories=[],
                    temporal_constraint="before" if category == "temporal" else None,
                )
                self.samples.append(sample)
        
        random.shuffle(self.samples)
        print(f"Generated {len(self.samples)} LongMemEval-style samples")
    
    def get_routing_labels(self) -> List[Tuple[Dict, Set[str], int]]:
        action_map = {
            "single_hop": 2,
            "multi_hop": 5,
            "temporal": 6,
            "knowledge_update": 2,
            "memory_capacity": 6,
        }
        
        store_map = {
            "single_hop": {"summary"},
            "multi_hop": {"summary", "ltm"},
            "temporal": {"ltm", "episodic"},
            "knowledge_update": {"summary"},
            "memory_capacity": {"ltm", "episodic"},
        }
        
        labeled = []
        for sample in self.samples:
            labeled.append((
                {"text": sample.query, "query_type": sample.category,
                 "criticality": "HIGH" if sample.category == "knowledge_update" else "MEDIUM"},
                store_map.get(sample.category, {"summary", "ltm"}),
                action_map.get(sample.category, 5),
            ))
        return labeled


#==============================================================================
# EVALUATION
#==============================================================================

def evaluate_policy(policy, test_data: List[Tuple], policy_name: str, verbose: bool = True) -> Dict:
    """Evaluate a policy on test data."""
    
    correct = 0
    total_k = 0
    per_type_correct = defaultdict(int)
    per_type_total = defaultdict(int)
    
    for query_dict, target_stores, oracle_action in test_data:
        pred = policy.select_action(query_dict)
        pred_stores = ACTIONS[pred]["stores"]
        
        # Correct if predicted stores cover required stores
        is_correct = target_stores.issubset(pred_stores)
        
        if is_correct:
            correct += 1
            per_type_correct[query_dict["query_type"]] += 1
        
        per_type_total[query_dict["query_type"]] += 1
        total_k += ACTIONS[pred]["k"]
    
    n = len(test_data)
    accuracy = correct / n if n > 0 else 0
    avg_k = total_k / n if n > 0 else 0
    
    results = {
        "accuracy": accuracy,
        "avg_k": avg_k,
        "efficiency": 10.0 / avg_k if avg_k > 0 else 0,
        "per_type": {qt: per_type_correct[qt] / per_type_total[qt] 
                     for qt in per_type_total},
    }
    
    if verbose:
        print(f"\n{policy_name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Avg k: {avg_k:.2f}")
        print(f"  Efficiency: {results['efficiency']:.2f}x")
    
    return results


def run_benchmark(benchmark_name: str, adapter, policies: Dict, 
                  train_epochs: int = 10, verbose: bool = True) -> Dict:
    """Run a complete benchmark evaluation."""
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*70}")
    
    # Get data
    labeled_data = adapter.get_routing_labels()
    random.shuffle(labeled_data)
    
    # Split: 70% train, 30% test
    n_train = int(len(labeled_data) * 0.7)
    train_data = labeled_data[:n_train]
    test_data = labeled_data[n_train:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Prepare training labels
    train_queries = [d[0] for d in train_data]
    train_labels = [d[2] for d in train_data]
    
    # Pre-train feature extractors
    for name, policy in policies.items():
        if hasattr(policy, 'fit_features'):
            policy.fit_features(train_queries, train_labels)
    
    # Train learned policies
    for epoch in range(train_epochs):
        random.shuffle(train_data)
        for query_dict, target_stores, oracle_action in train_data:
            for name, policy in policies.items():
                if hasattr(policy, 'update') and name not in ['uniform', 'rule_based']:
                    pred = policy.select_action(query_dict)
                    # Reward: +1 if correct, -1 if wrong
                    reward = 1.0 if target_stores.issubset(ACTIONS[pred]["stores"]) else -1.0
                    policy.update(query_dict, pred, reward)
    
    # Evaluate all policies
    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(policy, test_data, name, verbose)
    
    return results


#==============================================================================
# MAIN
#==============================================================================

def run_all_benchmarks():
    """Run comprehensive benchmark evaluation."""
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    all_results = {}
    
    # ============== LoCoMo ==============
    print("\n" + "="*70)
    print("LOCOMO BENCHMARK")
    print("="*70)
    
    locomo = LoCoMoAdapter()
    locomo.load_or_generate(n_per_type=100)
    
    policies_locomo = {
        "uniform": UniformPolicy(),
        "rule_based": RuleBasedPolicy(),
        "ucb_enhanced": ImprovedUCBPolicy(n_features=20, c=1.0),
        "embedding_only": EmbeddingOnlyPolicy(),
        "hybrid": HybridPolicy(),
    }
    
    locomo_results = run_benchmark("LoCoMo", locomo, policies_locomo, train_epochs=10)
    all_results["locomo"] = locomo_results
    
    # ============== LongMemEval ==============
    print("\n" + "="*70)
    print("LONGMEMEVAL BENCHMARK")
    print("="*70)
    
    longmemeval = LongMemEvalAdapter()
    longmemeval.load_or_generate(n_per_type=80)
    
    policies_longmem = {
        "uniform": UniformPolicy(),
        "rule_based": RuleBasedPolicy(),
        "ucb_enhanced": ImprovedUCBPolicy(n_features=20, c=1.0),
        "embedding_only": EmbeddingOnlyPolicy(),
        "hybrid": HybridPolicy(),
    }
    
    longmem_results = run_benchmark("LongMemEval", longmemeval, policies_longmem, train_epochs=10)
    all_results["longmemeval"] = longmem_results
    
    # ============== Summary ==============
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Benchmark':<15} {'Policy':<20} {'Accuracy':<12} {'Avg k':<10} {'Efficiency':<12}")
    print("-"*70)
    
    for benchmark in ["locomo", "longmemeval"]:
        for policy in ["uniform", "rule_based", "ucb_enhanced", "embedding_only", "hybrid"]:
            r = all_results[benchmark][policy]
            print(f"{benchmark:<15} {policy:<20} {r['accuracy']:<12.3f} {r['avg_k']:<10.2f} {r['efficiency']:<12.2f}x")
        print()
    
    # ============== Per-Type Breakdown ==============
    print("\n" + "-"*70)
    print("PER-TYPE ACCURACY COMPARISON")
    print("-"*70)
    
    for benchmark in ["locomo", "longmemeval"]:
        print(f"\n{benchmark.upper()}:")
        print(f"  {'Query Type':<25} {'Rule':<10} {'UCB+':<10} {'Embed':<10} {'Hybrid':<10}")
        print("  " + "-"*65)
        
        # Get all query types
        query_types = set()
        for policy in ["rule_based", "ucb_enhanced", "embedding_only", "hybrid"]:
            query_types.update(all_results[benchmark][policy]["per_type"].keys())
        
        for qt in sorted(query_types):
            row = f"  {qt:<25}"
            for policy in ["rule_based", "ucb_enhanced", "embedding_only", "hybrid"]:
                acc = all_results[benchmark][policy]["per_type"].get(qt, 0)
                row += f" {acc:<10.2f}"
            print(row)
    
    # ============== Key Findings ==============
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Best policy per benchmark
    for benchmark in ["locomo", "longmemeval"]:
        best_policy = max(
            ["ucb_enhanced", "embedding_only", "hybrid"],
            key=lambda p: all_results[benchmark][p]["accuracy"]
        )
        best_acc = all_results[benchmark][best_policy]["accuracy"]
        best_eff = all_results[benchmark][best_policy]["efficiency"]
        
        rule_acc = all_results[benchmark]["rule_based"]["accuracy"]
        improvement = (best_acc - rule_acc) / rule_acc * 100 if rule_acc > 0 else 0
        
        print(f"\n{benchmark.upper()}:")
        print(f"  Best policy: {best_policy}")
        print(f"  Accuracy: {best_acc:.1%} (vs rule-based {rule_acc:.1%}, +{improvement:.1f}%)")
        print(f"  Efficiency: {best_eff:.2f}x vs uniform")
    
    # ============== Plots ==============
    
    # Plot 1: Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    benchmarks = ["locomo", "longmemeval"]
    policy_names = ["uniform", "rule_based", "ucb_enhanced", "embedding_only", "hybrid"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        accs = [all_results[benchmark][p]["accuracy"] for p in policy_names]
        bars = ax.bar(range(len(policy_names)), accs, color=colors)
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha='right')
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{benchmark.upper()}")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accs):
            ax.annotate(f"{acc:.2f}", (bar.get_x() + bar.get_width()/2, acc + 0.02),
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("figures/benchmark_accuracy_all.png", dpi=150)
    plt.close()
    print("\nSaved: figures/benchmark_accuracy_all.png")
    
    # Plot 2: Accuracy vs Efficiency
    fig, ax = plt.subplots(figsize=(10, 7))
    
    markers = {"locomo": "o", "longmemeval": "s"}
    
    for benchmark in benchmarks:
        for j, policy in enumerate(policy_names):
            r = all_results[benchmark][policy]
            label = f"{policy}" if benchmark == "locomo" else None
            ax.scatter(r["efficiency"], r["accuracy"], s=150, c=colors[j], 
                      marker=markers[benchmark], label=label, alpha=0.8)
            
            # Annotate
            offset = (5, 5) if benchmark == "locomo" else (5, -10)
            ax.annotate(f"{benchmark[:4]}", (r["efficiency"], r["accuracy"]),
                       textcoords="offset points", xytext=offset, fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Efficiency (10 / avg_k)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Efficiency: All Policies", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 4)
    ax.set_ylim(0.2, 1.1)
    
    plt.tight_layout()
    plt.savefig("figures/benchmark_accuracy_efficiency_all.png", dpi=150)
    plt.close()
    print("Saved: figures/benchmark_accuracy_efficiency_all.png")
    
    # Save results
    with open("data/benchmark_results_comprehensive.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to data/benchmark_results_comprehensive.json")
    
    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
