import os, re, random, json, threading, math
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
app = Flask(__name__)

# বাংলা ও ইংরেজি অপ্রয়োজনীয় শব্দ (Stopwords)
STOPWORDS = {
    "কি", "কে", "কী", "নাম", "হলে", "কোথায়", "কোথায়", "কেন", "কিভাবে", "কত", 
    "কোন", "কোনটি", "এর", "ও", "এবং", "হলো", "আছে", "একটি", "একটা", "আমি", 
    "আমার", "আমাকে", "তুমি", "তোমার", "সে", "তার", "এটা", "এটি", "কর", "করা",
    "the", "is", "of", "what", "a", "an", "in", "to", "and", "are", "how", "who", "where"
}

# ইংরেজি প্রশ্নকে বাংলায় ম্যাপ করার ডিকশনারি (Cross-Lingual Match)
TRANSLATION_MAP = {
    "flower": "ফুল", "bird": "পাখি", "capital": "রাজধানী", "fruit": "ফল", 
    "animal": "পশু", "river": "নদী", "country": "দেশ", "national": "জাতীয়",
    "largest": "বড়", "smallest": "ছোট", "longest": "দীর্ঘতম", "highest": "উঁচু",
    "ocean": "মহাসাগর", "continent": "মহাদেশ", "creator": "নির্মাতা", "founder": "প্রতিষ্ঠাতা",
    "birth": "জন্ম", "died": "মৃত্যু", "president": "রাষ্ট্রপতি", "minister": "মন্ত্রী"
}

# বাংলা প্রত্যয় ছাঁটাই (Stemming)
def stem_bangla(word):
    suffixes = ['ের', 'টি', 'কে', 'রা', 'দের', 'তে', 'রে', 'টির', 'টিতে', 'টার', 'টাকে', 'য়ের', 'য়']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word

class SmartRetriever:
    def __init__(self):
        self.lock = threading.Lock()
        self.docs = []
        self.doc_vectors = []
        self.idf = {}

    def tokenize(self, text):
        words = re.findall(r'[\u0980-\u09FF\w]+', text.lower())
        clean_words = []
        for w in words:
            if w in TRANSLATION_MAP:
                w = TRANSLATION_MAP[w] # ইংরেজি থেকে বাংলায় রূপান্তর
            if w not in STOPWORDS:
                clean_words.append(stem_bangla(w))
        return clean_words

    def fit(self, lines):
        with self.lock:
            self.docs = list(dict.fromkeys([l for l in lines if l and l.strip()]))
            n = len(self.docs)
            if n == 0:
                self.doc_vectors = []
                self.idf = {}
                return
            
            tokenized = [self.tokenize(d) for d in self.docs]
            df = Counter()
            for toks in tokenized:
                for w in set(toks):
                    df[w] += 1
            
            self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
            self.doc_vectors = [self._vectorize(toks) for toks in tokenized]

    def _vectorize(self, tokens):
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec = {}
        for w, c in counts.items():
            if w in self.idf:
                vec[w] = c * self.idf[w]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {w: v / norm for w, v in vec.items()}
        return vec

    def predict(self, text, threshold=0.30):
        with self.lock:
            if not self.doc_vectors:
                return None
            toks = self.tokenize(text)
            if not toks:
                return None
            qvec = self._vectorize(toks)
            if not qvec:
                return None
            
            best_score = 0.0
            best_idx = -1
            for i, dvec in enumerate(self.doc_vectors):
                score = sum(v * dvec.get(w, 0.0) for w, v in qvec.items())
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score >= threshold and best_idx >= 0:
                return self.docs[best_idx]
            return None

    def stats(self):
        with self.lock:
            return {"docs": len(self.docs), "idf_terms": len(self.idf)}

retriever = SmartRetriever()
file_lock = threading.Lock()

def _load_knowledge():
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    raw = re.split(r'[\n।?!\.]\s*', content)
    lines = [s.strip() for s in raw if len(s.strip()) > 2]
    return list(dict.fromkeys(lines))

def _train():
    try:
        lines = _load_knowledge()
        if lines:
            retriever.fit(lines)
            print(f"✅ Smart Retriever trained on {len(lines)} lines")
    except Exception as e:
        print("Train error:", e)

def get_ai_response(user_input, history):
    norm = user_input.lower().strip()
    is_bengali = bool(re.search(r'[\u0980-\u09FF]', norm))

    # Hardcoded Intents
    if any(k in norm for k in ["অনিক নুডলস", "অনিকের প্রিয় খাবার", "অনিক কি পছন্দ করে"]):
        return "অনিক নুডলস খুব বেশি পছন্দ করে!"
    if any(k in norm for k in ["ভালোবাসে", "ভালোবাসা", "love"]):
        if "অনিক" in norm or "জনি" in norm:
            return "অনিক জনিকে অনেক ভালোবাসে এবং জনিও অনিকে অনেক ভালোবাসে।"
    if any(k in norm for k in ["অলক", "ছোট ভাই", "alok"]):
        return "অলক হলো অনিকের ছোট ভাই। সে অত্যন্ত বুদ্ধিমান এবং পড়াশোনায় অতুলনীয়। সে গাজীপুরে থাকে এবং চাকরি ও পড়াশোনা একসাথে করছে।"
    if any(k in norm for k in ["জনি কে", "জনি কি", "যোনি কে"]):
        return "জনি হলো অনিকের স্ত্রী। তাদের বিয়ে হয়েছে ২৮ নভেম্বর ২০২৫ সালে। জনির বাড়ি পাইকশা।"
    if any(k in norm for k in ["অনিক কে", "অনিক কি", "কে অনিক"]):
        return "অনিকের সম্পূর্ণ নাম মেহেদী হাসান অনিক। তিনিই আমাকে তৈরি করেছেন। তিনি একজন অসাধারণ, প্রতিভাবান এবং উদ্ভাবনী ডেভেলপার।"
    if any(k in norm for k in ["কে তৈরি", "কি দিয়ে তৈরি", "who created", "who are you", "তুমি কে"]):
        return "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক। তিনি একজন অসাধারণ, প্রতিভাবান এবং উদ্ভাবনী ডেভেলপার।"

    # Retriever
    ans = retriever.predict(norm)
    if ans:
        if not ans.endswith(("।", ".", "!", "?")):
            ans += "।"
        return ans

    if is_bengali:
        return "আমি এই বিষয়ে নিশ্চিত নই। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"
    return "I'm not sure about that. Please teach me using the 'Teach' button."

@app.route('/')
def index(): return send_from_directory(BASE_DIR, 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        reply = get_ai_response(data.get('message', ''), data.get('history', []))
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': 'দুঃখিত, একটি সমস্যা হয়েছে।'}), 500

@app.route('/teach', methods=['POST'])
def teach():
    try:
        data = request.get_json(silent=True) or {}
        fact = (data.get('fact') or '').strip()
        if not fact: return jsonify({'reply': 'তথ্য খালি!'})
        with file_lock:
            content = ""
            if os.path.exists(KNOWLEDGE_FILE):
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
            needs_nl = bool(content) and not content.endswith("\n")
            with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
                if needs_nl: f.write("\n")
                f.write(fact + "\n")
        threading.Thread(target=_train, daemon=True).start()
        return jsonify({'reply': 'দারুণ! নতুন এই তথ্যটি শিখে নিলাম।'})
    except Exception as e:
        return jsonify({'reply': 'তথ্য সংরক্ষণে সমস্যা হয়েছে।'}), 500

if __name__ == '__main__':
    threading.Thread(target=_train, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
