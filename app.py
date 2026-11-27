import os, re, random, threading
from flask import Flask, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator
from langdetect import detect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
app = Flask(__name__)

STOPWORDS = {
    "কি", "কে", "কী", "নাম", "হলে", "কোথায়", "কোথায়", "কেন", "কিভাবে", "কত", 
    "কোন", "কোনটি", "এর", "ও", "এবং", "হলো", "আছে", "একটি", "একটা", "আমি", 
    "আমার", "আমাকে", "তুমি", "তোমার", "সে", "তার", "এটা", "এটি", "কর", "করা",
    "the", "is", "of", "what", "a", "an", "in", "to", "and", "are", "how", "who", "where"
}

def stem_bangla(word):
    suffixes = ['ের', 'টি', 'কে', 'রা', 'দের', 'তে', 'রে', 'টির', 'টিতে', 'টার', 'টাকে', 'য়ের', 'য়']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word

class KeywordRetriever:
    def __init__(self):
        self.lock = threading.Lock()
        self.docs = []
        self.doc_tokens = []

    def tokenize(self, text):
        words = re.findall(r'[\u0980-\u09FF\w]+', text.lower())
        clean_words = []
        for w in words:
            if w not in STOPWORDS:
                clean_words.append(stem_bangla(w))
        return set(clean_words)

    def fit(self, lines):
        with self.lock:
            self.docs = list(dict.fromkeys([l for l in lines if l and l.strip()]))
            self.doc_tokens = [self.tokenize(doc) for doc in self.docs]
            print(f"✅ Retriever trained on {len(self.docs)} lines")

    def predict(self, text):
        with self.lock:
            q_tokens = self.tokenize(text)
            if not q_tokens: return None
            
            best_doc = None
            best_score = 0
            
            for i, d_tokens in enumerate(self.doc_tokens):
                intersection = len(q_tokens.intersection(d_tokens))
                union = len(q_tokens.union(d_tokens))
                if union > 0:
                    score = intersection / union
                    if score > best_score:
                        best_score = score
                        best_doc = self.docs[i]
            
            # ৪০% শব্দ মিললেই উত্তর দেবে
            if best_score >= 0.40:
                return best_doc
            return None

retriever = KeywordRetriever()
file_lock = threading.Lock()

def _load_knowledge():
    if not os.path.exists(KNOWLEDGE_FILE): return []
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    raw = re.split(r'[\n।?!\.]\s*', content)
    lines = [s.strip() for s in raw if len(s.strip()) > 2]
    return list(dict.fromkeys(lines))

def _train():
    try:
        lines = _load_knowledge()
        if lines: retriever.fit(lines)
    except Exception as e:
        print("Train error:", e)

def get_ai_response(user_input, history):
    norm = user_input.lower().strip()
    is_bengali = bool(re.search(r'[\u0980-\u09FF]', norm))

    # ১. হার্ডকোড করা মাল্টি-লিঙ্গুয়াল উত্তর (১০০% সঠিকতার জন্য)
    if ("national flower" in norm and "bangladesh" in norm) or "জাতীয় ফুল" in norm:
        return "বাংলাদেশের জাতীয় ফুল শাপলা। (The national flower of Bangladesh is Water Lily.)"
    
    if ("national bird" in norm and "bangladesh" in norm) or "জাতীয় পাখি" in norm:
        return "বাংলাদেশের জাতীয় পাখি দোয়েল। (The national bird of Bangladesh is Magpie Robin.)"
        
    if ("national fruit" in norm and "bangladesh" in norm) or "জাতীয় ফল" in norm:
        return "বাংলাদেশের জাতীয় ফল কাঁঠাল। (The national fruit of Bangladesh is Jackfruit.)"
        
    if ("national animal" in norm and "bangladesh" in norm) or "জাতীয় পশু" in norm:
        return "বাংলাদেশের জাতীয় পশু রয়েল বেঙ্গল টাইগার। (The national animal of Bangladesh is Royal Bengal Tiger.)"
        
    if ("capital" in norm and "bangladesh" in norm) or ("রাজধানী" in norm and "বাংলাদেশ" in norm):
        return "বাংলাদেশের রাজধানী ঢাকা। (The capital of Bangladesh is Dhaka.)"

    if "who created you" in norm or "who made you" in norm or "কে তৈরি" in norm or "তুমি কে" in norm:
        return "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক। (I was created by Mehedi Hasan Anik.)"

    # ২. ট্রান্সলেশন (যদি বাংলা না হয়)
    query_for_search = user_input
    if not is_bengali:
        try:
            query_for_search = GoogleTranslator(source='auto', target='bn').translate(user_input)
        except Exception:
            pass # ট্রান্সলেশন ফেইল করলে ইংরেজিতেই খুঁজবে

    norm_search = query_for_search.lower().strip()

    # ৩. ডাটাবেস থেকে খোঁজা
    ans = retriever.predict(norm_search)
    if ans:
        if not ans.endswith(("।", ".", "!", "?")):
            ans += "।"
        return ans

    # ৪. ফলব্যাক
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
