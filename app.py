import os, re, random, threading, requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

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
        return set(stem_bangla(w) for w in words if w not in STOPWORDS)

    def fit(self, lines):
        with self.lock:
            self.docs = list(dict.fromkeys([l for l in lines if l and l.strip()]))
            self.doc_tokens = [self.tokenize(doc) for doc in self.docs]
            print(f"✅ Retriever trained on {len(self.docs)} lines")

    def predict(self, text):
        with self.lock:
            q_tokens = self.tokenize(text)
            if not q_tokens: return None
            best_doc, best_score = None, 0
            for i, d_tokens in enumerate(self.doc_tokens):
                intersection = len(q_tokens.intersection(d_tokens))
                union = len(q_tokens.union(d_tokens))
                if union > 0:
                    score = intersection / union
                    if score > best_score:
                        best_score = score
                        best_doc = self.docs[i]
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

# --- নতুন হালকা ওয়েব সার্চ ইঞ্জিন ---
def get_web_answer(query):
    try:
        eng_query = query
        if not bool(re.search(r'[a-zA-Z]', query)):
            eng_query = GoogleTranslator(source='auto', target='en').translate(query)

        # ১. DuckDuckGo Instant Answer API (সবচেয়ে ফাস্ট)
        url = f"https://api.duckduckgo.com/?q={eng_query}&format=json&no_html=1"
        res = requests.get(url, timeout=5).json()
        if res.get('AbstractText'):
            return res['AbstractText']
        elif res.get('Answer'):
            return res['Answer']

        # ২. Wikipedia API
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{eng_query.replace(' ', '_')}"
        wiki_res = requests.get(wiki_url, timeout=5)
        if wiki_res.status_code == 200:
            return wiki_res.json().get('extract', None)
            
    except Exception as e:
        print("Web search error:", e)
    return None

def get_ai_response(user_input, history):
    norm_input = user_input.lower().strip()
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'bn'

    target_lang_code = detected_lang
    if target_lang_code.startswith('zh'): target_lang_code = 'zh-CN'

    ans_bn = None
    is_web_search = False

    # ১. লোকাল মাল্টি-লিঙ্গুয়াল ডিকশনারি
    if "首都" in norm_input or "首都是" in norm_input:
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "国花" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "national flower" in norm_input or "জাতীয় ফুল" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "national bird" in norm_input or "জাতীয় পাখি" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
    elif "capital" in norm_input or "রাজধানী" in norm_input:
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "who created" in norm_input or "who made" in norm_input or "কে তৈরি" in norm_input:
        ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"

    # ২. ডাটাবেসে খোঁজা
    if not ans_bn:
        query_in_bn = user_input
        if detected_lang != 'bn':
            try:
                query_in_bn = GoogleTranslator(source='auto', target='bn').translate(user_input)
            except Exception:
                pass
        ans_bn = retriever.predict(query_in_bn.lower().strip())

    # ৩. ইন্টারনেটে সার্চ করা
    if not ans_bn:
        web_result = get_web_answer(user_input)
        if web_result:
            ans_bn = web_result
            is_web_search = True
        else:
            ans_bn = "আমি এই বিষয়ে ইন্টারনেটে বা আমার ডাটাবেসে কোনো তথ্য খুঁজে পাইনি। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"

    # ৪. উত্তরটি ইউজারের ভাষায় অনুবাদ করা
    if detected_lang != 'bn' or is_web_search:
        try:
            src_lang = 'en' if is_web_search else 'bn'
            final_ans = GoogleTranslator(source=src_lang, target=target_lang_code).translate(ans_bn)
            return final_ans + (" 🌐" if is_web_search else "")
        except Exception:
            return ans_bn + (" 🌐" if is_web_search else "")

    return ans_bn

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
    except Exception:
        return jsonify({'reply': 'তথ্য সংরক্ষণে সমস্যা হয়েছে।'}), 500

if __name__ == '__main__':
    threading.Thread(target=_train, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
