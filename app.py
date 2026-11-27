import os, re, threading, requests, unicodedata, wikipedia
from flask import Flask, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
wikipedia.set_user_agent("AnikAI/1.0 (https://github.com/brazilvpngamemobile-lang/original-anik-ai)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
app = Flask(__name__)

STOPWORDS = {
    "কি", "কে", "কী", "নাম", "হলে", "কোথায়", "কোথায়", "কেন", "কিভাবে", "কত", 
    "কোন", "কোনটি", "এর", "ও", "এবং", "হলো", "আছে", "একটি", "একটা", "আমি", 
    "আমার", "আমাকে", "তুমি", "তোমার", "সে", "তার", "এটা", "এটি", "কর", "করা",
    "সব", "থেকে", "the", "is", "of", "what", "a", "an", "in", "to", "and", "are", "how", "who", "where"
}

def normalize_text(text):
    text = unicodedata.normalize('NFC', text.lower().strip())
    # ড়, ঢ়, য় এর ইউনিকোড সমস্যা সমাধান
    text = text.replace('\u09DC', '\u09A1\u09BC').replace('\u09DD', '\u09A2\u09BC').replace('\u09DF', '\u09AF\u09BC')
    return text

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
        words = re.findall(r'[\u0980-\u09FF\w]+', normalize_text(text))
        return set(stem_bangla(w) for w in words if w not in STOPWORDS)

    def fit(self, lines):
        with self.lock:
            self.docs = list(dict.fromkeys([l for l in lines if l and l.strip()]))
            self.doc_tokens = [self.tokenize(doc) for doc in self.docs]

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
            if best_score >= 0.25: return best_doc
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
    except: pass

def safe_translate(text, target_lang):
    if not text.strip(): return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        try:
            mm_target = 'en-GB' if target_lang == 'en' else ('bn-IN' if target_lang == 'bn' else target_lang)
            return MyMemoryTranslator(source='auto', target=mm_target).translate(text)
        except: return text

def get_web_answer(query, detected_lang):
    try:
        # ১. বাংলা প্রশ্ন হলে বাংলা উইকিপিডিয়ায় সার্চ
        if detected_lang == 'bn':
            wikipedia.set_lang("bn")
            results = wikipedia.search(query, results=1)
            if results:
                return wikipedia.summary(results[0], sentences=2)
        
        # ২. না পেলে বা অন্য ভাষা হলে ইংরেজিতে অনুবাদ করে ইংরেজি উইকিপিডিয়ায় সার্চ
        eng_query = safe_translate(query, 'en')
        wikipedia.set_lang("en")
        results = wikipedia.search(eng_query, results=1)
        if results:
            ans = wikipedia.summary(results[0], sentences=2)
            return safe_translate(ans, 'bn') if detected_lang == 'bn' else ans
    except Exception as e:
        print("Wikipedia Error:", e)
    return None

def get_ai_response(user_input, history):
    norm_input = normalize_text(user_input)
    detected_lang = 'bn' if re.search(r'[\u0980-\u09FF]', norm_input) else 'en'
    target_lang_code = 'zh-CN' if detected_lang.startswith('zh') else detected_lang
    ans_bn = None
    is_web_search = False

    # ==========================================
    # ১. লোকাল হার্ডকোড (সাধারণ প্রশ্নের নিশ্চিত উত্তর)
    # ==========================================
    if "বিল গেটস" in norm_input or "bill gates" in norm_input:
        ans_bn = "বিল গেটস মাইক্রোসফটের সহ-প্রতিষ্ঠাতা এবং বিশ্বের অন্যতম ধনী ব্যক্তি।"
    elif "eva elfie" in norm_input:
        ans_bn = "ইভা এলফি (Eva Elfie) একজন রাশিয়ান মডেল এবং প্রাপ্তবয়স্ক চলচ্চিত্র অভিনেত্রী।"
    elif "বড় দেশ" in norm_input or "বৃহত্তম দেশ" in norm_input or "largest country" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে বড় দেশ রাশিয়া।"
    elif "ছোট দেশ" in norm_input or "smallest country" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে ছোট দেশ ভ্যাটিকান সিটি।"
    elif "চায়না" in norm_input or "চীন" in norm_input or "china" in norm_input:
        ans_bn = "চীন (China) পূর্ব এশিয়ার একটি দেশ। এটি জনসংখ্যার দিক থেকে বিশ্বের সবচেয়ে বড় দেশ এবং আয়তনে তৃতীয়।"
    elif "পিরামিডের দেশ" in norm_input:
        ans_bn = "মিশরকে পিরামিডের দেশ বলা হয়।"
    elif "দীর্ঘতম নদী" in norm_input or "longest river" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে দীর্ঘতম নদী নীলনদ।"
    elif "রাজধানী" in norm_input or "capital" in norm_input:
        if "বাংলাদেশ" in norm_input or "bangladesh" in norm_input: ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
        elif "ভারত" in norm_input or "india" in norm_input: ans_bn = "ভারতের রাজধানী নতুন দিল্লি।"

    # ==========================================
    # ২. ডাটাবেসে খোঁজা
    # ==========================================
    if not ans_bn:
        query_in_bn = user_input if detected_lang == 'bn' else safe_translate(user_input, 'bn')
        ans_bn = retriever.predict(query_in_bn)

    # ==========================================
    # ৩. ইন্টারনেটে সার্চ করা (উইকিপিডিয়া এপিআই)
    # ==========================================
    if not ans_bn:
        web_result = get_web_answer(user_input, detected_lang)
        if web_result:
            ans_bn = web_result
            is_web_search = True
        else:
            ans_bn = "আমি এই বিষয়ে ইন্টারনেটে বা আমার ডাটাবেসে কোনো তথ্য খুঁজে পাইনি। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"

    # ==========================================
    # ৪. উত্তরটি ইউজারের ভাষায় অনুবাদ করা
    # ==========================================
    if detected_lang != 'bn' and not is_web_search:
        ans_bn = safe_translate(ans_bn, target_lang_code)

    return ans_bn + (" 🌐" if is_web_search else "")

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
    except:
        return jsonify({'reply': 'তথ্য সংরক্ষণে সমস্যা হয়েছে।'}), 500

if __name__ == '__main__':
    threading.Thread(target=_train, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
