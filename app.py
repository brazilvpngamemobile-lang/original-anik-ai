import os, re, threading, requests, unicodedata
from flask import Flask, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

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
    # "বড়" এবং "বড়" এর মতো ইউনিকোড সমস্যা সমাধান করে
    return unicodedata.normalize('NFC', text.lower().strip())

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
            
            if best_score >= 0.25:
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

def safe_translate(text, target_lang):
    if not text.strip(): return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        try:
            mm_target = 'en-GB' if target_lang == 'en' else ('bn-IN' if target_lang == 'bn' else target_lang)
            return MyMemoryTranslator(source='auto', target=mm_target).translate(text)
        except Exception:
            return text

def get_web_answer(query, detected_lang):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (AnikAI/1.0)'}
        
        # ১. প্রথমে বাংলা উইকিপিডিয়ায় খোঁজা
        search_url_bn = f"https://bn.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(query)}&format=json"
        res_bn = requests.get(search_url_bn, headers=headers, timeout=8)
        if res_bn.status_code == 200:
            search_data_bn = res_bn.json()
            if search_data_bn['query']['search']:
                top_title = search_data_bn['query']['search'][0]['title']
                summary_url = f"https://bn.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(top_title)}"
                summary_res = requests.get(summary_url, headers=headers, timeout=5)
                if summary_res.status_code == 200:
                    extract = summary_res.json().get('extract')
                    if extract: return extract

        # ২. বাংলায় না পেলে ইংরেজিতে অনুবাদ করে খোঁজা
        eng_query = safe_translate(query, 'en')
        search_url_en = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(eng_query)}&format=json"
        res_en = requests.get(search_url_en, headers=headers, timeout=8)
        if res_en.status_code == 200:
            search_data_en = res_en.json()
            if search_data_en['query']['search']:
                top_title = search_data_en['query']['search'][0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(top_title)}"
                summary_res = requests.get(summary_url, headers=headers, timeout=5)
                if summary_res.status_code == 200:
                    return summary_res.json().get('extract')

    except Exception as e:
        print("Web search error:", e)
    return None

def get_ai_response(user_input, history):
    norm_input = normalize_text(user_input)
    
    if re.search(r'[\u0980-\u09FF]', norm_input):
        detected_lang = 'bn'
    else:
        try: detected_lang = detect(norm_input)
        except: detected_lang = 'en'

    target_lang_code = 'zh-CN' if detected_lang.startswith('zh') else detected_lang
    ans_bn = None
    is_web_search = False

    # ১. লোকাল হার্ডকোড (আগের বাদ পড়া প্রশ্নগুলো ফিরিয়ে আনা হলো)
    if "জনি কে" in norm_input or "জনি কি" in norm_input:
        ans_bn = "জনি হলো অনিকের স্ত্রী। তাদের বিয়ে হয়েছে ২৮ নভেম্বর ২০২৫ সালে। জনির বাড়ি পাইকশা।"
    elif "অনিক কে" in norm_input or "অনিক কি" in norm_input:
        ans_bn = "অনিকের সম্পূর্ণ নাম মেহেদী হাসান অনিক। তিনিই আমাকে তৈরি করেছেন। তিনি একজন অসাধারণ, প্রতিভাবান এবং উদ্ভাবনী ডেভেলপার।"
    elif "কে তৈরি" in norm_input or "who created" in norm_input:
        ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"
    elif "বিয়ে" in norm_input:
        ans_bn = "অনিক এবং জনির বিয়ে হয়েছে ২৮ নভেম্বর ২০২৫ সালে।"
    elif "অলক" in norm_input:
        ans_bn = "অলক হচ্ছে অনিকের ছোট ভাই। সে অত্যন্ত বুদ্ধিমান এবং পড়াশোনায় অতুলনীয়। সে গাজীপুরে থাকে এবং চাকরি ও পড়াশোনা একসাথে করছে।"
    
    # রাজধানী, বড়-ছোট দেশ (ইউনিকোড ঠিক করা হয়েছে)
    elif "রাজধানী" in norm_input or "capital" in norm_input:
        if "বাংলাদেশ" in norm_input or "bangladesh" in norm_input: ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
        elif "ইরান" in norm_input or "iran" in norm_input: ans_bn = "ইরানের রাজধানী তেহরান।"
        elif "ভারত" in norm_input or "india" in norm_input: ans_bn = "ভারতের রাজধানী নতুন দিল্লি।"
    elif "বড় দেশ" in norm_input or "largest country" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে বড় দেশ রাশিয়া।"
    elif "ছোট দেশ" in norm_input or "smallest country" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে ছোট দেশ ভ্যাটিকান সিটি।"
    elif "ছোট পাখি" in norm_input or "smallest bird" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে ছোট পাখি হামিংবার্ড।"
    elif "দীর্ঘতম নদী" in norm_input or "longest river" in norm_input:
        ans_bn = "বিশ্বের সবচেয়ে দীর্ঘতম নদী নীলনদ।"

    # ২. ডাটাবেসে খোঁজা
    if not ans_bn:
        query_in_bn = user_input if detected_lang == 'bn' else safe_translate(user_input, 'bn')
        ans_bn = retriever.predict(query_in_bn)

    # ৩. ইন্টারনেটে সার্চ করা
    if not ans_bn:
        web_result = get_web_answer(user_input, detected_lang)
        if web_result:
            ans_bn = web_result
            is_web_search = True
        else:
            ans_bn = "আমি এই বিষয়ে ইন্টারনেটে বা আমার ডাটাবেসে কোনো তথ্য খুঁজে পাইনি। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"

    # ৪. উত্তরটি ইউজারের ভাষায় অনুবাদ করা
    if detected_lang != 'bn' or is_web_search:
        try:
            # ওয়েব সার্চ থেকে আসা বাংলা বা ইংরেজি উত্তরকে ইউজারের ভাষায় অনুবাদ করা
            src_lang = 'auto' 
            final_ans = safe_translate(ans_bn, target_lang_code)
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
