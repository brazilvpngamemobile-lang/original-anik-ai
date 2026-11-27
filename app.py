import os, re, random, threading, requests
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

# লোকাল ট্রান্সলেশন ডিকশনারি (ইন্টারনেট ছাড়াই কিছু শব্দ অনুবাদ করতে)
LOCAL_TRANS = {
    "রাজধানী": "capital", "নদী": "river", "হ্রদ": "lake", "বিশ্বের": "world",
    "দীর্ঘতম": "longest", "বড়": "largest", "ছোট": "smallest", "উঁচু": "highest",
    "দেশ": "country", "পাখি": "bird", "ফুল": "flower", "ফল": "fruit", "পশু": "animal"
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
            
            # থ্রেশহোল্ড ০.২৫ করা হলো, যাতে সহজেই ম্যাচ করে
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
            mm_target = target_lang
            if target_lang.startswith('zh'): mm_target = 'zh-CN'
            elif target_lang == 'en': mm_target = 'en-GB'
            elif target_lang == 'bn': mm_target = 'bn-IN'
            return MyMemoryTranslator(source='auto', target=mm_target).translate(text)
        except Exception:
            # অনুবাদ সম্পূর্ণ ফেইল করলে লোকাল ডিকশনারি ব্যবহার
            if target_lang == 'en':
                eng_words = []
                for w in text.split():
                    eng_words.append(LOCAL_TRANS.get(w, w))
                return " ".join(eng_words)
            return text

def get_web_answer(query):
    try:
        # ইংরেজিতে অনুবাদ
        eng_query = safe_translate(query, 'en')
        headers = {'User-Agent': 'AnikAI/1.0'}

        # Wikipedia API সার্চ
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(eng_query)}&format=json"
        res = requests.get(search_url, headers=headers, timeout=8)
        
        if res.status_code == 200:
            search_data = res.json()
            if search_data['query']['search']:
                top_title = search_data['query']['search'][0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(top_title)}"
                summary_res = requests.get(summary_url, headers=headers, timeout=5)
                if summary_res.status_code == 200:
                    extract = summary_res.json().get('extract')
                    if extract:
                        return extract
    except Exception as e:
        print("Web search error:", e)
    return None

def get_ai_response(user_input, history):
    norm_input = user_input.lower().strip()
    if re.search(r'[\u0980-\u09FF]', norm_input):
        detected_lang = 'bn'
    else:
        try: detected_lang = detect(user_input)
        except: detected_lang = 'en'

    target_lang_code = 'zh-CN' if detected_lang.startswith('zh') else detected_lang
    ans_bn = None
    is_web_search = False

    # ১. লোকাল হার্ডকোড (শুধু নিশ্চিত প্রশ্নের জন্য)
    if ("capital" in norm_input or "রাজধানী" in norm_input) and ("bangladesh" in norm_input or "বাংলাদেশ" in norm_input):
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "national flower" in norm_input or "জাতীয় ফুল" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "national bird" in norm_input or "জাতীয় পাখি" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
    elif "who created" in norm_input or "who made" in norm_input or "কে তৈরি" in norm_input:
        ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"
    elif "অলক" in norm_input:
        ans_bn = "অলক হচ্ছে অনিকের ছোট ভাই। সে অত্যন্ত বুদ্ধিমান এবং পড়াশোনায় অতুলনীয়। সে গাজীপুরে থাকে এবং চাকরি ও পড়াশোনা একসাথে করছে।"

    # ২. ডাটাবেসে খোঁজা
    if not ans_bn:
        query_in_bn = user_input if detected_lang == 'bn' else safe_translate(user_input, 'bn')
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
