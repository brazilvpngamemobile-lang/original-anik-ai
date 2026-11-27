import os, re, random, threading
from flask import Flask, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# ভাষা শনাক্তকরণে ধারাবাহিকতার জন্য সিড সেট করা
DetectorFactory.seed = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
app = Flask(__name__)

STOPWORDS = {
    "কি", "কে", "কী", "নাম", "হলে", "কোথায়", "কোথায়", "কেন", "কিভাবে", "কত", 
    "কোন", "কোনটি", "এর", "ও", "এবং", "হলো", "আছে", "একটি", "একটা", "আমি", 
    "আমার", "আমাকে", "তুমি", "তোমার", "সে", "তার", "এটা", "এটি", "কর", "করা",
    "the", "is", "of", "what", "a", "an", "in", "to", "and", "are", "how", "who", "where",
    "国家", "的", "是", "什么", "名字", "の", "は", "何", "이", "는", "무엇"
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
    # ১. ভাষা ডিটেক্ট করা
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'bn'

    # ২. যদি বাংলা না হয়, তবে বাংলায় অনুবাদ করা
    query_in_bn = user_input
    if detected_lang != 'bn':
        try:
            query_in_bn = GoogleTranslator(source='auto', target='bn').translate(user_input)
        except Exception as e:
            print("Translation to BN failed:", e)
            query_in_bn = user_input # ফেইল করলে অরিজিনাল টেক্সট দিয়েই খুঁজবে

    norm = query_in_bn.lower().strip()

    # ৩. উত্তর তৈরি করা (বাংলায়)
    ans_bn = None

    # হার্ডকোড করা বেসিক প্রশ্ন
    if "জাতীয় ফুল" in norm or "national flower" in norm:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "জাতীয় পাখি" in norm or "national bird" in norm:
        ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
    elif "জাতীয় ফল" in norm or "national fruit" in norm:
        ans_bn = "বাংলাদেশের জাতীয় ফল কাঁঠাল।"
    elif "জাতীয় পশু" in norm or "national animal" in norm:
        ans_bn = "বাংলাদেশের জাতীয় পশু রয়েল বেঙ্গল টাইগার।"
    elif "রাজধানী" in norm or "capital" in norm:
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "কে তৈরি" in norm or "who created" in norm or "who made" in norm:
        ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"

    # যদি হার্ডকোডে না পাওয়া যায়, তবে ডাটাবেস থেকে খোঁজা
    if not ans_bn:
        ans_bn = retriever.predict(norm)

    # যদি কোনো উত্তর না পাওয়া যায়
    if not ans_bn:
        ans_bn = "আমি এই বিষয়ে নিশ্চিত নই। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"

    # ৪. যদি ইউজারের ভাষা বাংলা না হয়, তবে উত্তরটি তার ভাষায় অনুবাদ করা
    if detected_lang != 'bn':
        try:
            # ভাষার কোড ম্যাপিং (কিছু ক্ষেত্রে langdetect 'zh-cn' দেয়, গুগল 'zh-CN' চায়)
            target_lang = detected_lang.split('-')[0] 
            final_ans = GoogleTranslator(source='bn', target=target_lang).translate(ans_bn)
            return final_ans
        except Exception as e:
            print("Translation reply failed:", e)
            # অনুবাদ ফেইল করলে বাংলা + ইংরেজি মিক্স করে দেবে
            return f"{ans_bn}"

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
    except Exception as e:
        return jsonify({'reply': 'তথ্য সংরক্ষণে সমস্যা হয়েছে।'}), 500

if __name__ == '__main__':
    threading.Thread(target=_train, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
