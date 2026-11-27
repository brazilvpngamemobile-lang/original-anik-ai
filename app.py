import os, re, random, threading
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

# নতুন ফাংশন: ট্রান্সলেশন ফেইল হলে বিকল্প উপায়
def safe_translate(text, target_lang):
    if not text.strip(): return text
    
    # ১. প্রথমে গুগল ট্রান্সলেটর চেষ্টা
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print("Google Translator failed, trying MyMemory:", e)
        
    # ২. গুগল ফেইল করলে MyMemory ব্যবহার
    try:
        mm_target = target_lang
        if target_lang.startswith('zh'): mm_target = 'zh-CN'
        elif target_lang == 'en': mm_target = 'en-GB'
        elif target_lang == 'bn': mm_target = 'bn-IN'
        
        return MyMemoryTranslator(source='auto', target=mm_target).translate(text)
    except Exception as e:
        print("MyMemory Translator also failed:", e)
        
    return text # সব ফেইল করলে অরিজিনাল টেক্সট ফেরত দেবে

def get_ai_response(user_input, history):
    norm_input = user_input.lower().strip()
    
    # ভাষা ডিটেক্ট
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'bn'

    target_lang_code = detected_lang
    if target_lang_code.startswith('zh'):
        target_lang_code = 'zh-CN'

    ans_bn = None

    # ১. সরাসরি মাল্টি-লিঙ্গুয়াল শব্দ চেক (ট্রান্সলেশন ফেইল করলেও কাজ করবে)
    if "首都" in norm_input and "孟加拉" in norm_input:
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "国花" in norm_input and "孟加拉" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "国鸟" in norm_input and "孟加拉" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
        
    elif "national flower" in norm_input or "জাতীয় ফুল" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
    elif "national bird" in norm_input or "জাতীয় পাখি" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
    elif "national fruit" in norm_input or "জাতীয় ফল" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় ফল কাঁঠাল।"
    elif "national animal" in norm_input or "জাতীয় পশু" in norm_input:
        ans_bn = "বাংলাদেশের জাতীয় পশু রয়েল বেঙ্গল টাইগার।"
    elif "capital" in norm_input or "রাজধানী" in norm_input:
        ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
    elif "who created" in norm_input or "who made" in norm_input or "কে তৈরি" in norm_input:
        ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"

    # ২. সরাসরি না পেলে, অনুবাদ করে খোঁজা
    if not ans_bn:
        query_in_bn = user_input
        if detected_lang != 'bn':
            query_in_bn = safe_translate(user_input, 'bn')
            
        norm = query_in_bn.lower().strip()
        
        # অনুবাদ করা বাংলা টেক্সট চেক
        if "জাতীয় ফুল" in norm: ans_bn = "বাংলাদেশের জাতীয় ফুল শাপলা।"
        elif "জাতীয় পাখি" in norm: ans_bn = "বাংলাদেশের জাতীয় পাখি দোয়েল।"
        elif "জাতীয় ফল" in norm: ans_bn = "বাংলাদেশের জাতীয় ফল কাঁঠাল।"
        elif "জাতীয় পশু" in norm: ans_bn = "বাংলাদেশের জাতীয় পশু রয়েল বেঙ্গল টাইগার।"
        elif "রাজধানী" in norm: ans_bn = "বাংলাদেশের রাজধানী ঢাকা।"
        elif "কে তৈরি" in norm: ans_bn = "আমাকে তৈরি করেছেন মেহেদী হাসান অনিক।"
        else: 
            ans_bn = retriever.predict(norm)

    # ৩. ডিফল্ট ফলব্যাক
    if not ans_bn:
        ans_bn = "আমি এই বিষয়ে নিশ্চিত নই। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে এটি শিখিয়ে দেবেন?"

    # ৪. উত্তরটি ইউজারের ভাষায় অনুবাদ করে দেওয়া
    if detected_lang != 'bn':
        final_ans = safe_translate(ans_bn, target_lang_code)
        return final_ans

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
