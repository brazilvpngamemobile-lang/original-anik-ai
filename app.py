import os, re, random, json, threading, math
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
app = Flask(__name__)


# ------------------------------------------------------------------
# TF-IDF Retriever  (আগের বাগ-যুক্ত NeuralBrain-এর বদলে)
# ------------------------------------------------------------------
class TFIDFRetriever:
    def __init__(self):
        self.lock = threading.Lock()
        self.docs = []
        self.doc_vectors = []
        self.idf = {}

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

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

    def predict(self, text, threshold=0.20):
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


retriever = TFIDFRetriever()
file_lock = threading.Lock()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def _load_knowledge():
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    raw = re.split(r'[\n।?!\.]\s*', content)
    lines = []
    for s in raw:
        s = s.strip()
        if len(s) > 3:
            lines.append(s)
    return list(dict.fromkeys(lines))


def _train():
    try:
        lines = _load_knowledge()
        if lines:
            retriever.fit(lines)
            print(f"✅ Retriever trained on {len(lines)} lines")
        else:
            print("⚠️  knowledge.txt খালি বা পাওয়া যায়নি।")
    except Exception as e:
        print("Train error:", e)


# ------------------------------------------------------------------
# AI response logic
# ------------------------------------------------------------------
def get_ai_response(user_input, history):
    norm = user_input.lower().strip()
    is_bengali = bool(re.search(r'[\u0980-\u09FF]', norm))

    # অনিক ও নুডলস
    if any(k in norm for k in ["অনিক নুডলস", "অনিকের প্রিয় খাবার", "অনিক কি পছন্দ করে"]):
        return "অনিক নুডলস খুব বেশি পছন্দ করে!"

    # অনিক ও জনির ভালোবাসা
    if any(k in norm for k in ["ভালোবাসে", "ভালোবাসা", "love"]):
        if "অনিক" in norm or "জনি" in norm:
            return "অনিক জনিকে অনেক ভালোবাসে এবং জনিও অনিকে অনেক ভালোবাসে।"

    # শাশুড়ি মা ও জনির উপহার
    if any(k in norm for k in ["শাশুড়ি", "মাল্টি কুকার", "কাচের গ্লাস", "আম্মু", "উপহার"]):
        return ("জনি তার শাশুড়ি মা (অনিকের মা) কে একটি মাল্টি কুকার এবং অনেকগুলো চা "
                "খাওয়ার কাচের গ্লাস কিনে দিয়েছে। এতে আম্মু অনেক খুশি হয়েছেন।")

    # অলক
    if any(k in norm for k in ["অলক", "ছোট ভাই", "alok"]):
        if any(k in norm for k in ["চাকরি", "পড়াশোনা", "কোথায় থাকে", "গাজীপুর"]):
            return ("অলক অনিকের ছোট ভাই। সে গাজীপুরে চাকরি করে এবং বর্তমানে "
                    "পড়াশোনা ও চাকরি দুটোই করছে।")
        if any(k in norm for k in ["রেজাল্ট", "ফলাফল", "বুদ্ধি", "A+", "পরীক্ষা", "থ্রীফলি", "ক্লাস"]):
            return ("অলক অত্যন্ত বুদ্ধিমান এবং তার পড়াশোনা অতুলনীয়। সে ক্লাস প্লে থেকে ৫ম "
                    "শ্রেণি পর্যন্ত পুরস্কার পেয়েছে এবং PSC, JSC, SSC, HSC-তে A+ পেয়েছে। "
                    "বর্তমানে সে বিজ্ঞান বিভাগে থ্রীফলি (৩য় বর্ষ) নিয়ে পড়ছে। অলক একজন "
                    "Extra ordinary ছেলে।")
        return ("অলক হলো অনিকের ছোট ভাই। সে অত্যন্ত বুদ্ধিমান এবং পড়াশোনায় অতুলনীয়। "
                "সে গাজীপুরে থাকে এবং চাকরি ও পড়াশোনা একসাথে করছে।")

    # জনি
    if any(k in norm for k in ["জনি কে", "জনি কি", "যোনি কে"]):
        return "জনি হলো অনিকের স্ত্রী। তাদের বিয়ে হয়েছে ২৮ নভেম্বর ২০২৫ সালে। জনির বাড়ি পাইকশা।"

    # অনিক
    if any(k in norm for k in ["অনিক কে", "অনিক কি", "কে অনিক"]):
        return ("অনিকের সম্পূর্ণ নাম মেহেদী হাসান অনিক। তিনিই আমাকে তৈরি করেছেন। "
                "তিনি একজন অসাধারণ, প্রতিভাবান এবং উদ্ভাবনী ডেভেলপার।")

    # বিয়ে
    if any(k in norm for k in ["বিয়ে", "বিবাহ", "wedding", "marriage"]):
        if "অনিক" in norm or "জনি" in norm:
            return "অনিক এবং জনির বিয়ে হয়েছে ২৮ নভেম্বর ২০২৫ সালে।"
        return "বিয়ে একটি পবিত্র সামাজিক বন্ধন। উভয় পক্ষের পূর্ণ সম্মতি অত্যন্ত জরুরি।"

    if any(k in norm for k in ["অনিকের পরিবার", "family", "সদস্য"]):
        return "অনিকের পরিবারের সদস্য সংখ্যা পাঁচ জন: অনিক, জনি, অলক, মা এবং দাদু।"

    if any(k in norm for k in ["অনিকের বাড়ি", "anik home", "বায়োনচর", "চরজামাইল"]):
        return "অনিকের বাড়ি চরজামাইল শেখ চান্দের বাড়ি। অনেকে এই এলাকাকে বায়োনচর বলে।"

    if any(k in norm for k in ["জনির বাড়ি", "joni home"]):
        return "জনির বাড়ি পাইকশা।"

    if any(k in norm for k in ["অনিকের শখ", "ছোট বেলা", "আগ্রহ"]):
        return ("অনিক ছোট বেলা থেকেই ছবি আকা, নতুন কিছু তৈরি করা এবং আবিষ্কার করতে "
                "বেশি পছন্দ করত।")

    if any(k in norm for k in ["অনিকের পড়াশোনা", "marketing", "মার্কেটিং"]):
        return ("অনিক পড়াশোনা করেছে মার্কেটিং নিয়ে। তবে সে কোডিং, হ্যাকিং, অ্যাপ বা "
                "সফটওয়্যার তৈরি বিষয়ে বেশি আগ্রহী।")

    # সাধারণ কমান্ড
    if any(k in norm for k in ["কে তৈরি", "কি দিয়ে তৈরি", "who created", "who are you", "তুমি কে"]):
        return ("আমাকে তৈরি করেছেন মেহেদী হাসান অনিক। তিনি একজন অসাধারণ, প্রতিভাবান "
                "এবং উদ্ভাবনী ডেভেলপার।")

    if "বড় দ্বীপ" in norm:
        return "বিশ্বের সবচেয়ে বড় দ্বীপ হলো গ্রিনল্যান্ড (Greenland)।"
    if "ব দ্বীপ" in norm or "ব-দ্বীপ" in norm:
        return ("ব-দ্বীপ (Delta) হলো নদীর মোহনায় সৃষ্ট একটি ভূমিরূপ। বাংলাদেশ পৃথিবীর "
                "সবচেয়ে বড় ব-দ্বীপ।")

    # Context memory (নুডলস)
    if isinstance(history, list) and len(history) >= 2:
        prev_msg = str(history[-2]).lower()
        if "নুডলস" in prev_msg and any(k in norm for k in ["হ্যা", "হ্যাঁ", "দাম", "কত", "price"]):
            return ("নুডলসের দাম: কাংশিফু ৫০-৬০ টাকা, ইউনিপ্রেসিডেন্ট ৬০-৬৫ টাকা, "
                    "মিস্টার নুডলস ৫০ টাকা, ম্যাগি ২৮৫-৩৬৫ টাকা।")

    if "নুডলস" in norm or "নুডুলস" in norm:
        if "বাংলাদেশ" in norm:
            return "বাংলাদেশের জনপ্রিয় নুডলস: মিস্টার নুডলস, ম্যাগি, চপস্টিক।"
        return "চীনের জনপ্রিয় নুডলস: কাংশিফু, ইউনিপ্রেসিডেন্ট, বাইশিয়াং, তাংদারেন।"

    # ক্রিয়েটিভ কমান্ড
    if any(k in norm for k in ["কবি", "poet", "কবিতা"]):
        return ("আমি একজন কবি। শুনুন:\nআকাশ জুড়ে তারা ভরা রাত,\n"
                "তোমার হাসিতে আমার হৃদয় মাতোয়ার।\n"
                "বাতাসে ভেসে আসে ভালোবাসার গান,\nতুমি আছো বলে সুন্দর এই প্রাণ।")
    if any(k in norm for k in ["গান", "song", "গাইতে"]):
        return ("লালা লা লা... 🎶\nআমি একটি এআই, আমার গলা নেই, "
                "তবে আমি ছন্দে ছন্দে কথা বলতে পারি।")

    # Retriever fallback
    ans = retriever.predict(norm)
    if ans:
        if not ans.endswith(("।", ".", "!", "?")):
            ans += "।"
        return ans

    # Fallback
    if is_bengali:
        return random.choice([
            "আমি এই বিষয়ে নিশ্চিত নই। তবে আপনি আপনার দিনটি কেমন কাটছে তা বলতে পারেন!",
            "হুমম... আমার এই তথ্যটি নেই। আপনি কি 'Teach' বাটন (🎓) দিয়ে আমাকে শিখিয়ে দেবেন?"
        ])
    return "I'm not sure about that. But I am always learning."


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        reply = get_ai_response(data.get('message', ''), data.get('history', []))
        return jsonify({'reply': reply})
    except Exception as e:
        print("ask error:", e)
        return jsonify({'reply': 'দুঃখিত, একটি সমস্যা হয়েছে। আবার চেষ্টা করুন।'}), 500


@app.route('/teach', methods=['POST'])
def teach():
    try:
        data = request.get_json(silent=True) or {}
        fact = (data.get('fact') or '').strip()
        if not fact:
            return jsonify({'reply': 'তথ্য খালি!'})
        with file_lock:
            content = ""
            if os.path.exists(KNOWLEDGE_FILE):
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
            needs_nl = bool(content) and not content.endswith("\n")
            with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
                if needs_nl:
                    f.write("\n")
                f.write(fact + "\n")
        threading.Thread(target=_train, daemon=True).start()
        return jsonify({'reply': 'দারুণ! নতুন এই তথ্যটি শিখে নিলাম।'})
    except Exception as e:
        print("teach error:", e)
        return jsonify({'reply': 'তথ্য সংরক্ষণে সমস্যা হয়েছে।'}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    _train()
    return jsonify({'reply': 'Retrained.', **retriever.stats()})


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(retriever.stats())


if __name__ == '__main__':
    threading.Thread(target=_train, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
