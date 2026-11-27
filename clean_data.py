import os

if not os.path.exists('knowledge.txt'):
    print("❌ knowledge.txt ফাইলটি পাওয়া যায়নি!")
    raise SystemExit(1)

with open('knowledge.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

seen = set()
unique_lines = []
for line in lines:
    stripped = line.strip()
    if stripped and stripped not in seen:
        seen.add(stripped)
        unique_lines.append(stripped + "\n")

with open('knowledge.txt', 'w', encoding='utf-8') as f:
    f.writelines(unique_lines)

print(f"✅ ডুপ্লিকেট মুছে ফেলা হয়েছে! আগে ছিল {len(lines)} লাইন, এখন আছে {len(unique_lines)} লাইন।")
