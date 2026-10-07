def detect_intent(question):
    """
    Detects the main intent/category of the user's question.
    This helps the backend decide which dataset should be searched.
    """

    question = question.lower()

    intent_keywords = {
        "professors": [
            "hoca",
            "hocası",
            "profesör",
            "profesor",
            "akademisyen",
            "öğretim üyesi",
            "ogretim uyesi",
            "araştırma görevlisi",
            "arastirma gorevlisi",
            "mail",
            "email",
            "profil",
            "profile",
            "bölüm başkanı",
            "bolum baskani",
            "bilgisayar mühendisliği hocası",
            "yazılım mühendisliği hocası",
            "rowanda",
            "burhan",
            "gamze",
            "ahmet",
            "murat",
            "fatih"
        ],

        "erasmus": [
            "erasmus",
            "hibe",
            "başvuru",
            "basvuru",
            "ortalama",
            "gano",
            "not ortalaması",
            "not ortalamasi",
            "yurt dışı",
            "yurt disi",
            "staj hareketliliği",
            "staj hareketliligi",
            "öğrenim hareketliliği",
            "ogrenim hareketliligi",
            "partner üniversite",
            "partner universite",
            "değişim programı",
            "degisim programi"
        ],

        "clubs": [
            "kulüp",
            "kulup",
            "topluluk",
            "student club",
            "sks",
            "developer",
            "dans",
            "animasyon",
            "bilimsel araştırma kulübü",
            "bilimsel arastirma kulubu",
            "biyoteknoloji",
            "beslenme",
            "diyetetik"
        ],

        "events": [
            "etkinlik",
            "event",
            "seminer",
            "seminar",
            "kongre",
            "congress",
            "sempozyum",
            "symposium",
            "atölye",
            "atolye",
            "workshop",
            "sergi",
            "exhibition",
            "mezuniyet",
            "tören",
            "toren",
            "mayıs",
            "mayis",
            "haziran",
            "temmuz",
            "ne zaman",
            "tarih"
        ],

        "research": [
            "araştırma",
            "arastirma",
            "research",
            "laboratuvar",
            "laboratory",
            "lab",
            "çalışma grubu",
            "calisma grubu",
            "bap",
            "bilimsel araştırma projesi",
            "bilimsel arastirma projesi",
            "tto",
            "brainpark",
            "arge",
            "r&d",
            "etik kurul",
            "ethics",
            "yayınlar",
            "publications",
            "kütüphane",
            "kutuphane"
        ],

        "smalltalk": [
            "merhaba",
            "selam",
            "hello",
            "hi",
            "nasılsın",
            "nasilsin",
            "naber",
            "teşekkür",
            "tesekkur",
            "sağ ol",
            "sag ol",
            "görüşürüz",
            "gorusuruz",
            "bye"
        ]
    }

    scores = {}

    for intent, keywords in intent_keywords.items():
        score = 0

        for keyword in keywords:
            if keyword in question:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "unknown"

    return best_intent


if __name__ == "__main__":
    test_questions = [
        "Rowanda hoca hangi bölümde?",
        "Erasmus hibesi ne kadar?",
        "Hangi kulüpler var?",
        "Mayıs ayında etkinlik var mı?",
        "Araştırma merkezleri neler?",
        "Merhaba nasılsın?",
        "Kafeterya nerede?"
    ]

    for question in test_questions:
        print(question, "->", detect_intent(question))