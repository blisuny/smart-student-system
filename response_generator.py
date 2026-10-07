import random

from data_loader import load_all_data
from intent_detector import detect_intent
from search_engine import search_dataset, calculate_score


def is_list_question(question):
    """
    Detects whether the user is asking for a list instead of one specific item.
    """

    question = question.lower()

    list_keywords = [
        "hangi",
        "neler",
        "nelerdir",
        "listele",
        "hepsi",
        "tüm",
        "tum",
        "var mı",
        "var mi",
        "kulüpler",
        "kulupler",
        "etkinlikler",
        "hocalar",
        "akademisyenler",
        "araştırmalar",
        "arastirmalar"
    ]

    return any(keyword in question for keyword in list_keywords)


def get_top_matches(question, intent, all_data, limit=5):
    """
    Returns the most relevant dataset records for list-type questions.
    """

    if intent not in all_data:
        return []

    scored_items = []

    for item in all_data[intent]:
        score = calculate_score(question, item)
        scored_items.append((score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)

    # If the question is broad, return first records even if score is low.
    if all(score == 0 for score, item in scored_items):
        return [item for score, item in scored_items[:limit]]

    return [item for score, item in scored_items[:limit] if score > 0]


def generate_smalltalk_response(question):
    """
    Generates friendly and varied responses for daily conversation.
    """

    question = question.lower()

    greetings = ["merhaba", "selam", "hello", "hi"]
    mood_questions = ["nasılsın", "nasilsin", "naber", "ne haber"]
    thanks = ["teşekkür", "tesekkur", "sağ ol", "sag ol"]
    goodbyes = ["görüşürüz", "gorusuruz", "bye"]

    has_greeting = any(word in question for word in greetings)
    asks_mood = any(word in question for word in mood_questions)
    says_thanks = any(word in question for word in thanks)
    says_goodbye = any(word in question for word in goodbyes)

    if has_greeting and asks_mood:
        return random.choice([
            "Merhaba! İyiyim, teşekkür ederim. Sana Üsküdar Üniversitesi hakkında yardımcı olabilirim. Akademik kadro, Erasmus, kulüpler, etkinlikler veya araştırma bilgileriyle ilgili bir şey sorabilirsin.",
            "Selam! Ben iyiyim, umarım sen de iyisindir. Üniversiteyle ilgili bilgi arıyorsan buradayım.",
            "Merhaba, iyiyim teşekkürler. Sana Erasmus, hocalar, kulüpler, etkinlikler ve araştırma sayfaları hakkında yardımcı olabilirim."
        ])

    if has_greeting:
        return random.choice([
            "Merhaba! Sana nasıl yardımcı olabilirim?",
            "Selam! Üsküdar Üniversitesi ile ilgili ne öğrenmek istersin?",
            "Merhaba, buradayım. Akademik kadro, Erasmus, etkinlikler, kulüpler veya araştırma bilgileri hakkında soru sorabilirsin."
        ])

    if asks_mood:
        return random.choice([
            "İyiyim, teşekkür ederim. Sana nasıl yardımcı olabilirim?",
            "Gayet iyiyim, teşekkürler. Üniversiteyle ilgili bir konuda yardımcı olayım mı?",
            "İyiyim. Bugün sana Erasmus, hocalar, kulüpler, etkinlikler veya araştırma bilgileri hakkında yardımcı olabilirim."
        ])

    if says_thanks:
        return random.choice([
            "Rica ederim!",
            "Ne demek, yardımcı olabildiysem sevindim.",
            "Rica ederim. Başka bir şey sormak istersen buradayım."
        ])

    if says_goodbye:
        return random.choice([
            "Görüşürüz!",
            "Kendine iyi bak, tekrar yardıma ihtiyacın olursa yazabilirsin.",
            "Görüşmek üzere!"
        ])

    return "Sana yardımcı olmaya hazırım. Üniversiteyle ilgili bir şey sormak ister misin?"


def generate_professor_response(item):
    """
    Generates a natural Turkish response for professor data.
    """

    name = item.get("name", "Bu akademisyen")
    title = item.get("title", "")
    department = item.get("department", "bölüm bilgisi bulunamadı")
    faculty = item.get("faculty", "fakülte bilgisi bulunamadı")
    profile_url = item.get("profile_url", "")

    templates = [
        f"{name}, {department} bölümünde görev yapan {title} akademik personeldir. Fakülte bilgisi olarak {faculty} görünüyor.",
        f"Aradığın akademisyen {name}. Kendisi {department} bölümünde {title} olarak yer alıyor.",
        f"{name} hakkında elimdeki bilgiye göre kendisi {faculty} bünyesinde, {department} alanında görev yapıyor.",
        f"{department} bölümünde {title} olarak görünen akademisyenlerden biri {name}."
    ]

    response = random.choice(templates)

    if profile_url and profile_url != "Not available":
        response += f"\n\nDaha detaylı bilgi için akademik profil sayfası: {profile_url}"

    return response


def generate_professors_list(items):
    """
    Generates a list response for professor questions.
    """

    if not items:
        return "Akademik kadro datasında uygun bir kayıt bulamadım."

    response = "Akademik kadro datasında öne çıkan bazı hocalar şunlar:\n"

    for item in items:
        name = item.get("name", "İsim bilgisi yok")
        title = item.get("title", "")
        department = item.get("department", "Bölüm bilgisi yok")
        response += f"\n- {title} {name} — {department}"

    return response


def generate_erasmus_response(item):
    """
    Generates a natural Turkish response for Erasmus data.
    """

    title = item.get("title", "Erasmus Bilgisi")
    description = item.get("description", "")

    openings = [
        f"Bu konuda Erasmus datasında şu bilgi yer alıyor: {description}",
        f"{title} başlığı altında verilen bilgiye göre: {description}",
        f"Erasmus ile ilgili bu soruya şöyle cevap verebilirim: {description}",
        f"Elimdeki Erasmus bilgilerine göre {description}"
    ]

    response = random.choice(openings)

    if "requirements" in item:
        response += "\n\nBaşvuru kriterleri şöyle:"
        for requirement in item["requirements"]:
            level = requirement.get("student_level", "")
            gpa = requirement.get("minimum_gpa", "")
            response += f"\n- {level}: minimum {gpa}"

    if "evaluation_criteria" in item:
        response += "\n\nDeğerlendirme genelde şu kriterlere göre yapılıyor:"
        for criterion in item["evaluation_criteria"]:
            name = criterion.get("criterion", "")
            weight = criterion.get("weight", "")
            response += f"\n- {name}: {weight}"

    if "grant_amounts" in item:
        response += "\n\nHibe miktarları hareketlilik türüne göre değişiyor:"
        for grant in item["grant_amounts"]:
            mobility_type = grant.get("mobility_type", "")
            amount = grant.get("amount", "")
            response += f"\n- {mobility_type}: {amount}"

    if "details" in item:
        response += "\n\nEk bilgiler:"
        for detail in item["details"]:
            response += f"\n- {detail}"

    if "mobility_types" in item:
        response += "\n\nHareketlilik türleri:"
        for mobility in item["mobility_types"]:
            response += f"\n- {mobility}"

    if "countries_examples" in item:
        response += "\n\nÖrnek ülkeler:"
        for country in item["countries_examples"]:
            response += f"\n- {country}"

    if "note" in item:
        response += f"\n\nNot: {item['note']}"

    if "website" in item:
        response += f"\n\nDetaylı bilgi için: {item['website']}"

    if "partner_universities_url" in item:
        response += f"\n\nPartner üniversiteler için: {item['partner_universities_url']}"

    return response


def generate_club_response(item):
    """
    Generates a natural Turkish response for student club data.
    """

    name = item.get("name", "Bu kulüp")
    category = item.get("category", "Kategori bilgisi yok")
    description = item.get("description", "")
    activities = item.get("activities", [])

    templates = [
        f"{name}, {category} kategorisinde yer alan bir öğrenci kulübüdür. {description}",
        f"Evet, {name} adında bir kulüp var. Bu kulüp {category} alanıyla ilişkilendiriliyor. {description}",
        f"{name} kulübü, Üsküdar Üniversitesi öğrenci kulüpleri arasında yer alıyor. Kategorisi: {category}. {description}",
        f"Bu konuda öne çıkan kulüp {name}. {description}"
    ]

    response = random.choice(templates)

    if activities:
        response += "\n\nBu kulüple ilgili bazı etkinlik türleri:"
        for activity in activities[:4]:
            response += f"\n- {activity}"

    return response


def generate_clubs_list(items):
    """
    Generates a list response for club questions.
    """

    if not items:
        return "Kulüp datasında uygun bir kayıt bulamadım."

    response = "Üsküdar Üniversitesi kulüp datasında yer alan bazı kulüpler şunlar:\n"

    for item in items:
        name = item.get("name", "Kulüp adı yok")
        category = item.get("category", "Kategori yok")
        response += f"\n- {name} — {category}"

    response += "\n\nBelirli bir kulüp hakkında detay istersen kulüp adını yazarak sorabilirsin."

    return response


def generate_event_response(item):
    """
    Generates a natural Turkish response for event data.
    """

    title = item.get("title", "Etkinlik")
    date = item.get("date", "Tarih bilgisi yok")
    location = item.get("location", "Konum bilgisi yok")
    category = item.get("category", "Etkinlik")
    description = item.get("description", "")
    source_url = item.get("source_url", "")

    templates = [
        f"{title} adlı etkinlik {date} tarihinde düzenleniyor. Kategori olarak {category} altında görünüyor. Konum bilgisi: {location}. {description}",
        f"Etkinlik bilgilerine göre {title}, {date} tarihinde gerçekleşecek. Yer bilgisi: {location}. {description}",
        f"{date} tarihinde {title} isimli bir etkinlik var. Bu etkinlik {category} kategorisinde yer alıyor. {description}",
        f"Bu soruyla ilgili bulduğum etkinlik: {title}. Tarih: {date}. Konum: {location}. {description}"
    ]

    response = random.choice(templates)

    if source_url:
        response += f"\n\nEtkinlik sayfası: {source_url}"

    return response


def generate_events_list(items):
    """
    Generates a list response for event questions.
    """

    if not items:
        return "Etkinlik datasında uygun bir kayıt bulamadım."

    response = "Üniversite etkinlik datasında öne çıkan bazı etkinlikler şunlar:\n"

    for item in items:
        title = item.get("title", "Etkinlik adı yok")
        date = item.get("date", "Tarih bilgisi yok")
        category = item.get("category", "Etkinlik")
        response += f"\n- {title} — {date} — {category}"

    response += "\n\nBelirli bir etkinlik hakkında detay istersen etkinlik adını sorabilirsin."

    return response


def generate_research_response(item):
    """
    Generates a natural Turkish response for research data.
    """

    title = item.get("title", "Araştırma Bilgisi")
    category = item.get("category", "Araştırma")
    description = item.get("description", "")
    source_url = item.get("source_url", "")

    templates = [
        f"{title}, {category} kategorisinde yer alan bir araştırma başlığıdır. {description}",
        f"Araştırma bilgilerine göre {title} başlığı öne çıkıyor. Bu alan {category} kapsamında değerlendiriliyor. {description}",
        f"Bu konuda ilgili araştırma sayfası {title}. {description}",
        f"Üniversitenin araştırma yapısı içinde {title} önemli bir başlık olarak görünüyor. {description}"
    ]

    response = random.choice(templates)

    if source_url:
        response += f"\n\nİlgili sayfa: {source_url}"

    return response


def generate_research_list(items):
    """
    Generates a list response for research questions.
    """

    if not items:
        return "Araştırma datasında uygun bir kayıt bulamadım."

    response = "Üniversitenin araştırma datasında öne çıkan bazı başlıklar şunlar:\n"

    for item in items:
        title = item.get("title", "Başlık yok")
        category = item.get("category", "Kategori yok")
        response += f"\n- {title} — {category}"

    return response


def generate_unknown_response():
    """
    Generates a professional fallback response when no suitable answer is found.
    """

    return (
        "Bu soruya mevcut üniversite datasetlerimden net bir cevap bulamadım. "
        "Şu anda akademik kadro, Erasmus bilgileri, öğrenci kulüpleri, etkinlikler ve araştırma sayfaları hakkında yardımcı olabiliyorum. "
        "Sorunu bu başlıklardan biriyle ilgili biraz daha açık yazarsan daha iyi yardımcı olabilirim."
    )


def generate_response(question):
    """
    Main response generation function.
    It detects intent, searches the related dataset and returns a natural Turkish answer.
    """

    all_data = load_all_data()
    intent = detect_intent(question)

    if intent == "smalltalk":
        return generate_smalltalk_response(question)

    if intent == "unknown":
        return generate_unknown_response()

    if is_list_question(question):
        items = get_top_matches(question, intent, all_data, limit=6)

        if intent == "professors":
            return generate_professors_list(items)

        if intent == "clubs":
            return generate_clubs_list(items)

        if intent == "events":
            return generate_events_list(items)

        if intent == "research":
            return generate_research_list(items)

    item = search_dataset(question, intent, all_data)

    if item is None:
        return generate_unknown_response()

    if intent == "professors":
        return generate_professor_response(item)

    if intent == "erasmus":
        return generate_erasmus_response(item)

    if intent == "clubs":
        return generate_club_response(item)

    if intent == "events":
        return generate_event_response(item)

    if intent == "research":
        return generate_research_response(item)

    return generate_unknown_response()


if __name__ == "__main__":
    test_questions = [
        "Rowanda hoca hangi bolumde?",
        "Bilgisayar muhendisligi hocalari kimler?",
        "Erasmus hibesi ne kadar?",
        "Erasmus basvuru sartlari neler?",
        "Hangi kulupler var?",
        "Developer kulubu var mi?",
        "Mayis ayinda etkinlik var mi?",
        "Etkinlikler neler?",
        "Arastirma merkezleri neler?",
        "Merhaba nasilsin?",
        "Kafeterya nerede?"
    ]

    for question in test_questions:
        print("Question:", question)
        print("Answer:", generate_response(question))
        print("-----------------------------")