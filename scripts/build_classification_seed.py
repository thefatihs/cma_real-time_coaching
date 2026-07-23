"""Rebuild the versioned general synthetic Turkish classification seed."""

import json
from pathlib import Path

OUTPUT_PATH = Path("data/synthetic/classification_seed.jsonl")

BUSINESS_LABELS = (
    "product_information",
    "price_objection",
    "cancellation_request",
    "technical_issue",
    "complaint",
    "renewal_interest",
    "churn_risk",
)

SECONDARY_LABEL = {
    "product_information": "price_objection",
    "price_objection": "churn_risk",
    "cancellation_request": "churn_risk",
    "technical_issue": "complaint",
    "complaint": "technical_issue",
    "renewal_interest": "product_information",
    "churn_risk": "price_objection",
}

TEXTS: dict[str, tuple[str, ...]] = {
    "product_information": (
        "Yeni pakette hangi kanallar var? Ücreti de beklediğimden yüksek görünüyor.",
        "Yurt dışı kullanım kapsamını anlatır mısınız, bu fiyata değer mi anlamadım.",
        "Aile planına kaç kişi ekleniyor? Aylık bedel bana biraz pahalı geldi.",
        "Bulut alanı kaç gigabayt, ayrıca ek alan ücreti neden bu kadar yüksek?",
        "Taahhütsüz seçenekte neler sunuluyor? Liste fiyatını makul bulmadım.",
        "Kurumsal plandaki raporları öğrenmek istiyorum ama teklif bütçemi aşıyor.",
        "Güvenlik özelliği tam olarak ne yapıyor; bunun için alınan ek ücret fazla.",
        "Uygulamadaki ebeveyn kontrolü nasıl çalışıyor?",
        "Bu paket aynı anda kaç cihazda kullanılabilir?",
        "Hız yükseltme seçeneğinin teknik sınırı nedir?",
        "Seyahat paketinin geçerli olduğu ülkeleri sayabilir misiniz?",
        "Yeni üyelikte kurulum süreci nasıl ilerliyor?",
        "Fatura bildirimlerini e-posta yerine uygulamadan alabilir miyim?",
        "Ek kullanıcı tanımlama hakkı hangi planlarda bulunuyor?",
        "Hizmeti dondurma özelliğinin koşulları nelerdir?",
        "Standart plan ile gelişmiş planın farkını kısa anlatın.",
        "Cihaz koruma paketinin kapsam dışı bıraktığı durumlar neler?",
        "Numara taşıma sırasında mevcut hizmet kesilir mi?",
        "Yeni dönem paketi hangi özellikleri içeriyor? Fiyat tablosu da oldukça yüksek.",
        "Yedekleme hizmeti ne sıklıkta çalışıyor, ek bedeli düşürme seçeneği var mı?",
        "Bu pakette canlı destek bulunuyor mu?",
        "Kampanya bittikten sonra hangi özellikler devam eder?",
        "Modem kiralama ile satın alma arasındaki fark nedir?",
        "Çocuk profili için yaş sınırı uygulanıyor mu?",
        "uygulama içi arama ozelligi nerde nasıl kullanılıyor",
        "Sınırsız denilen pakette adil kullanım sınırı var mı, fiyatı da yüksek geldi.",
        "Hediye kullanım hakkı başka bir hesaba aktarılabiliyor mu?",
        "Yeni cihaz desteği hangi marka ve modelleri kapsıyor?",
        "Paket değişince biriken haklar siliniyor mu?",
        "Abonelik olmadan yalnızca tek hizmeti satın almak mümkün mü?",
    ),
    "price_objection": (
        "Aylık bedel çok yükseldi; böyle giderse başka bir hizmete bakacağım.",
        "Bu zam oranını kabul edemiyorum, yenilememeyi ciddi ciddi düşünüyorum.",
        "Rakip teklif yarı fiyatına, burada kalmam için daha iyi bir ücret lazım.",
        "Kurulumla birlikte çıkan toplam tutar bütçemi aştı; devam etmek zor.",
        "Aynı paket için geçen aya göre çok fazla istiyorsunuz, alternatiflere bakarım.",
        "İndirim yapılmazsa sözleşme sonunda ayrılabilirim.",
        "Ek kullanım ücretleri yüzünden hesabımı başka yere taşımayı düşünüyorum.",
        "Bu özellik için istenen on lira bile gereksiz yüksek.",
        "Faturadaki hizmet bedeli konuştuğumuz rakamdan pahalı.",
        "Öğrenci bütçesine uygun dediniz ama bu fiyat hiç uygun değil.",
        "Peşin ödemede bile indirim olmaması beni şaşırttı.",
        "Cihaz kirası her ay alınacaksa paketi istemiyorum.",
        "Erken yenileme teklifinin avantajı yok, normal fiyatla aynı.",
        "Kullanmadığım ek hizmet için ücret ödemek istemem.",
        "Vergiler eklendiğinde toplam bedel gereğinden fazla oluyor.",
        "Daha düşük hız için bu kadar ödeme mantıklı değil.",
        "İlk ay ücretsiz denmişti, yine de ücret yansımış.",
        "Tek seferlik aktivasyon bedelini çok yüksek buluyorum.",
        "Yıllık toplamı görünce teklif cazibesini kaybetti; başka seçenekleri araştıracağım.",
        "Bu ücret devam ederse gelecek dönem burada kalacağımdan emin değilim.",
        "fatura cok geldi bu rakamı kabul etmiyom",
        "Ben sadece temel hizmet istiyorum, bu paket gereksiz pahalı.",
        "Eski müşteriye yeni müşteriden fazla fiyat çıkması adil değil.",
        "Ücret iki kez alınmış; toplam tutara itiraz ediyorum.",
        "İndirim sözü olmadan bu fiyat üzerinden karar veremem.",
        "Ben fiyatı sormuyorum, doğrudan bu tutarın fazla olduğunu söylüyorum.",
        "Kısa süre kullanacağım için aylık bedel bana uygun düşmüyor.",
        "Taahhüt cezasıyla birlikte maliyet kabul edilebilir seviyede değil.",
        "Kampanya adı indirim ama önceki faturamdan daha pahalı.",
        "Daha ucuz pakete geçsem bile ek ücret alınmasına itiraz ediyorum.",
    ),
    "cancellation_request": (
        "Aboneliğimi şimdi kapatın; zaten başka bir sağlayıcıya geçmeye karar verdim.",
        "Bugün iptal işlemini başlatın, bu hizmette kalmayacağım.",
        "Üyeliği sona erdirme talebimi kaydedin; alternatifimi seçtim.",
        "Paketi derhal iptal etmek istiyorum, gelecek ay devam etmeyeceğim.",
        "Sözleşmemi kapatın lütfen; başka bir çözüm kullanacağım.",
        "Bu görüşmede iptali tamamlayalım, kararım kesin.",
        "Hesabımı tamamen kapatın; artık sizden hizmet almayı düşünmüyorum.",
        "Otomatik yenilemeyi değil, aboneliğin tamamını iptal edin.",
        "İptal formunu gönderin, bugün dolduracağım.",
        "Hizmetin bu ay sonunda sonlandırılmasını talep ediyorum.",
        "Üyeliğimi kapatma konusunda kararımı değiştirmeyeceğim.",
        "iptal edecem işlemi baslatın lütfen",
        "Hatlarımdan yalnızca bu aboneliği hemen kapatmak istiyorum.",
        "Deneme süresi bitmeden hesabımı silip üyeliği sonlandırın.",
        "Fesih talebimi hangi kanaldan onaylayacağımı söyleyin.",
        "Paket değişikliği istemiyorum, doğrudan iptal istiyorum.",
        "Hizmeti dondurmayın; tamamen kapatılmasını istiyorum.",
        "Bugünün tarihiyle sözleşme feshi talebi oluşturun.",
        "İptal için aradım. Bir süre daha beklemek gibi bir niyetim yok.",
        "Üyeliğimi kesin olarak sonlandıracağım; gerekli adımları şimdi başlatalım.",
        "Temsilciye bağlayıp kapatma işlemini tamamlar mısınız?",
        "Artık kullanmıyorum, hesabın ve paketin iptalini rica ederim.",
        "Bir sonraki fatura çıkmadan aboneliği kapatın.",
        "Taahhüt bedelini biliyorum, yine de iptal etmek istiyorum.",
        "Kararsız değilim; hizmeti bugün sonlandırın ve başka yere geçeceğim.",
        "İptal talebimi geri çekmedim, ayrılma kararım geçerli.",
        "Bu numaradaki ücretli servisi hemen kapatın.",
        "Sözleşme bitişini beklemeden fesih işlemi istiyorum.",
        "Hesap kapanış onayını e-postayla gönderin.",
        "Bütün ek paketlerle beraber ana aboneliği iptal edin.",
    ),
    "technical_issue": (
        "Bağlantı her akşam kopuyor ve destek kaydım hâlâ çözülmedi.",
        "Uygulama açılırken kapanıyor; aynı sorunu üç kez bildirdim.",
        "Modem çevrim içi görünse de internet yok, bu kesintiden bıktım.",
        "Ses görüşmede sürekli kesiliyor ve hizmet kalitesinden memnun değilim.",
        "Şifre sıfırlama bağlantısı çalışmıyor; günlerdir çözüm bekliyorum.",
        "Güncellemeden sonra cihaz bağlanmıyor, verilen destek yetersiz kaldı.",
        "Ekranda hata kodu çıkıyor ve kimse nedenini açıklamadı.",
        "İki aşamalı doğrulama kodu telefonuma ulaşmıyor.",
        "Dosya yükleme yüzde doksanda takılıp başa dönüyor.",
        "Kablosuz ağ yalnızca bir odada tamamen kayboluyor.",
        "Uygulama bildirimleri açık olduğu hâlde hiç gelmiyor.",
        "Cihaz yeniden başlayınca kayıtlı ayarlar siliniyor.",
        "arama yapınca ses var görüntü donuyo",
        "Fatura sayfası boş beyaz ekran gösteriyor.",
        "Kurulum tamamlandı mesajından sonra sistem çevrim dışı kalıyor.",
        "Bluetooth eşleşmesi her seferinde başarısız oluyor.",
        "Yurt dışında veri bağlantısı etkinleşmiyor.",
        "Gelen aramalar doğrudan meşgule düşüyor.",
        "Dünden beri hizmete erişemiyorum. Daha önce önerilen yeniden başlatma işe yaramadı.",
        "Yeni sürümde kamera izni algılanmıyor; bu yüzden görüşme başlatamıyorum.",
        "modemin isiklari yanıyor ama sayfalar acılmıyo",
        "Tek kullanımlık kod sürekli süresi dolmuş diyor.",
        "Hesap eşitleme işlemi eski verileri geri getiriyor.",
        "Ethernet bağlı olmasına rağmen IP adresi alınamıyor.",
        "Çağrı sırasında mikrofon kendiliğinden sessize geçiyor ve bu durum çok can sıkıcı.",
        "Arıza kaydı kapatılmış ama bağlantı sorunu devam ediyor.",
        "İndirme hızı ölçümde söz verilen değerin çok altında.",
        "Uygulamadan çıkış yaptığımda tekrar giriş ekranı yüklenmiyor.",
        "Cihaz uzaktan kumanda komutlarını gecikmeli algılıyor.",
        "Ödeme tamamlandıktan sonra paket hesabıma tanımlanmadı.",
    ),
    "complaint": (
        "Son görüşmede kaba davranıldı, ayrıca uygulamadaki hata hâlâ sürüyor.",
        "Üç gündür çözüm bekliyorum ve bağlantı sorunum devam ediyor.",
        "Arıza kaydı haber verilmeden kapatılmış; internetim yine çalışmıyor.",
        "Destek ekibi beni sürekli aktardı, üstelik giriş hatası çözülmedi.",
        "Verilen saat aralığında gelinmedi ve kurulum hâlâ tamamlanmadı.",
        "Kesinti hakkında bilgilendirme yapılmaması kabul edilemez, hizmet de yok.",
        "Aynı teknik problemi tekrar anlatmaktan yoruldum; ses hâlâ kesiliyor.",
        "Teslimat sürecinde hiçbir aşamada bilgi verilmedi.",
        "Temsilcinin açıklaması sorumu geçiştirmekten ibaretti.",
        "Bekleme süresi kırk dakikayı geçti, bu hizmetten memnun değilim.",
        "Talebim yanlış kategoriye alınmış ve günler kaybettim.",
        "Söz verilen geri dönüş yapılmadığı için şikayetçiyim.",
        "Paket içeriği satış sırasında eksik anlatılmış.",
        "İade sürecinde her temsilci farklı bilgi verdi.",
        "Randevu iki kez ertelendi, planım tamamen bozuldu.",
        "Gizli bir ücret sonradan faturaya eklenmiş, bu yaklaşım doğru değil.",
        "Sorunumu dinlemeden hazır cevap verilmesi rahatsız edici.",
        "Kayıt numaram olmasına rağmen talebim sistemde bulunamadı.",
        "Kurulum ekibi geç geldi. Üstelik cihazı çalışır durumda bırakmadı.",
        "Çağrı sürekli düşüyor ve tekrar aradığımda baştan anlatmam isteniyor.",
        "hic memnun kalmadım kimse yardımcı olmadı",
        "İptal etmediğim bir hizmet kapatılmış, süreç yönetimi çok kötü.",
        "Kampanya koşulları sonradan değiştirilmiş gibi görünüyor.",
        "Yanlış adrese gönderim yapıldığı için mağdur oldum.",
        "Bağlantı arızası sürerken kaydın çözüldü denmesine itiraz ediyorum.",
        "Teknik ekip gelmedi ve bütün gün boşuna bekledim.",
        "Soruma yanıt aldım ama temsilcinin üslubu uygun değildi.",
        "Bildirim mesajları gece yarısı geliyor, bu durum rahatsız edici.",
        "Hesabımda iznim olmadan değişiklik yapılmış.",
        "Çözüm süresi hakkında gerçekçi bilgi verilmedi.",
    ),
    "renewal_interest": (
        "Yenilemek istiyorum; yeni dönemde pakete hangi özellikler ekleniyor?",
        "Bir yıl daha devam etmeyi düşünüyorum, seçenekleri ayrıntılı anlatın.",
        "Sözleşmeyi uzatacağım; üst paketin kapsamını da öğrenebilir miyim?",
        "Mevcut hizmeti yenileme niyetim var, yeni haklar neler?",
        "Uygun yenileme teklifini seçmek istiyorum; plan farklarını açıklayın.",
        "Aboneliği sürdüreceğim, yıllık paketin içeriğini paylaşın.",
        "Yenileme işlemini yapalım; aile seçeneği bana uygun mu?",
        "Sözleşmem bitmeden aynı paketle devam etmek istiyorum.",
        "Yeni dönem için erken yenileme teklifini kabul edebilirim.",
        "Hizmetten memnunum, üyeliğimi bir yıl uzatalım.",
        "Otomatik yenilemeyi açmak istiyorum.",
        "Mevcut numaramla devam edip paketi yenileyeceğim.",
        "Yenileme tarihini öne çekmek mümkünse işlem yapalım.",
        "İki yıllık uzatma seçeneğini değerlendirmek istiyorum.",
        "Kampanyalı koşullar geçerliyse bugün yenileyebilirim.",
        "Aynı fiyat korunursa aboneliği sürdürmeye hazırım.",
        "Yeni sözleşme metnini onaylayıp devam edeceğim.",
        "Üyeliğin kesintisiz sürmesi için yenilemeyi şimdi tamamlayalım.",
        "Yeni dönem paketlerini inceledim. Orta seviye olanla devam etmek istiyorum.",
        "Sözleşme bitişinde hizmet kapanmasın; yenileme kaydımı oluşturun.",
        "yenileme yapmak istiyom hangi paket uygun",
        "Bu yıl da aynı hizmeti kullanmaya devam edeceğim.",
        "Sadakat teklifini kabul edersem süreyi uzatmak istiyorum.",
        "Yenileme onayımı telefon üzerinden verebilir miyim?",
        "Devam etmeye karar verdim; yeni dönemde destek kapsamı nedir?",
        "Üyeliği uzatacağım, taşınma hâlinde adres değişikliği yapılabiliyor mu?",
        "Ailem de kullanacak, bu nedenle daha geniş paketle yenilemek istiyorum.",
        "Kesinti yaşamadan yıllık planı tekrar başlatabilir misiniz?",
        "Yeni dönem için ödeme yöntemini seçip sözleşmeyi uzatalım.",
        "Deneme üyeliğini ücretli yıllık üyeliğe çevirmek istiyorum.",
    ),
    "churn_risk": (
        "Başka sağlayıcıları inceliyorum; buradaki fiyat da kararımı olumsuz etkiliyor.",
        "Yenileyip yenilememekte kararsızım, ücret biraz daha artarsa giderim.",
        "Rakipte benzer paket daha ucuz, sözleşme bitince geçebilirim.",
        "Şimdilik iptal istemiyorum ama bu fiyatla uzun süre kalmam.",
        "Devam etmek konusunda tereddütlüyüm; daha uygun teklif bekliyorum.",
        "Bir sonraki dönem başka seçenek denemeyi düşünüyorum çünkü burası pahalı.",
        "Henüz karar vermedim, toplam maliyet düşmezse yenilemeyeceğim.",
        "Taahhüt bitince hizmetleri yeniden karşılaştıracağım.",
        "Arkadaşımın kullandığı alternatife geçme fikrim var.",
        "Şu an kapatmayacağım ama memnuniyetim böyle sürerse ayrılabilirim.",
        "Yeni dönemde burada kalacağıma dair söz veremem.",
        "Bir süre daha deneyeceğim, sonra başka çözüme bakabilirim.",
        "Sözleşmeyi yenilememe ihtimalim oldukça yüksek.",
        "Rakibin deneme paketini kullanıp karar vereceğim.",
        "Taşındığım yerde farklı bir sağlayıcı seçebilirim.",
        "Hizmete eskisi kadar ihtiyaç duymuyorum, devam etmeyebilirim.",
        "Ailem başka bir firmaya geçmemizi öneriyor.",
        "Şimdilik beklemedeyim; kalmak için ikna olmuş değilim.",
        "Ücret indirimi olmazsa sözleşme sonunda alternatif arayacağım.",
        "Paketin değeri bu fiyata göre düşük, yenileme konusunda kuşkuluyum.",
        "baska yere gecsem mi diye dusunuyom fiyat cok arttı",
        "Hemen iptal talebim yok, yalnızca devam kararım belirsiz.",
        "Önümüzdeki ay kullanımımı azaltıp başka hizmetleri deneyeceğim.",
        "Yeni teklif gelmezse hesabı açık tutmanın anlamı kalmayabilir.",
        "Fiyat yine yükselirse ayrılmayı düşüneceğim.",
        "İndirim bitince rakip paketlere geçme olasılığım var.",
        "Henüz fesih istemiyorum fakat memnun kalmazsam uzatmayacağım.",
        "Sadece kısa süre daha kullanıp sonra karar vereceğim.",
        "Yeni evimde bu hizmet yerine farklı teknoloji seçebilirim.",
        "Devam seçeneğini açık bırakıyorum ama başka teklifler daha cazip.",
    ),
}

NO_ACTION_TEXTS = (
    "İptal etmek istemiyorum.",
    "Fiyatı sadece merak ettim.",
    "Şu anda herhangi bir şikayetim yok.",
    "Günaydın, görüşme saatimizi teyit etmek için aradım.",
    "Verdiğiniz bilgiyi not aldım, teşekkür ederim.",
    "Şimdilik başka bir sorum bulunmuyor.",
    "Yanlış numarayı aramışım, kusura bakmayın.",
    "Temsilciye teşekkür etmek istedim, işlemim tamamlandı.",
    "Bugün hava oldukça sıcak, kolay gelsin.",
    "Bekleyebilirim, acelem yok.",
    "Adımı doğru yazdığınızı teyit eder misiniz?",
    "Evet, güvenlik sorusunun yanıtı doğru.",
    "Görüşmeyi daha sonra sürdürmek üzere kapatabiliriz.",
    "Sadece kayıt numaramı öğrenip çıkacağım.",
    "Sesiniz geliyor, beni duyuyor musunuz?",
    "Bir dakika hatta kalacağım.",
    "Adres bilgim değişmedi.",
    "Mesajınızı aldım ve okudum.",
    "Bugün için başka işlem yapmayalım.",
    "Hayır, aboneliği kapatma gibi bir düşüncem yok.",
    "Hizmetle ilgili kötü bir deneyim yaşamadım.",
    "Ben ücret yüksek demedim, yalnızca rakamı tekrar sordum.",
    "Teknik bir arıza yok; cihazın rengini merak etmiştim.",
    "Yenileme kararı vermedim, sadece bitiş tarihini teyit ettim.",
    "şikayetçi değilim yanlış anlasıldı",
    "Merhaba, önceki görüşmenin tarihini öğrenebilir miyim?",
    "İyi çalışmalar, bağlantı kuruldu mu diye kontrol ettim.",
    "Kimlik doğrulama adımını tamamladım.",
    "E-postadaki bağlantıyı gördüm, henüz açmadım.",
    "Bana dönüş yapmanıza gerek kalmadı.",
    "Sorun yok, yanlışlıkla yardım düğmesine bastım.",
    "Teşekkürler, açıklamanız yeterli oldu.",
    "Şu an konuşmaya uygun değilim, sonra ben ararım.",
    "Telefon numaramın son iki hanesini teyit ediyorum.",
    "Paket adını bir kez daha söyler misiniz?",
    "Herhangi bir değişiklik talep etmiyorum.",
    "Görüşme kalite kaydı için onay veriyorum.",
    "Bekleme müziği biraz yüksek ama bu bir şikayet değil.",
    "İptal kelimesini örnek olarak söyledim, işlem istemiyorum.",
    "Ücret bilgisini aldım; pahalı ya da ucuz olduğuna karar vermedim.",
    "Tamamdır iyi günler başka bişi yok",
)

SUPPLEMENTAL_TEXTS = {
    "cancellation_request": (
        "Bu ek hizmeti artık istemiyorum, hemen iptal kaydı açın.",
        "Üyeliğimin kapanışını bugün tamamlamanızı rica ediyorum.",
        "Kararım net; hesabı dondurmak yerine tamamen sonlandırın.",
        "Yeni teklif dinlemek istemiyorum, doğrudan fesih işlemi yapın.",
        "Paketin bir sonraki döneme uzamasını önleyip şimdi kapatın.",
        "Tüm kullanım haklarıyla birlikte aboneliğimi iptal edin.",
        "Kapatma onayını aldıktan sonra görüşmeyi bitirebiliriz.",
        "Deneme hesabını ücretliye çevirmeden iptal etmek istiyorum.",
        "Bugün itibarıyla hizmet sonlandırma talebi oluşturur musunuz?",
        "Ek cihaz aboneliğini değil ana sözleşmeyi iptal ettiriyorum.",
        "Fesih bedeli çıkabilir, yine de hesabı kapatma işlemine devam edin.",
    ),
    "renewal_interest": (
        "Yeni dönem için devam onayımı vermek istiyorum.",
        "Sözleşmemi aynı koşullarla uzatabilirsek hemen yenileyelim.",
        "Üyeliğin bitmesini istemiyorum, yenileme işlemini başlatın.",
        "Bir sonraki yıl için mevcut planı sürdürmeye karar verdim.",
        "Yıllık aboneliği tekrar satın alıp hizmete devam edeceğim.",
        "Yeni dönem sözleşmesini onaylamaya hazırım.",
        "Sadakat teklifini seçtim, üyeliğimi uzatın lütfen.",
        "Yenileme seçeneklerini inceledim ve standart planla devam edeceğim.",
        "Aboneliği kesintisiz sürdürmek için bugün yenilemek istiyorum.",
        "Süre bitmeden iki yıllık uzatma işlemini tamamlayalım.",
        "Devam kararım kesin, otomatik yenilemeyi etkinleştirin.",
    ),
}


def main() -> None:
    records: list[dict[str, object]] = []
    for label in BUSINESS_LABELS:
        texts = TEXTS[label]
        if len(texts) != 30:
            raise ValueError(f"{label} must contain exactly 30 source texts")
        for index, text in enumerate(texts):
            split, local_index = _split_for_business_index(index)
            labels = [label]
            if local_index < _secondary_count(split):
                labels.append(SECONDARY_LABEL[label])
            records.append(
                _record(
                    f"synthetic_{label}_{index + 1:03d}",
                    text,
                    labels,
                    split,
                )
            )

    for label, texts in SUPPLEMENTAL_TEXTS.items():
        for index, text in enumerate(texts):
            split = "train" if index < 7 else "validation" if index < 9 else "test"
            records.append(
                _record(
                    f"synthetic_{label}_supplemental_{index + 1:03d}",
                    text,
                    [label],
                    split,
                )
            )

    for index, text in enumerate(NO_ACTION_TEXTS):
        split = "train" if index < 25 else "validation" if index < 33 else "test"
        records.append(
            _record(
                f"synthetic_no_action_{index + 1:03d}",
                text,
                ["no_action"],
                split,
            )
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def _split_for_business_index(index: int) -> tuple[str, int]:
    if index < 18:
        return "train", index
    if index < 24:
        return "validation", index - 18
    return "test", index - 24


def _secondary_count(split: str) -> int:
    return 7 if split == "train" else 2


def _record(
    example_id: str,
    text: str,
    labels: list[str],
    split: str,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "conversation_id": f"conversation_{example_id}",
        "text": text,
        "labels": labels,
        "split": split,
        "source": "synthetic",
    }


if __name__ == "__main__":
    main()
