
🚀 Visual Mouse & Drawing Suite
Bu proje, bilgisayar kamerasını kullanarak el hareketleriyle temassız bir şekilde mouse kontrolü sağlamanıza ve ekran üzerinde dijital çizim yapmanıza olanak tanıyan bir bilgisayarlı görü (computer vision) uygulamasıdır.

✨ Özellikler
Uygulama iki ana moddan oluşmaktadır:

1. Mouse Modu (Sanal Mouse)
Hassas Kontrol: Yumruk yaparak mouse imlecini hareket ettirebilirsiniz.

Tıklama: Başparmak hareketi ile sol tık ve hızlı çift tık.

Sağ Tık: Serçe parmak kaldırıldığında tetiklenir.

Kaydırma (Scroll): İşaret ve orta parmak havada iken elinizi yukarı-aşağı hareket ettirerek sayfaları kaydırabilirsiniz.

ROI (İlgi Alanı): Ekranın ortasında belirlenen yeşil bölge, mouse hassasiyetini artırmak için özel bir kontrol alanı sunar.

2. Çizim Modu (Air Canvas)
Çizim Yapma: Sadece işaret parmağınızı kaldırarak ekranda çizim yapabilirsiniz.

İmleç Modu: İşaret ve orta parmak havada iken çizim yapmadan imleci gezdirebilirsiniz.

Renk Değiştirme: Başparmak hareketiyle 6 farklı renk arasında geçiş yapabilirsiniz.

Dinamik Kalınlık: Serçe parmağınızı kaldırarak çizgi kalınlığını artırabilirsiniz.

Silgi: Elinizi yumruk yaparak ekrandaki çizimleri silebilirsiniz.

Kaydetme & Temizleme: 'S' tuşu ile çiziminizi kaydedebilir, 'C' tuşu ile tuvali temizleyebilirsiniz.

🛠️ Kurulum
1. Gereksinimler
Sisteminizde Python 3.8+ yüklü olmalıdır. Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

Bash
pip install opencv-python mediapipe numpy pyautogui
2. Model Dosyası
Uygulama MediaPipe'ın yeni Tasks API'sini kullanmaktadır.

Hand Landmarker Model dosyasını indirin.

Kod içerisindeki HAND_MODEL_PATH değişkenini, modelin bilgisayarınızdaki yoluyla güncelleyin:
HAND_MODEL_PATH = r"C:/YOLUNUZ/hand_landmarker.task"

🎮 Kullanım Kılavuzu
Uygulamayı çalıştırdığınızda sizi şık bir ana menü karşılar:

[1]'e Basın: Mouse moduna girer.

[2]'ye Basın: Çizim moduna girer.

'Q'ya Basın: Modlardan menüye döner veya uygulamadan çıkar.

Kontrol Tablosu
Hareket / Tuş	Mouse Modu Fonksiyonu	Çizim Modu Fonksiyonu
Yumruk	Mouse Hareket Ettir	Silgi
İşaret Parmağı 👆	Bekleme	Çizim Yap
İşaret + Orta ✌️	Sayfa Kaydırma (Scroll)	Serbest İmleç
Başparmak 👍	Sol Tık / Çift Tık	Renk Değiştir
Serçe Parmak 🤙	Sağ Tık	Kalın Çizgi
'S' Tuşu	-	Çizimi Klasöre Kaydet
'C' Tuşu	-	Tuvali Temizle

🏗️ Teknik Detaylar
Yumuşatma (Smoothing): Mouse ve çizim hareketlerinde titremeyi önlemek için Üstel Hareketli Ortalama (EMA) algoritması kullanılmıştır.

Hızlandırma: Mouse hareketlerinde ani ivmelenmeleri algılayan özel bir ACCEL_FACTOR mekanizması eklenmiştir.

Görsel Arayüz: OpenCV kullanılarak hazırlanan HUD (Heads-up Display) ile aktif mod, koordinatlar ve renk paleti kullanıcıya anlık gösterilir.

📝 Notlar
Işığın yeterli olduğu bir ortamda en iyi performansı verir.

pyautogui.FAILSAFE özelliği etkindir; bir sorun olursa elinizi kameradan çekip mouse'u ekranın en köşesine sürükleyerek durdurabilirsiniz.
