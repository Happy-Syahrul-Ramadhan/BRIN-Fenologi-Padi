# 🌾 Aplikasi Klasifikasi Fenologi Padi

Aplikasi web untuk klasifikasi fenologi (fase pertumbuhan) padi menggunakan data Sentinel-1 SAR dan model Random Forest. Aplikasi ini dikembangkan untuk membantu monitoring dan analisis pertumbuhan tanaman padi di wilayah Indramayu, Jawa Barat.

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Instalasi](#-instalasi)
- [Konfigurasi](#-konfigurasi)
- [Penggunaan](#-penggunaan)
- [API Documentation](#-api-documentation)
- [Struktur Project](#-struktur-project)
- [Metodologi](#-metodologi)
- [Troubleshooting](#-troubleshooting)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

## 🚀 Fitur Utama

### 🗓️ Klasifikasi Berdasarkan Dasarian
- Klasifikasi fenologi padi berdasarkan periode dasarian (10 hari)
- 36 periode dasarian dalam satu tahun (Januari P1 - Desember P3)
- Interface slider dan dropdown untuk pemilihan periode

### 📅 Klasifikasi Berdasarkan Tanggal
- Klasifikasi fleksibel dengan rentang tanggal custom
- Data real-time dari Sentinel-1
- Input tanggal mulai dan akhir

### 🗄️ Custom Image Collection Support ⭐ NEW
- **Dukungan data custom** dari Google Earth Engine
- **Multi-platform support**: Sentinel-1, Sentinel-2, Landsat, dan data custom
- **Validasi otomatis** collection sebelum digunakan
- **Fleksibilitas tinggi** untuk penelitian spesifik
- Lihat [CUSTOM_COLLECTION_GUIDE.md](CUSTOM_COLLECTION_GUIDE.md) untuk panduan detail

### 🎯 Fase Pertumbuhan Padi

### 📊 Analisis dan Statistik
- Perhitungan area per fase dalam hektar
- Distribusi persentase setiap fase
- Visualisasi statistik dengan progress bar

### 🗺️ Visualisasi Peta Interaktif
- Peta berbasis Folium dengan kontrol layer
- Zoom dan navigasi interaktif
- Legenda dinamis dengan kode warna fase
- Mode fullscreen untuk analisis detail

### 🔄 Perbandingan Periode
- Analisis perubahan fase antar periode
- Tracking evolusi pertumbuhan padi
- Visualisasi trend temporal

## 💻 Teknologi yang Digunakan

### Backend
- **Python 3.8+** - Bahasa pemrograman utama
- **Flask** - Web framework
- **Google Earth Engine (GEE)** - Platform cloud computing geospasial
- **scikit-learn** - Machine learning library
- **Geemap** - Interactive mapping dengan Google Earth Engine

### Frontend
- **HTML5 & CSS3** - Struktur dan styling
- **Bootstrap 5** - UI framework responsif
- **JavaScript (ES6+)** - Interaktivitas client-side
- **Folium** - Visualisasi peta interaktif
- **Font Awesome** - Icons

### Data & Model
- **Sentinel-1 SAR** - Data satelit radar
- **Random Forest** - Algoritma machine learning
- **Earth Engine Assets** - Pre-processed image collections

## 📋 Persyaratan Sistem

### Hardware
- RAM minimal 8GB (16GB direkomendasikan)
- Storage minimal 2GB ruang kosong
- Koneksi internet stabil untuk akses Google Earth Engine

### Software
- Python 3.8 atau lebih baru
- Google Earth Engine account dan project
- Web browser modern (Chrome, Firefox, Safari, Edge)

## 🔧 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/Happy-Syahrul-Ramadhan/BRIN-Fenologi-Padi.git
cd BRIN-Fenologi-Padi
```


### 2. Install Dependencies
```bash
pip install flask
pip install earthengine-api
pip install geemap
pip install scikit-learn
pip install numpy
pip install pandas
```

### 4. Setup Google Earth Engine
```bash
earthengine authenticate
```

## ⚙️ Konfigurasi

### Environment Variables

ubah file `.env.example` menjadi `.env` di root directory:
```env
FLASK_ENV=development
FLASK_DEBUG=True
EE_PROJECT=your project
```

### Google Earth Engine Setup
1. Buat akun di [Google Earth Engine](https://earthengine.google.com/)
2. Daftarkan project dengan nama `try-spasial`
3. Upload assets yang diperlukan:
   - Training points: `projects/try-spasial/assets/output_shapefile`
   - Image collection: `your data asset image`

### Konfigurasi Aplikasi
Edit parameter di `app.py`:
```python
SCALE = 10  # Resolusi spasial (meter)
MAX_TRAINING_POINTS = 2000  # Maksimal titik training
BANDS_SELECTED = ['VV_int', 'VH_int', 'RPI', 'API', 'NDPI', 'RVI', 'angle']
```

## 📖 Penggunaan

### 1. Menjalankan Aplikasi
```bash
python app.py
```
Akses aplikasi di: http://localhost:5000

### 2. Interface Utama

#### Halaman Beranda (`/`)
- Informasi umum aplikasi
- Panduan penggunaan
- Link ke halaman klasifikasi

#### Halaman Klasifikasi (`/rice-phase`)
- Panel kontrol di sisi kiri
- Peta interaktif di sisi kanan
- Panel statistik dan informasi

### 3. Workflow Klasifikasi

#### Klasifikasi dengan Data Default:
1. Pilih "Gunakan Data Default (Sentinel-1)"
2. Pilih dasarian menggunakan slider/dropdown ATAU input tanggal
3. Klik tombol "Mulai Klasifikasi" atau "Klasifikasi by Tanggal"
4. Tunggu proses loading
5. Lihat hasil di peta dan panel statistik

#### Klasifikasi dengan Custom Collection: ⭐ NEW
1. Pilih "Gunakan Data Custom"
2. Input path asset collection GEE (contoh: `users/username/my_collection`)
3. Pilih tipe collection (Sentinel-1, Sentinel-2, Landsat, atau Custom)
4. Klik "Validasi Collection" untuk memastikan data dapat diakses
5. Setelah validasi berhasil, lakukan klasifikasi seperti biasa
6. Hasil akan menggunakan data custom Anda

#### Panel Statistik:
- Total area per fase dalam hektar
- Persentase distribusi setiap fase
- Jumlah kelas yang terdeteksi
- Fase dominan di area studi
- Informasi sumber data yang digunakan

## 📚 API Documentation

### Endpoint Utama

#### `GET /`
Halaman beranda aplikasi.

#### `GET /rice-phase`
Halaman utama klasifikasi dengan peta interaktif.

#### `POST /api/get_classification_map_dasarian`
Klasifikasi berdasarkan dasarian.

**Request Body:**
```json
{
  "dasarian": 15
}
```

**Response:**
```json
{
  "success": true,
  "map_html": "<html>...</html>",
  "area_stats": {
    "class_distribution": [...],
    "total_area_hectares": 1234.56
  },
  "dasarian_info": {
    "display_name": "Mei P2",
    "month_name": "Mei"
  }
}
```

#### `POST /api/classify-by-date`
Klasifikasi berdasarkan rentang tanggal.

**Request Body:**
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-15"
}
```

#### `GET /api/model-info`
Informasi tentang model yang digunakan.

#### `POST /api/analyze-phase`
Analisis mendalam fase pertumbuhan.

#### `POST /api/compare_periods`
Perbandingan antar periode.

#### `GET /api/collection_info`
Informasi tentang koleksi data yang tersedia.

### Error Handling
Semua endpoint mengembalikan error dalam format:
```json
{
  "error": "Deskripsi error",
  "success": false
}
```

## 📁 Struktur Project

```
brin-fenologi-padi/
├── app.py                 # Aplikasi Flask utama
├── model/                 # Folder untuk model
│   └── rf_model.pkl       # Model
├── templates/            # Template HTML
│   ├── base.html         # Template dasar
│   ├── index.html        # Halaman beranda
│   └── rice_phase.html   # Halaman klasifikasi utama
├── __pycache__/          # Python cache
├── .git/                 # Git repository
├── .env.example          # Contoh file konfigurasi environment
├── .gitignore            # File untuk mengabaikan file dan folder tertentu di Git
├── README.md             # Dokumentasi ini
└── requirements.txt      # Daftar dependensi yang diperlukan

```

### Penjelasan File Utama

#### `app.py`
- **Konfigurasi**: Setup Flask, Earth Engine, dan parameter
- **Model Loading**: Fungsi untuk load model Random Forest
- **Data Processing**: Fungsi untuk preprocessing data Sentinel-1
- **Classification**: Algoritma klasifikasi menggunakan EE dan RF
- **API Endpoints**: Route handler untuk semua endpoint
- **Visualization**: Pembuatan peta dengan Folium/Geemap

#### `templates/rice_phase.html`
- **UI Components**: Interface untuk kontrol klasifikasi
- **Interactive Map**: Container untuk peta Folium
- **Statistics Panel**: Tampilan statistik dan informasi
- **JavaScript**: Logika client-side untuk interaktivitas
- **Responsive Design**: CSS untuk berbagai ukuran layar

#### `rf_model.pkl`
- Model Random Forest yang telah dilatih
- Input: 7 band SAR (VV, VH, RPI, API, NDPI, RVI, angle)
- Output: 4 kelas fase pertumbuhan padi

## 🔬 Metodologi

### 1. Data Input
**Sentinel-1 SAR Bands:**
- **VV_int**: Intensitas polarisasi VV
- **VH_int**: Intensitas polarisasi VH
- **RPI**: Radar Polarization Index
- **API**: Angle Polarization Index  
- **NDPI**: Normalized Difference Polarization Index
- **RVI**: Radar Vegetation Index
- **angle**: Sudut insiden radar

### 2. Preprocessing
- Noise reduction dan filtering
- Geometric correction
- Radiometric calibration
- Index calculation (RPI, API, NDPI, RVI)

### 3. Machine Learning
**Random Forest Classifier:**
- Ensemble method dengan multiple decision trees
- Robust terhadap noise dan outliers
- Dapat menangani fitur dengan skala berbeda
- Memberikan feature importance

**Training Process:**
- Data split: 80% training, 20% validation
- Cross-validation untuk parameter tuning
- Feature selection berdasarkan importance
- Model evaluation dengan confusion matrix

### 4. Area of Interest (AOI)
- **Lokasi**: Kabupaten Indramayu, Jawa Barat
- **Koordinat**: Polygon yang mencakup area sawah utama
- **Alasan Pemilihan**: 
  - Sentra produksi padi nasional
  - Pola tanam yang teratur
  - Representatif untuk sawah irigasi

### 5. Temporal Analysis
**Dasarian System:**
- Sistem pembagian bulan menjadi 3 periode (10-10-sisa hari)
- Periode 1: Tanggal 1-10
- Periode 2: Tanggal 11-20  
- Periode 3: Tanggal 21-akhir bulan
- Total 36 periode dalam setahun

## 🔧 Troubleshooting

### Common Issues

#### 1. Google Earth Engine Authentication Error
```
Error: Please authorize access to your Earth Engine account
```
**Solution:**
```bash
earthengine authenticate
```

#### 2. Model Loading Error
```
Error: Model file not found
```
**Solution:**
- Pastikan `rf_model.pkl` ada di root directory
- Check file permissions
- Verify model compatibility

#### 3. Memory Error
```
MemoryError: Unable to allocate array
```
**Solution:**
- Kurangi `MAX_TRAINING_POINTS`
- Gunakan machine dengan RAM lebih besar
- Optimize image processing parameters

#### 4. Slow Classification
**Possible Causes:**
- Large AOI size
- High resolution setting
- Network latency to GEE servers

**Solutions:**
- Reduce `SCALE` parameter
- Limit AOI size
- Use asset collections instead of real-time data

#### 5. Empty Classification Results
**Check:**
- Date range availability
- Cloud coverage
- AOI intersection with data

### Debug Mode
Aktifkan debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization
1. **Use Asset Collections**: Lebih cepat dari real-time processing
2. **Optimize Scale**: Sesuaikan dengan kebutuhan akurasi
3. **Cache Results**: Implement caching untuk queries yang sering
4. **Batch Processing**: Process multiple periods sekaligus

## 🤝 Kontribusi

### Guidelines
1. Fork repository ini
2. Buat branch untuk fitur baru: `git checkout -b feature/nama-fitur`
3. Commit changes: `git commit -m 'Add some feature'`
4. Push to branch: `git push origin feature/nama-fitur`
5. Submit Pull Request

### Development Setup
```bash
# Install development dependencies
pip install flask-debugtoolbar
pip install pytest
pip install black  # Code formatter
```

### Code Style
- Gunakan Black untuk formatting
- Follow PEP 8 guidelines
- Add docstrings untuk functions
- Include type hints where possible

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Klasifikasi dasarian dengan asset collection
- ✅ Visualisasi peta interaktif
- ✅ Statistik area dan distribusi fase
- ✅ Responsive web interface
- ✅ API endpoints lengkap

### Planned Features
- 🔄 Model accuracy display
- 🔄 Export hasil ke CSV/GeoJSON
- 🔄 Multi-region support
- 🔄 Time series analysis
- 🔄 Automated reporting

## 📄 Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- Google Earth Engine Team untuk platform cloud computing
- Sentinel-1 Mission untuk data SAR berkualitas tinggi
- Komunitas open source untuk libraries yang digunakan

---

**⚠️ Disclaimer**: Aplikasi ini dikembangkan untuk tujuan penelitian dan edukasi. Hasil klasifikasi dapat dipengaruhi oleh kondisi cuaca, kualitas data, dan faktor lingkungan lainnya. Selalu lakukan validasi lapangan untuk aplikasi operasional.

**📊 Data Usage**: Aplikasi ini menggunakan data Sentinel-1 yang tersedia bebas melalui Google Earth Engine. Penggunaan data harus sesuai dengan terms of service ESA dan Google.

**🔒 Privacy**: Aplikasi tidak menyimpan data pribadi pengguna. Semua processing dilakukan secara real-time dan tidak ada data yang disimpan permanent di server.
