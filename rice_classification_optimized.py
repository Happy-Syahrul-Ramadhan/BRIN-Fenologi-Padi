import ee
import geemap
import ipywidgets as widgets
from datetime import datetime
from IPython.display import display

# Import custom model loader utility
from model_loader_utils import ModelLoader, create_fallback_model

# ========================
# 1. INISIALISASI EARTH ENGINE
# ========================
# Pastikan Earth Engine sudah terautentikasi
# ee.Authenticate()  # Uncomment jika belum terautentikasi
ee.Initialize()

# ========================
# 2. KONFIGURASI DAN PARAMETER
# ========================
SCALE = 10
MAX_TRAINING_POINTS = 2000
BANDS_SELECTED = ['API', 'NDPI', 'RPI', 'RVI', 'VH_int', 'VV_int', 'angle']
PALET = ['#FF0000', '#00FF00', '#0000FF', '#8B4513', '#FFFF00']  # red, green, blue, brown, yellow
LABEL = ['Unknown', 'Vegetatif 1', 'Vegetatif 2', 'Generatif', 'Panen', 'Bera']
MODEL_PATH = '/content/rf_model.pkl'

# ========================
# 3. MUAT DATA
# ========================
print("🔄 Memuat data Earth Engine...")
titik_pelatihan = ee.FeatureCollection("projects/try-spasial/assets/output_shapefile")
koleksi_pelatihan = ee.ImageCollection("projects/try-spasial/assets/CollectionImage")

# Batasi titik pelatihan
titik_pelatihan_limit = titik_pelatihan.limit(MAX_TRAINING_POINTS)
print(f"✅ Data berhasil dimuat. Titik pelatihan dibatasi: {MAX_TRAINING_POINTS}")

# ========================
# 4. KONVERSI LABEL STRING KE ANGKA
# ========================
def konversi_label_numerik(feature):
    fase_string = ee.String(feature.get('Fase')).toLowerCase().trim()
    fase_numerik = ee.Algorithms.If(
        fase_string.equals('vegetatif 1'), 1,
        ee.Algorithms.If(
            fase_string.equals('vegetatif 2'), 2,
            ee.Algorithms.If(
                fase_string.equals('generatif'), 3,
                ee.Algorithms.If(
                    fase_string.equals('panen'), 4,
                    ee.Algorithms.If(
                        fase_string.equals('bera'), 5, 0
                    )
                )
            )
        )
    )
    return feature.set('FaseNumerik', fase_numerik)

titik_pelatihan_numerik = titik_pelatihan_limit.map(konversi_label_numerik)
print("✅ Label berhasil dikonversi ke format numerik")

# ========================
# 5. PERSIAPAN DATA TRAINING
# ========================
print("🔄 Mempersiapkan data training...")
koleksi_latih = koleksi_pelatihan.limit(20)

def ekstrak_fitur(citra):
    return citra.select(BANDS_SELECTED).sampleRegions(
        collection=titik_pelatihan_numerik,
        properties=['FaseNumerik'],
        scale=SCALE,
        geometries=False
    )

data_latih = koleksi_latih.map(ekstrak_fitur).flatten().filter(
    ee.Filter.notNull(BANDS_SELECTED + ['FaseNumerik'])
)
print("✅ Data training berhasil disiapkan")

# ========================
# 6. MUAT MODEL MENGGUNAKAN MODEL LOADER
# ========================
print("🔄 Memuat model menggunakan ModelLoader...")
model_loader = ModelLoader(MODEL_PATH, BANDS_SELECTED)

# Inisialisasi model
model = None
loaded_successfully = False

# Coba muat model yang sudah ada
if model_loader.load_model():
    # Validasi kompatibilitas
    if model_loader.validate_model_compatibility():
        # Buat model Earth Engine
        model = model_loader.create_ee_model(data_latih)
        if model is not None:
            loaded_successfully = True
            print("🎉 Model berhasil dimuat dan dikonversi untuk Earth Engine!")
        else:
            print("❌ Gagal membuat model Earth Engine dari model yang dimuat")
    else:
        print("⚠️  Model tidak kompatibel, akan menggunakan fallback")

# Jika gagal muat model, gunakan fallback
if not loaded_successfully:
    print("🔄 Menggunakan model fallback...")
    model = create_fallback_model(data_latih, BANDS_SELECTED)
    if model is not None:
        print("✅ Model fallback berhasil dibuat!")
    else:
        raise Exception("❌ Gagal membuat model fallback!")

# ========================
# 7. KLASIFIKASI SEMUA CITRA
# ========================
print("🔄 Memulai klasifikasi semua citra...")
koleksi_lengkap = ee.ImageCollection("projects/try-spasial/assets/CollectionImage").select(BANDS_SELECTED)

def klasifikasi_collection(collection):
    def klasifikasi_image(image):
        classified = image.classify(model).rename('classification')
        tanggal = ee.Date(image.get('system:time_start'))
        return classified.set({
            'system:time_start': image.get('system:time_start'),
            'tanggal': tanggal.format('YYYY-MM-dd')
        })
    return collection.map(klasifikasi_image)

hasil_klasifikasi = klasifikasi_collection(koleksi_lengkap)

# Check ukuran koleksi
collection_size = hasil_klasifikasi.size().getInfo()
print(f'✅ Klasifikasi selesai! Jumlah citra yang diklasifikasi: {collection_size}')

# ========================
# 8. BUAT PETA INTERAKTIF
# ========================
print("🔄 Menyiapkan peta interaktif...")

# Inisialisasi peta dengan geemap
Map = geemap.Map(center=[0, 0], zoom=2)

# Tambahkan layer titik pelatihan
Map.addLayer(titik_pelatihan, {'color': 'red'}, 'Titik Pelatihan')

# Tambahkan layer klasifikasi mean jika ada data
if collection_size > 0:
    klasifikasi_mean = hasil_klasifikasi.mean()
    vis_params = {
        'min': 1,
        'max': 5,
        'palette': PALET
    }
    Map.addLayer(klasifikasi_mean, vis_params, 'Klasifikasi Mean')

    # Center peta pada area titik pelatihan
    Map.centerObject(titik_pelatihan, 10)
    print("✅ Layer klasifikasi berhasil ditambahkan ke peta")
else:
    print('⚠️  Koleksi kosong, tidak ada klasifikasi.')

# ========================
# 9. WIDGET FILTER TANGGAL
# ========================
# Widget untuk input tanggal
tanggal_mulai = widgets.DatePicker(
    description='Tanggal Mulai:',
    value=datetime(2024, 1, 1),
    style={'description_width': 'initial'}
)

tanggal_akhir = widgets.DatePicker(
    description='Tanggal Akhir:',
    value=datetime(2024, 12, 31),
    style={'description_width': 'initial'}
)

tombol_tampilkan = widgets.Button(
    description='Tampilkan Klasifikasi',
    button_style='info',
    icon='eye'
)

tombol_clear = widgets.Button(
    description='Clear Layers',
    button_style='warning',
    icon='trash'
)

output_widget = widgets.Output()

def tampilkan_klasifikasi(b):
    with output_widget:
        output_widget.clear_output()

        start_date = tanggal_mulai.value.strftime('%Y-%m-%d')
        end_date = tanggal_akhir.value.strftime('%Y-%m-%d')

        try:
            # Filter berdasarkan tanggal
            filtered = hasil_klasifikasi.filterDate(start_date, end_date)
            filtered_size = filtered.size().getInfo()

            if filtered_size > 0:
                # Tambahkan layer ke peta
                layer_name = f'Klasifikasi {start_date} s.d. {end_date}'
                Map.addLayer(filtered.mean(), vis_params, layer_name)
                print(f'✅ Layer "{layer_name}" berhasil ditambahkan!')
                print(f'📊 Jumlah citra: {filtered_size}')
                
                # Tampilkan statistik sederhana
                mean_class = filtered.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=titik_pelatihan.geometry().bounds(),
                    scale=SCALE*10,  # Scale lebih besar untuk performa
                    maxPixels=1e6
                ).getInfo()
                
                if 'classification' in mean_class:
                    avg_class = mean_class['classification']
                    if avg_class:
                        print(f'📈 Rata-rata kelas: {avg_class:.2f}')
            else:
                print(f'❌ Tidak ada citra untuk periode {start_date} s.d. {end_date}')

        except Exception as e:
            print(f'❌ Error: {str(e)}')

def clear_layers(b):
    with output_widget:
        output_widget.clear_output()
        # Remove all layers except base and training points
        layer_names = list(Map.layers.keys())
        for name in layer_names:
            if 'Klasifikasi' in name and name != 'Klasifikasi Mean':
                Map.remove_layer(name)
        print('🧹 Layer klasifikasi temporal berhasil dihapus')

tombol_tampilkan.on_click(tampilkan_klasifikasi)
tombol_clear.on_click(clear_layers)

# ========================
# 10. FUNGSI UNTUK MENAMPILKAN LEGENDA
# ========================
def buat_legenda():
    """Membuat HTML untuk legenda"""
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 220px; height: 180px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 10px 0;"><b>🌾 Legenda Fase Padi</b></p>
    """

    for i in range(5):
        legend_html += f"""
        <p style="margin: 5px 0;"><span style="background-color:{PALET[i]}; 
                      width: 20px; height: 15px; display: inline-block; 
                      margin-right: 8px; border: 1px solid #333;"></span>
           <span style="font-weight: bold;">{i+1}</span> - {LABEL[i+1]}</p>
        """

    legend_html += """
    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
    🔄 Model dari: """ + MODEL_PATH + """</p>
    </div>"""
    return legend_html

# ========================
# 11. TAMPILKAN SEMUA KOMPONEN
# ========================
print("="*70)
print("🌾 KLASIFIKASI FASE PADI - MENGGUNAKAN MODEL PRE-TRAINED")
print("="*70)

# Tampilkan informasi model
if loaded_successfully and model_loader.loaded_model:
    model_info = model_loader.get_model_info()
    print(f"\n📋 INFORMASI MODEL:")
    print(f"   📁 Path: {MODEL_PATH}")
    print(f"   🏷️  Type: {model_info.get('model_type', 'Unknown')}")
    if 'n_estimators' in model_info:
        print(f"   🌳 Trees: {model_info['n_estimators']}")
    if 'n_features' in model_info:
        print(f"   📊 Features: {model_info['n_features']}")
else:
    print(f"\n📋 INFORMASI MODEL:")
    print(f"   📁 Menggunakan: Model Fallback (Random Forest)")
    print(f"   🌳 Trees: 200")
    print(f"   📊 Features: {len(BANDS_SELECTED)}")

# Tampilkan widget kontrol
print(f"\n📅 KONTROL FILTER TANGGAL:")
display(widgets.VBox([
    widgets.HBox([tanggal_mulai, tanggal_akhir]),
    widgets.HBox([tombol_tampilkan, tombol_clear]),
    output_widget
]))

# Tampilkan peta
print(f"\n🗺️  PETA INTERAKTIF:")
display(Map)

# Tambahkan legenda ke peta
legend_html = buat_legenda()
Map.add_html(legend_html, layer_name='Legenda')

print(f"\n✅ Setup selesai! Sistem klasifikasi siap digunakan.")
print(f"\n📖 CARA PENGGUNAAN:")
print(f"   1. 📅 Pilih tanggal mulai dan akhir")
print(f"   2. 👁️  Klik 'Tampilkan Klasifikasi'")
print(f"   3. 🗺️  Layer baru akan ditambahkan ke peta")
print(f"   4. 🧹 Gunakan 'Clear Layers' untuk membersihkan")

# ========================
# 12. FUNGSI ANALISIS LANJUTAN
# ========================
def analisis_komprehensif(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Fungsi untuk analisis komprehensif fase padi
    """
    try:
        print(f"\n🔍 ANALISIS KOMPREHENSIF PERIODE {start_date} s.d. {end_date}")
        print("="*60)
        
        filtered = hasil_klasifikasi.filterDate(start_date, end_date)
        count = filtered.size().getInfo()

        if count > 0:
            mean_image = filtered.mean()
            
            # Statistik per region
            stats = mean_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), '', True
                ).combine(
                    ee.Reducer.min(), '', True
                ).combine(
                    ee.Reducer.max(), '', True
                ),
                geometry=titik_pelatihan.geometry().bounds(),
                scale=SCALE,
                maxPixels=1e9
            ).getInfo()

            print(f"📊 STATISTIK KLASIFIKASI:")
            print(f"   🖼️  Jumlah citra: {count}")
            
            if 'classification_mean' in stats:
                print(f"   📈 Rata-rata kelas: {stats['classification_mean']:.2f}")
            if 'classification_stdDev' in stats:
                print(f"   📏 Std deviasi: {stats['classification_stdDev']:.2f}")
            if 'classification_min' in stats:
                print(f"   📉 Min kelas: {stats['classification_min']:.0f}")
            if 'classification_max' in stats:
                print(f"   📊 Max kelas: {stats['classification_max']:.0f}")

            # Histogram distribusi kelas
            histogram = mean_image.reduceRegion(
                reducer=ee.Reducer.histogram(maxBuckets=6, minBucketWidth=0.1),
                geometry=titik_pelatihan.geometry().bounds(),
                scale=SCALE*5,
                maxPixels=1e7
            ).getInfo()

            print(f"\n📊 DISTRIBUSI KELAS:")
            if 'classification' in histogram:
                hist_data = histogram['classification']
                if hist_data and 'histogram' in hist_data:
                    for i, count in enumerate(hist_data['histogram']):
                        if count > 0:
                            class_name = LABEL[i+1] if i+1 < len(LABEL) else f"Kelas {i+1}"
                            print(f"   🏷️  {class_name}: {count}")

            return {
                'period': f"{start_date} to {end_date}",
                'image_count': count,
                'statistics': stats,
                'histogram': histogram
            }
        else:
            print(f"❌ Tidak ada data untuk periode {start_date} s.d. {end_date}")
            return None

    except Exception as e:
        print(f"❌ Error dalam analisis: {str(e)}")
        return None

def export_hasil_klasifikasi(start_date='2024-01-01', end_date='2024-12-31',
                           scale=10, folder='GEE_Export_Padi'):
    """
    Fungsi untuk mengekspor hasil klasifikasi ke Google Drive
    """
    try:
        print(f"\n📤 EKSPOR HASIL KLASIFIKASI")
        print("="*40)
        
        filtered = hasil_klasifikasi.filterDate(start_date, end_date)
        count = filtered.size().getInfo()
        
        if count == 0:
            print(f"❌ Tidak ada data untuk diekspor pada periode {start_date} s.d. {end_date}")
            return None
            
        mean_classification = filtered.mean()

        # Konfigurasi ekspor
        filename = f'Klasifikasi_Padi_{start_date}_to_{end_date}_scale{scale}m'
        export_config = {
            'image': mean_classification,
            'description': filename,
            'folder': folder,
            'scale': scale,
            'region': titik_pelatihan.geometry().bounds(),
            'maxPixels': 1e9,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF'
        }

        # Mulai ekspor
        task = ee.batch.Export.image.toDrive(**export_config)
        task.start()

        print(f"✅ Ekspor berhasil dimulai!")
        print(f"   📁 Folder: {folder}")
        print(f"   📄 Filename: {filename}")
        print(f"   🔍 Scale: {scale}m")
        print(f"   🖼️  Jumlah citra: {count}")
        print(f"   📤 Status: {task.status()}")
        print(f"\n💡 Cek Google Drive Anda untuk file hasil ekspor.")

        return task

    except Exception as e:
        print(f"❌ Error dalam ekspor: {str(e)}")
        return None

# ========================
# 13. INFORMASI FUNGSI TERSEDIA
# ========================
print(f"\n🔧 FUNGSI LANJUTAN TERSEDIA:")
print(f"   📊 analisis_komprehensif(start_date, end_date)")
print(f"   📤 export_hasil_klasifikasi(start_date, end_date, scale, folder)")
print(f"\n💡 CONTOH PENGGUNAAN:")
print(f"   analisis_komprehensif('2024-06-01', '2024-08-31')")
print(f"   export_hasil_klasifikasi('2024-06-01', '2024-08-31', scale=30)")

print(f"\n🎉 Sistem klasifikasi fase padi siap digunakan!")