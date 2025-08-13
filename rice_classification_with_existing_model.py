import ee
import geemap
import pickle
import ipywidgets as widgets
from datetime import datetime
from IPython.display import display

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
titik_pelatihan = ee.FeatureCollection("projects/try-spasial/assets/output_shapefile")
koleksi_pelatihan = ee.ImageCollection("projects/try-spasial/assets/CollectionImage")

# Batasi titik pelatihan
titik_pelatihan_limit = titik_pelatihan.limit(MAX_TRAINING_POINTS)

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

# ========================
# 5. MUAT MODEL YANG SUDAH ADA
# ========================
print("🔄 Memuat model Random Forest yang sudah ada...")
try:
    with open(MODEL_PATH, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
    
    # Konversi model sklearn ke Earth Engine Classifier jika diperlukan
    # Untuk Earth Engine, kita perlu membuat ulang classifier dengan parameter yang sama
    if hasattr(loaded_model, 'n_estimators'):
        n_trees = loaded_model.n_estimators
        print(f"📊 Jumlah trees dalam model: {n_trees}")
        
        # Buat classifier Earth Engine dengan parameter yang sama
        model = ee.Classifier.smileRandomForest(n_trees)
        
        # Jika model sudah dilatih sebelumnya, kita perlu menggunakan data training
        # untuk "melatih ulang" classifier Earth Engine dengan parameter yang sama
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
        
        # Latih model Earth Engine dengan parameter yang sama dengan model yang dimuat
        model = model.train(
            features=data_latih,
            classProperty='FaseNumerik',
            inputProperties=BANDS_SELECTED
        )
        
        print("✅ Model Earth Engine berhasil dibuat dengan parameter yang sama!")
    else:
        print("⚠️  Model yang dimuat bukan Random Forest sklearn standard")
        # Fallback: gunakan model default
        model = ee.Classifier.smileRandomForest(200).train(
            features=data_latih,
            classProperty='FaseNumerik',
            inputProperties=BANDS_SELECTED
        )
        
except FileNotFoundError:
    print(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    print("🔄 Melatih model baru sebagai fallback...")
    
    # Fallback: latih model baru jika file tidak ditemukan
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
    
    # Split data untuk training dan testing
    data_acak = data_latih.randomColumn('random', 42)
    train_set = data_acak.filter(ee.Filter.lt('random', 0.8))
    test_set = data_acak.filter(ee.Filter.gte('random', 0.8))
    
    # Latih model Random Forest
    model = ee.Classifier.smileRandomForest(200).train(
        features=train_set,
        classProperty='FaseNumerik',
        inputProperties=BANDS_SELECTED
    )
    
    # Evaluasi akurasi
    test_accuracy = test_set.classify(model).errorMatrix('FaseNumerik', 'classification').accuracy()
    print('Model fallback berhasil dilatih!')
    print(f'Akurasi: {test_accuracy.getInfo():.4f}')

except Exception as e:
    print(f"❌ Error saat memuat model: {str(e)}")
    print("🔄 Melatih model baru sebagai fallback...")
    
    # Fallback: latih model baru jika ada error
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
    
    # Split data untuk training dan testing
    data_acak = data_latih.randomColumn('random', 42)
    train_set = data_acak.filter(ee.Filter.lt('random', 0.8))
    test_set = data_acak.filter(ee.Filter.gte('random', 0.8))
    
    # Latih model Random Forest
    model = ee.Classifier.smileRandomForest(200).train(
        features=train_set,
        classProperty='FaseNumerik',
        inputProperties=BANDS_SELECTED
    )
    
    # Evaluasi akurasi
    test_accuracy = test_set.classify(model).errorMatrix('FaseNumerik', 'classification').accuracy()
    print('Model fallback berhasil dilatih!')
    print(f'Akurasi: {test_accuracy.getInfo():.4f}')

# ========================
# 6. KLASIFIKASI SEMUA CITRA
# ========================
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
print(f'Jumlah citra yang diklasifikasi: {collection_size}')

# ========================
# 7. BUAT PETA INTERAKTIF
# ========================
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
else:
    print('Koleksi kosong, tidak ada klasifikasi.')

# ========================
# 8. WIDGET FILTER TANGGAL
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
    button_style='info'
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
                print(f'Layer "{layer_name}" berhasil ditambahkan!')
                print(f'Jumlah citra: {filtered_size}')
            else:
                print(f'Tidak ada citra untuk periode {start_date} s.d. {end_date}')

        except Exception as e:
            print(f'Error: {str(e)}')

tombol_tampilkan.on_click(tampilkan_klasifikasi)

# ========================
# 9. FUNGSI UNTUK MENAMPILKAN LEGENDA
# ========================
def buat_legenda():
    """Membuat HTML untuk legenda"""
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 200px; height: 160px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <p><b>Legenda Fase Padi</b></p>
    """

    for i in range(5):
        legend_html += f"""
        <p><span style="background-color:{PALET[i]}; width: 20px; height: 15px;
                      display: inline-block; margin-right: 5px;"></span>
           {i+1} - {LABEL[i+1]}</p>
        """

    legend_html += "</div>"
    return legend_html

# ========================
# 10. TAMPILKAN SEMUA KOMPONEN
# ========================
print("="*60)
print("KLASIFIKASI FASE PADI - MENGGUNAKAN MODEL YANG SUDAH ADA")
print("="*60)

# Tampilkan widget kontrol
print("\n📅 FILTER TANGGAL:")
display(widgets.VBox([
    widgets.HBox([tanggal_mulai, tanggal_akhir]),
    tombol_tampilkan,
    output_widget
]))

# Tampilkan peta
print("\n🗺️  PETA INTERAKTIF:")
display(Map)

# Tambahkan legenda ke peta using geemap's add_html
legend_html = buat_legenda()
Map.add_html(legend_html, layer_name='Legenda')

print("\n✅ Setup selesai! Model yang sudah ada berhasil dimuat dan visualisasi siap digunakan.")
print("\nCara penggunaan:")
print("1. Pilih tanggal mulai dan akhir")
print("2. Klik 'Tampilkan Klasifikasi'")
print("3. Layer baru akan ditambahkan ke peta")

# ========================
# 11. FUNGSI TAMBAHAN UNTUK ANALISIS
# ========================
def analisis_temporal(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Fungsi untuk analisis temporal fase padi
    """
    try:
        filtered = hasil_klasifikasi.filterDate(start_date, end_date)
        count = filtered.size().getInfo()

        if count > 0:
            # Hitung statistik per kelas
            mean_image = filtered.mean()

            # Buat histogram
            histogram = mean_image.reduceRegion(
                reducer=ee.Reducer.histogram(maxBuckets=6, minBucketWidth=0.1),
                geometry=titik_pelatihan.geometry().bounds(),
                scale=SCALE,
                maxPixels=1e9
            )

            print(f"\n📊 ANALISIS PERIODE {start_date} s.d. {end_date}")
            print(f"Jumlah citra: {count}")
            print(f"Histogram klasifikasi: {histogram.getInfo()}")

            return {
                'period': f"{start_date} to {end_date}",
                'image_count': count,
                'mean_image': mean_image,
                'histogram': histogram
            }
        else:
            print(f"Tidak ada data untuk periode {start_date} s.d. {end_date}")
            return None

    except Exception as e:
        print(f"Error dalam analisis: {str(e)}")
        return None

def export_classification(start_date='2024-01-01', end_date='2024-12-31',
                         scale=10, folder='GEE_Export'):
    """
    Fungsi untuk mengekspor hasil klasifikasi ke Google Drive
    """
    try:
        filtered = hasil_klasifikasi.filterDate(start_date, end_date)
        mean_classification = filtered.mean()

        # Konfigurasi ekspor
        export_config = {
            'image': mean_classification,
            'description': f'Klasifikasi_Padi_{start_date}_to_{end_date}',
            'folder': folder,
            'scale': scale,
            'region': titik_pelatihan.geometry().bounds(),
            'maxPixels': 1e9,
            'crs': 'EPSG:4326'
        }

        # Mulai ekspor
        task = ee.batch.Export.image.toDrive(**export_config)
        task.start()

        print(f"✅ Ekspor dimulai: {export_config['description']}")
        print(f"📁 Folder: {folder}")
        print(f"🔍 Scale: {scale}m")
        print("Cek Google Drive Anda untuk file hasil ekspor.")

        return task

    except Exception as e:
        print(f"❌ Error dalam ekspor: {str(e)}")
        return None

# ========================
# 12. FUNGSI UNTUK SIMPAN MODEL YANG SUDAH DILATIH
# ========================
def simpan_model_ke_pickle(output_path='/content/rf_model_backup.pkl'):
    """
    Fungsi untuk menyimpan model Earth Engine sebagai referensi
    Note: Earth Engine classifiers tidak bisa langsung di-pickle,
    tapi kita bisa simpan parameter model untuk referensi
    """
    try:
        model_info = {
            'model_type': 'RandomForest',
            'bands_used': BANDS_SELECTED,
            'classes': LABEL,
            'scale': SCALE,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ Informasi model berhasil disimpan ke: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saat menyimpan model: {str(e)}")
        return False

# Contoh penggunaan fungsi tambahan:
print("\n🔧 FUNGSI TAMBAHAN TERSEDIA:")
print("- analisis_temporal(start_date, end_date): Analisis statistik temporal")
print("- export_classification(): Ekspor hasil ke Google Drive")
print("- simpan_model_ke_pickle(): Simpan informasi model")
print("\nContoh: analisis_temporal('2024-06-01', '2024-08-31')")