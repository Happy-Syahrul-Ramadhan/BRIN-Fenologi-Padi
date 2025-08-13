import pickle
import joblib
import ee
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

class ModelLoader:
    """
    Utility class untuk memuat berbagai jenis model yang sudah dilatih
    dan mengkonversinya untuk digunakan dengan Google Earth Engine
    """
    
    def __init__(self, model_path, bands_selected):
        self.model_path = model_path
        self.bands_selected = bands_selected
        self.loaded_model = None
        self.ee_model = None
        
    def load_model(self):
        """
        Memuat model dari berbagai format file
        """
        try:
            # Coba muat dengan pickle
            with open(self.model_path, 'rb') as f:
                self.loaded_model = pickle.load(f)
            print(f"✅ Model berhasil dimuat dengan pickle dari: {self.model_path}")
            return True
            
        except Exception as e1:
            try:
                # Coba muat dengan joblib
                self.loaded_model = joblib.load(self.model_path)
                print(f"✅ Model berhasil dimuat dengan joblib dari: {self.model_path}")
                return True
                
            except Exception as e2:
                print(f"❌ Gagal memuat model dengan pickle: {str(e1)}")
                print(f"❌ Gagal memuat model dengan joblib: {str(e2)}")
                return False
    
    def get_model_info(self):
        """
        Mengambil informasi dari model yang dimuat
        """
        if self.loaded_model is None:
            return None
            
        info = {
            'model_type': type(self.loaded_model).__name__,
            'model_class': str(type(self.loaded_model))
        }
        
        # Informasi spesifik untuk Random Forest
        if hasattr(self.loaded_model, 'n_estimators'):
            info['n_estimators'] = self.loaded_model.n_estimators
            
        if hasattr(self.loaded_model, 'max_depth'):
            info['max_depth'] = self.loaded_model.max_depth
            
        if hasattr(self.loaded_model, 'n_features_in_'):
            info['n_features'] = self.loaded_model.n_features_in_
            
        if hasattr(self.loaded_model, 'feature_names_in_'):
            info['feature_names'] = self.loaded_model.feature_names_in_.tolist()
            
        if hasattr(self.loaded_model, 'classes_'):
            info['classes'] = self.loaded_model.classes_.tolist()
            
        return info
    
    def create_ee_model(self, training_data):
        """
        Membuat model Earth Engine berdasarkan model yang dimuat
        """
        if self.loaded_model is None:
            print("❌ Model belum dimuat. Jalankan load_model() terlebih dahulu.")
            return None
            
        model_info = self.get_model_info()
        print(f"📊 Informasi model: {model_info}")
        
        # Untuk Random Forest
        if isinstance(self.loaded_model, RandomForestClassifier):
            n_trees = getattr(self.loaded_model, 'n_estimators', 100)
            max_depth = getattr(self.loaded_model, 'max_depth', None)
            
            print(f"🌳 Membuat Random Forest dengan {n_trees} trees")
            if max_depth:
                print(f"📏 Max depth: {max_depth}")
            
            # Buat classifier Earth Engine
            if max_depth:
                self.ee_model = ee.Classifier.smileRandomForest(
                    numberOfTrees=n_trees,
                    maxDepth=max_depth
                )
            else:
                self.ee_model = ee.Classifier.smileRandomForest(
                    numberOfTrees=n_trees
                )
                
        else:
            # Fallback untuk model lain
            print(f"⚠️  Model type {model_info['model_type']} tidak didukung secara khusus.")
            print("🔄 Menggunakan Random Forest default...")
            self.ee_model = ee.Classifier.smileRandomForest(100)
        
        # Latih model Earth Engine
        try:
            self.ee_model = self.ee_model.train(
                features=training_data,
                classProperty='FaseNumerik',
                inputProperties=self.bands_selected
            )
            print("✅ Model Earth Engine berhasil dilatih!")
            return self.ee_model
            
        except Exception as e:
            print(f"❌ Error saat melatih model Earth Engine: {str(e)}")
            return None
    
    def validate_model_compatibility(self):
        """
        Validasi kompatibilitas model dengan bands yang dipilih
        """
        if self.loaded_model is None:
            return False
            
        model_info = self.get_model_info()
        
        # Cek jumlah fitur
        if 'n_features' in model_info:
            expected_features = model_info['n_features']
            actual_features = len(self.bands_selected)
            
            if expected_features != actual_features:
                print(f"⚠️  Warning: Model dilatih dengan {expected_features} fitur, "
                      f"tapi bands yang dipilih: {actual_features}")
                return False
        
        # Cek nama fitur jika tersedia
        if 'feature_names' in model_info:
            model_features = set(model_info['feature_names'])
            selected_features = set(self.bands_selected)
            
            if model_features != selected_features:
                print(f"⚠️  Warning: Fitur model tidak cocok dengan bands yang dipilih")
                print(f"Model features: {model_features}")
                print(f"Selected bands: {selected_features}")
                return False
        
        print("✅ Model kompatibel dengan bands yang dipilih")
        return True

def load_and_create_ee_model(model_path, bands_selected, training_data):
    """
    Fungsi helper untuk memuat model dan membuat model Earth Engine
    """
    loader = ModelLoader(model_path, bands_selected)
    
    # Muat model
    if not loader.load_model():
        return None, None
    
    # Validasi kompatibilitas
    if not loader.validate_model_compatibility():
        print("⚠️  Melanjutkan meskipun ada warning kompatibilitas...")
    
    # Buat model Earth Engine
    ee_model = loader.create_ee_model(training_data)
    
    return loader.loaded_model, ee_model

def create_fallback_model(training_data, bands_selected, n_trees=200):
    """
    Membuat model fallback jika model yang dimuat gagal
    """
    print(f"🔄 Membuat model fallback dengan {n_trees} trees...")
    
    # Split data untuk training dan testing
    data_acak = training_data.randomColumn('random', 42)
    train_set = data_acak.filter(ee.Filter.lt('random', 0.8))
    test_set = data_acak.filter(ee.Filter.gte('random', 0.8))
    
    # Latih model Random Forest
    model = ee.Classifier.smileRandomForest(n_trees).train(
        features=train_set,
        classProperty='FaseNumerik',
        inputProperties=bands_selected
    )
    
    # Evaluasi akurasi
    try:
        test_accuracy = test_set.classify(model).errorMatrix('FaseNumerik', 'classification').accuracy()
        print(f'✅ Model fallback berhasil dilatih dengan akurasi: {test_accuracy.getInfo():.4f}')
    except:
        print('✅ Model fallback berhasil dilatih (akurasi tidak dapat dihitung)')
    
    return model

# Contoh penggunaan:
if __name__ == "__main__":
    # Contoh parameter
    MODEL_PATH = '/content/rf_model.pkl'
    BANDS_SELECTED = ['API', 'NDPI', 'RPI', 'RVI', 'VH_int', 'VV_int', 'angle']
    
    print("🔧 Model Loader Utility - Contoh Penggunaan")
    print("="*50)
    
    # Buat instance loader
    loader = ModelLoader(MODEL_PATH, BANDS_SELECTED)
    
    # Muat model
    if loader.load_model():
        # Tampilkan informasi model
        info = loader.get_model_info()
        print(f"📋 Informasi model: {info}")
        
        # Validasi kompatibilitas
        loader.validate_model_compatibility()
    else:
        print("❌ Gagal memuat model")