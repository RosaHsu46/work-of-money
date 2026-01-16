
import sys
import importlib.util

def check_import(package_name):
    print(f"Checking {package_name}...", end=" ")
    if importlib.util.find_spec(package_name):
        print("INSTALLED")
        return True
    else:
        print("MISSING")
        return False

def verify_lightgbm_gpu():
    print("\nVerifying LightGBM GPU support...")
    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.datasets import make_classification
        
        print(f"LightGBM Version: {lgb.__version__}")
        
        X, y = make_classification(n_samples=1000, n_features=20)
        
        # Test GPU params
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': -1
        }
        
        print("Attempting to train with device='gpu'...")
        train_data = lgb.Dataset(X, label=y)
        bst = lgb.train(params, train_data, num_boost_round=10)
        print("SUCCESS: LightGBM training with GPU completed without errors.")
        return True
        
    except Exception as e:
        print(f"FAILURE: LightGBM GPU check failed.\nError: {e}")
        # Add hint about OpenCL if that seems to be the issue
        if "OpenCL" in str(e):
             print("\nHint: Ensure you have latest GPU drivers installed which include OpenCL runtime.")
        return False

def main():
    print(f"Python: {sys.version}\n")
    
    packages = ["yfinance", "pandas_ta", "sklearn", "lightgbm"]
    all_installed = all([check_import(pkg) for pkg in packages])
    
    if all_installed:
        print("\nAll packages installed successfully.")
        verify_lightgbm_gpu()
    else:
        print("\nSome packages are missing.")

if __name__ == "__main__":
    main()
