import sys
print("Python path:", sys.path)
print("Attempting import...")
try:
    import src.main
    print("Import successful")
except Exception as e:
    print("Import failed:", str(e))
    import traceback
    traceback.print_exc()
