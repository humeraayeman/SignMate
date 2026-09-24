import os
import zipfile

def create_package():
    src = r"C:\Users\humer\Documents\SignSync"
    dst = r"C:\Users\humer\Documents\SignMate_VSCode.zip"
    exclude = {"venv", ".venv", "__pycache__", ".git", "Indian", "RealSign", "dataset_splits"}

    if os.path.exists(dst):
        try:
            os.remove(dst)
        except Exception:
            pass

    count = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in exclude]
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, src)
                zf.write(full_path, rel_path)
                count += 1

    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"Successfully packaged {count} files -> {dst} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    create_package()
