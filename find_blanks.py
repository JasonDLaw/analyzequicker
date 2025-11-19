#%%
import os
import glob
from collections import Counter

deleted_count = 0
deleted_files = []

data_dir = 'data'
# Use glob to find all CSV files recursively
pattern = os.path.join(data_dir, '**', '*.csv')
csv_files = glob.glob(pattern, recursive=True)

#%%

# Get file sizes and sort
files_with_size = [(f, os.path.getsize(f)) for f in csv_files]
files_with_size.sort(key=lambda x: x[1])
files_with_size
#%%

sizes = [size for _, size in files_with_size]
size_counts = Counter(sizes)
print("\nFile size distribution (showing smallest 10 sizes):")
for size, count in sorted(size_counts.items())[:10]:
    print(f"  {size} bytes: {count} files")

#%%
# Only check files that are 2 bytes or smaller
files_to_check = [(f, s) for f, s in files_with_size if s <= 50]
print(f"\nChecking {len(files_to_check)} files that are 2 bytes or smaller...")

for file_path, file_size in files_to_check:
    

        os.remove(file_path)
        deleted_count += 1
        if deleted_count % 100 == 0:
            print(f"Deleted {deleted_count} files so far...")
        



# %%
