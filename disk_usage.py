import os
from pathlib import Path

def get_dir_size(path='.'):
    """Calculate the total size of a directory and its contents."""
    total_size = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total_size += get_dir_size(entry.path)
    except PermissionError:
        # Handle folders we don't have access to
        return 0
    return total_size

def format_size(size_bytes):
    """Convert bytes to a human-readable string (MB, GB, etc)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def list_largest_folders(root_dir, limit=10):
    """Find and print the largest top-level folders in the root_dir."""
    folder_sizes = []
    root_path = Path(root_dir)

    print(f"Analyzing: {root_path.absolute()}\n")

    # Iterate through only the immediate subdirectories of the root
    for item in root_path.iterdir():
        if item.is_dir():
            size = get_dir_size(item)
            folder_sizes.append((item.name, size))

    # Sort folders by size in descending order
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Folder Name':<30} | {'Size'}")
    print("-" * 45)
    for name, size in folder_sizes[:limit]:
        print(f"{name:<30} | {format_size(size)}")

if __name__ == "__main__":
    # Change '.' to your specific project path if needed
    list_largest_folders('./experiments/rzz_ising_experiment')