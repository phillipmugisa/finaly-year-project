"""
scripts/download_rdd2022.py
---------------------------
Download the RDD2022 dataset from FigShare (DOI: 10.6084/m9.figshare.21431547).

NOTE: RDD2022 is only available as a single monolithic zip (~13.3 GB) containing
all 6 country subsets. There is no per-country download option on FigShare.

For pre-training the PCI head you only need the Japan subset. After downloading
the full zip, this script extracts only the requested countries and discards the
rest (unless --keep-zip is passed).

Recommended workflow on machines with limited bandwidth:
  - Use Google Colab (pre-training GPU + fast download):
      !pip install gdown
      !python scripts/download_rdd2022.py --output data/rdd2022

Usage
-----
# Download full zip and extract Japan only (default, ~13 GB download)
python scripts/download_rdd2022.py --output data/rdd2022

# Extract multiple countries from the already-downloaded zip
python scripts/download_rdd2022.py --output data/rdd2022 --countries Japan India

# Keep the full zip after extraction (for re-extracting later)
python scripts/download_rdd2022.py --output data/rdd2022 --keep-zip

Citation
--------
Arya, D., et al. (2022). RDD2022: A multi-national image dataset for
automatic Road Damage Detection. arXiv:2209.08538.
FigShare: https://doi.org/10.6084/m9.figshare.21431547
"""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# FigShare direct download — single zip with all 6 countries (~13.3 GB)
_FIGSHARE_URL  = "https://ndownloader.figshare.com/files/38030910"
_FULL_ZIP_NAME = "RDD2022_released_through_CRDDC2022.zip"
_FULL_ZIP_SIZE = "13.3 GB"

# Country folder names inside the zip
_COUNTRY_DIRS = {
    "Japan":           "Japan",
    "India":           "India",
    "Norway":          "Norway",
    "United_States":   "United_States",
    "Czech":           "Czech",
    "China_MotorBike": "China_MotorBike",
}


def download_file(url: str, dest: Path, label: str) -> bool:
    """Stream-download url to dest, showing progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  {label}: already downloaded at {dest} — skipping")
        return True

    print(f"  Downloading {label} from FigShare …")
    print(f"  URL: {url}")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)   # 64 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        gb  = done / 1e9
                        print(f"\r    {pct:5.1f}%  {gb:.2f} GB", end="", flush=True)
        print()
        return True
    except URLError as e:
        print(f"\n  ERROR downloading: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_countries(zip_path: Path, out_dir: Path, countries: list[str]) -> list[str]:
    """Extract only the requested country directories from the full zip."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        for country in countries:
            country_dir = out_dir / country
            if country_dir.exists() and any(country_dir.iterdir()):
                print(f"  {country}: already extracted — skipping")
                extracted.append(country)
                continue

            # Collect members belonging to this country
            prefix = f"{country}/"
            members = [n for n in all_names if n.startswith(prefix)]
            if not members:
                # Some zips nest under a top-level directory
                members = [n for n in all_names if f"/{country}/" in n]

            if not members:
                print(f"  {country}: not found in zip — skipping")
                continue

            print(f"  {country}: extracting {len(members):,} files …")
            for member in members:
                zf.extract(member, out_dir)
            print(f"  {country}: done → {country_dir}")
            extracted.append(country)

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Download RDD2022 dataset from FigShare")
    parser.add_argument("--output",    default="data/rdd2022",
                        help="Destination directory (default: data/rdd2022)")
    parser.add_argument("--countries", nargs="+",
                        choices=list(_COUNTRY_DIRS.keys()),
                        default=["Japan"],
                        help="Countries to extract after download (default: Japan)")
    parser.add_argument("--all",       action="store_true",
                        help="Extract all 6 countries")
    parser.add_argument("--keep-zip",  action="store_true",
                        help="Keep the full zip after extraction (13.3 GB)")
    args = parser.parse_args()

    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = list(_COUNTRY_DIRS.keys()) if args.all else args.countries

    print(f"RDD2022 download (FigShare) — will extract: {', '.join(countries)}")
    print(f"WARNING: The full zip is {_FULL_ZIP_SIZE}. Downloading the entire file.")
    print(f"Output: {out_dir.resolve()}\n")

    zip_dest = out_dir / _FULL_ZIP_NAME
    if not download_file(_FIGSHARE_URL, zip_dest, _FULL_ZIP_SIZE):
        sys.exit("Download failed.")

    print("\nExtracting requested countries …")
    ok = extract_countries(zip_dest, out_dir, countries)

    if not args.keep_zip and zip_dest.exists():
        print(f"\nRemoving full zip to save space ({_FULL_ZIP_SIZE}) …")
        zip_dest.unlink()

    print(f"\nDone. {len(ok)}/{len(countries)} countries ready under {out_dir}/")
    if ok:
        print("Run pre-training with:")
        print(f"  python scripts/pretrain_pci.py --data {out_dir} "
              f"--countries {' '.join(ok)}")


if __name__ == "__main__":
    main()
