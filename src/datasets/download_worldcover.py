import os
import requests
import argparse

def download_worldcover_tile(tile_id, output_dir="data/worldcover", year=2021, version="v200"):
    """
    Downloads an ESA WorldCover tile from the public AWS S3 bucket.
    Example tile_id: 'N00E006'
    """
    filename = f"ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif"
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/{filename}"
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping download.")
        return out_path
        
    print(f"Downloading {url} to {out_path}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download tile {tile_id}. HTTP Status: {response.status_code}")
        
    # Download with progress chunks
    with open(out_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB...", end='\r')
    print(f"\nSuccessfully downloaded {filename}.")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ESA WorldCover tile")
    parser.add_argument("--tile", type=str, default="N00E006", help="Tile ID (e.g. N00E006, N20E072)")
    parser.add_argument("--out", type=str, default="data/worldcover", help="Output directory")
    args = parser.parse_args()
    
    download_worldcover_tile(args.tile, args.out)
