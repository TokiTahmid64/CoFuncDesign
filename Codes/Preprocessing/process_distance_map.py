import os
import numpy as np


def parse_c_atoms(pdb_path):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[13:15].strip() == "C":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)


def compute_distance_map(coords):
    if len(coords) == 0:
        return np.array([])

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


def process_all_pdbs(folder_path, save_folder=None):
    distance_maps = {}
    pdb_files = [f for f in os.listdir(folder_path)]

    for pdb_file in pdb_files:
        pdb_path = os.path.join(folder_path, pdb_file)
        coords = parse_c_atoms(pdb_path)
        dist_map = compute_distance_map(coords)

        distance_maps[pdb_file] = dist_map

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, pdb_file), dist_map)

    return distance_maps


if __name__ == "__main__":
    folder = "dompdb"
    output = "distance_maps"
    all_maps = process_all_pdbs(folder, save_folder=output)
    print(f"Processed {len(all_maps)} PDB files.")
