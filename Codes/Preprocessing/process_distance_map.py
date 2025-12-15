
# Written by: Lana Glisic

import os
import numpy as np

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def parse_c_atoms_and_sequence(pdb_path):
    coords = []
    seq = []
    seen_residues = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "C":
                continue

            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq  = line[22:26].strip()
            res_id = (chain_id, res_seq)

            if res_id in seen_residues:
                continue
            seen_residues.add(res_id)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

            seq.append(three_to_one.get(res_name, "X"))

    return "".join(seq), np.array(coords)


def compute_distance_map(coords):
    if len(coords) == 0:
        return np.array([])

    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def process_all_pdbs(folder_path, save_folder, length_file):
    os.makedirs(save_folder, exist_ok=True)

    pdb_files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    )

    sequence_lengths = []

    for pdb_file in pdb_files:
        pdb_path = os.path.join(folder_path, pdb_file)

        seq, coords = parse_c_atoms_and_sequence(pdb_path)
        dist_map = compute_distance_map(coords)

        protein_id = os.path.splitext(pdb_file)[0]
        L = len(seq)

        print(f"{protein_id}: sequence length = {L}")

        if len(seq) > 510:
            print("Sequence was too long. Ignoring.")
            continue

        if coords.shape[0] != len(seq):
            print(f"Residue mismatch in {protein_id}, skipping")
            continue

        np.savez(
            os.path.join(save_folder, protein_id + ".npz"),
            sequence=seq,
            distance_map=dist_map,
        )


        sequence_lengths.append((protein_id, L))

    with open(length_file, "w") as f:
        f.write("protein_id,sequence_length\n")
        for pid, L in sequence_lengths:
            f.write(f"{pid},{L}\n")

    print(f"\nSaved sequence lengths to: {length_file}")
    print(f"Processed {len(sequence_lengths)} PDB files.")


if __name__ == "__main__":
    pdb_folder = "dompdb"
    dist_out   = "distance_maps"
    length_out = "sequence_lengths.csv"

    process_all_pdbs(
        folder_path=pdb_folder,
        save_folder=dist_out,
        length_file=length_out
    )
