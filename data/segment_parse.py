import argparse
import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from mat73 import loadmat
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Save PulseDB Segmented Data as PICKLE files")
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="./pulsedb/raw_data/",
        help="Path to the PulseDB waveform data",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./pulsedb/waveform/",
        help="Path to save the processed waveform data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for processing"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds per file"
    )

    return parser.parse_args()


def process_and_save(args, file):
    try:
        full_path = os.path.join(args.raw_data_path, file)
        if os.path.getsize(full_path) < 1024:  # Skip tiny or corrupt files
            return f"[SKIP small file] {file}"

        signals = loadmat(full_path)
        subj = signals["Subj_Wins"]

        # Handle both segmented and single
        try:
            n_segments = len(subj["SegDBP"])
            keys = list(subj.keys())
            seg_list = [{key: subj[key][i][0] for key in keys} for i in range(n_segments)]
        except Exception:
            seg_list = [subj]

        for seg in seg_list:
            caseID = seg["CaseID"].replace("c", "").replace("-", "")
            segmentID = int(seg["SegmentID"])
            case_dir = os.path.join(args.save_dir, caseID)
            os.makedirs(case_dir, exist_ok=True)
            save_path = os.path.join(case_dir, f"{caseID}_{segmentID}.pkl")
            with open(save_path, "wb") as f:
                pkl.dump(seg, f)

        return None
    except Exception as e:
        return f"[ERROR] {file}: {e}"


if __name__ == "__main__":
    args = parse_args()
    files = [f for f in os.listdir(args.raw_data_path) if f.endswith('.mat')]
    print(f"Processing {len(files)} .mat files...")

    batch_size = args.batch_size  # restart pool every `batch_size` files
    for start in range(0, len(files), batch_size):
        batch = files[start:start + batch_size]
        print(f"\n\tProcessing batch {start // batch_size + 1} / {len(files) // batch_size + 1}")

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_and_save, args, f): f for f in batch}
            for fut in tqdm(as_completed(futures), total=len(batch)):
                try:
                    res = fut.result(timeout=args.timeout)  # `timeout` seconds timeout per file
                    if res:  # log skip/errors
                        print(res)
                except TimeoutError:
                    print(f"[TIMEOUT] {futures[fut]}")
                except Exception as e:
                    print(f"[FAILED] {futures[fut]}: {e}")

    print("Done.")
