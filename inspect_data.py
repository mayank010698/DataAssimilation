
import h5py
import numpy as np

def inspect_data():
    try:
        with h5py.File("data.h5", "r") as f:
            print("Keys in data.h5:", list(f.keys()))
            if "train" in f:
                print("Keys in train:", list(f["train"].keys()))
                traj = f["train/trajectories"]
                print("Trajectories shape:", traj.shape)
                print("Trajectories sample (first step of first traj):", traj[0, 0])
                print("Trajectories sample (second step of first traj):", traj[0, 1])
                print("Std of trajectories:", np.std(traj[:]))
                
    except FileNotFoundError:
        print("data.h5 not found.")

    try:
        with h5py.File("data_scaled.h5", "r") as f:
            print("\nKeys in data_scaled.h5:", list(f.keys()))
            traj = f["train/trajectories"]
            print("Scaled Trajectories std:", np.std(traj[:]))
    except FileNotFoundError:
        print("data_scaled.h5 not found.")

if __name__ == "__main__":
    inspect_data()

