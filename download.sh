## Download the pretraiend models and structure files from the github release
github_release="https://github.com/jlaw9/upper-bound-energy-gnn/releases/download/0.1"

f="fully_relaxed.json.gz"
echo "wget -O inputs/$f $github_release/$f"
wget -O inputs/$f $github_release/$f

f="volume_relaxed.json.gz"
echo "wget -O inputs/$f $github_release/$f"
wget -O inputs/$f $github_release/$f

f="icsd_and_full_vol_battery.hdf5"
echo "wget -O pretrained_models/$f $github_release/$f"
wget -O pretrained_models/$f $github_release/$f

f="20220607_icsd_and_full_vol_battery.hdf5"
echo "wget -O pretrained_models/$f $github_release/$f"
wget -O pretrained_models/$f $github_release/$f

f="icsd_and_unrel_battery.hdf5"
echo "wget -O pretrained_models/$f $github_release/$f"
wget -O pretrained_models/$f $github_release/$f
