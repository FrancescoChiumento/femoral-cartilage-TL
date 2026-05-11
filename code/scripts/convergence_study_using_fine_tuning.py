# Francesco Chiumento, 2025
# Convergence study for fine-tuning

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import SimpleITK as sitk
import sys

SOURCE_IMAGES = "YOUR_PATH_HERE/patients/images"
SOURCE_MASKS = "YOUR_PATH_HERE/patients/masks"
PYKNEER_PATH = "YOUR_PATH_HERE/pykneer"

sys.path.insert(0, PYKNEER_PATH)
try:
    import sitk_functions as sitkf
    print("pyKNEEr loaded successfully!")
except ImportError as e:
    print(f"ERROR importing pyKNEEr: {e}")
    sitkf = None

MIN_PATIENTS = 3
MAX_PATIENTS = 44  #Change based on total number of patients
NUM_EPOCHS_PER_ITERATION = 50
RANDOM_SEED = 42

RESULTS_DIR = "convergence_study_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_available_patients():
    patients = sorted([f for f in os.listdir(SOURCE_IMAGES) if f.endswith('.mha')])
    print(f"Found {len(patients)} total patients")
    return patients

def run_finetuning_iteration(train_patients, val_patients, iteration_num):
    print(f"\n  Fine-tuning with {len(train_patients)} training patients")
    
    checkpoint_dir = os.path.join(RESULTS_DIR, f"iteration_{iteration_num}_patients")
    
    import fine_tuning
    
    best_checkpoint = fine_tuning.main(
        train_patients=train_patients,
        val_patients=val_patients,
        num_epochs=NUM_EPOCHS_PER_ITERATION,
        checkpoint_dir=checkpoint_dir,
        source_images_path=SOURCE_IMAGES, 
        source_masks_path=SOURCE_MASKS
    )

    if not os.path.exists(best_checkpoint):
        print(f"  ERROR: Checkpoint not found!")
        return None
        
    print(f"  Fine-tuning completed!")
    return best_checkpoint

def prepare_test_data(test_patients, iteration_num):
    test_dir = os.path.join(RESULTS_DIR, f"test_iteration_{iteration_num}")
    test_images_dir = os.path.join(test_dir, "images")
    test_masks_dir = os.path.join(test_dir, "masks")
    
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)
    
    for patient in test_patients:
        src_img = os.path.join(SOURCE_IMAGES, patient)
        dst_img = os.path.join(test_images_dir, patient)
        shutil.copy2(src_img, dst_img)
        
        mask_name = patient.replace('.mha', '.nii.gz')
        src_mask = os.path.join(SOURCE_MASKS, mask_name)
        dst_mask = os.path.join(test_masks_dir, mask_name)
        shutil.copy2(src_mask, dst_mask)
    
    return test_images_dir, test_masks_dir

def calculate_dice(pred_path, gt_path):
    """Calculate Dice using pyKNEEr ONLY - stop if not available"""
    segmented_mask = sitk.ReadImage(pred_path)
    ground_truth_mask = sitk.ReadImage(gt_path)
    
    if sitkf is not None:
        #pyKNEEr
        dice, _, _ = sitkf.overlap_measures(segmented_mask, ground_truth_mask)
        return dice
    else:
        # STOP EXECUTION
        print("\n" + "="*80)
        print("CRITICAL ERROR: pyKNEEr is not available!")
        print("This study requires pyKNEEr for Dice calculations.")
        print("Please install ITK and ensure pyKNEEr is properly loaded.")
        print("STOPPING EXECUTION.")
        print("="*80)
        raise RuntimeError("pyKNEEr not available - cannot continue")
    
def run_testing_iteration(checkpoint_path, test_patients, iteration_num):
    print(f"\n  Testing on test patients...")
    
    #Data preparation
    test_images_dir, test_masks_dir = prepare_test_data(test_patients, iteration_num)
    
    # Directory for results 
    test_dir = os.path.join(RESULTS_DIR, f"test_iteration_{iteration_num}")
    images_slices = os.path.join(test_dir, "images_slices")
    masks_slices = os.path.join(test_dir, "masks_slices")
    predictions = os.path.join(test_dir, "predictions")
    segmentations = os.path.join(test_dir, "segmentations")
    postprocessed_dir = os.path.join(test_dir, "postprocessed")

    os.makedirs(images_slices, exist_ok=True)
    os.makedirs(masks_slices, exist_ok=True)
    os.makedirs(predictions, exist_ok=True)
    os.makedirs(segmentations, exist_ok=True)
    os.makedirs(postprocessed_dir, exist_ok=True)
    
    from unet_testing import test_main
    
    test_main(
        test_images_dir, test_masks_dir,
        images_slices, masks_slices,
        checkpoint_path, predictions,
        segmentations, postprocessed_dir,
        test_images_dir
    )
    

    dice_scores = []
    
    for patient in test_patients:
        patient_base = patient.replace('.mha', '')
        pred_path = os.path.join(postprocessed_dir, f"{patient_base}_modified.mha")
        gt_path = os.path.join(test_masks_dir, f"{patient_base}.nii.gz")
        
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            try:
                dice = calculate_dice(pred_path, gt_path)
                dice_scores.append(dice)
                print(f"    Patient {patient_base}: Dice = {dice:.4f}")

            except Exception as e:
                print(f"    ERROR computing Dice: {e}")
        else:  
            if not os.path.exists(pred_path):
                print(f"    WARNING: Prediction not found for {patient_base}")
                print(f"    Searched in: {pred_path}")
            if not os.path.exists(gt_path):
                print(f"    WARNING: Ground truth not found for {patient_base}")
                print(f"    Searched in: {gt_path}")
    
    if dice_scores:
        mean_dice = float(np.mean(dice_scores))
        std_dice = float(np.std(dice_scores, ddof=1)) if len(dice_scores) > 1 else 0.0
    else:
        mean_dice = 0.0
        std_dice = 0.0
        
    return mean_dice, std_dice, dice_scores

def cleanup_iteration(iteration_num):
    dirs_to_clean = [
        os.path.join(RESULTS_DIR, f"iteration_{iteration_num}_patients"),
        os.path.join(RESULTS_DIR, f"test_iteration_{iteration_num}")
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path) and iteration_num > MIN_PATIENTS:
            shutil.rmtree(dir_path)
            print(f"  Cleaned temporary files for iteration {iteration_num}")

def plot_convergence(results, baseline_dice=0.90):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    plt.figure(figsize=(16, 10))

    n_patients = results['n_patients']
    mean_dice = results['mean_dice']
    std_dice = results['std_dice']
    
    n_train_patients = []
    for i in range(len(n_patients)):
        n_val = len(results['val_patients'][i])
        n_test = len(results['test_patients'][i])
        n_train = n_patients[i] - n_val - n_test
        n_train_patients.append(n_train)
    
    plt.errorbar(n_patients, mean_dice, yerr=std_dice, 
                marker='o', markersize=12, linewidth=3,
                capsize=8, capthick=2.5, color='#1f77b4')
    
    for i, (n, dice) in enumerate(zip(n_patients, mean_dice)):
        plt.annotate(f'{dice:.3f}', 
                    (n, dice + 0.015), 
                    ha='center', fontsize=11, fontweight='bold')
    
    for i, (n, n_train) in enumerate(zip(n_patients, n_train_patients)):
        plt.annotate(f'({n_train} train)', 
                    (n, mean_dice[i] - 0.025), 
                    ha='center', fontsize=9, color='gray', style='italic')
    
    val_n = len(results["val_patients"][0])
    test_n = len(results["test_patients"][0])
    plt.xlabel(f'Total Number of Patients\n(Training + {val_n} Validation + {test_n} Test)', fontsize=16)
    plt.ylabel('Dice Coefficient (mean ± SD)', fontsize=16)
    plt.title('Fine-Tuning Convergence: Growing Training Set with Fixed Val/Test', fontsize=18)
    
    x_margin = 0.5
    plt.xlim(min(n_patients) - x_margin, max(n_patients) + x_margin)
    plt.ylim(0.75, 0.95)
    plt.xticks(n_patients, fontsize=12)
    plt.yticks(np.arange(0.75, 0.96, 0.05), fontsize=12)
    
    plt.minorticks_on()
    plt.grid(True, alpha=0.3, which='major', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.15, which='minor', linestyle=':', linewidth=0.3)
    
    plt.axhline(y=0.85, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='0.85 threshold')
    
    plt.axhline(y=baseline_dice, color='darkred', linestyle='--', linewidth=2, 
            alpha=0.6, label=f'Original model (no fine-tuning): {baseline_dice:.2f}')
    
    plt.legend(loc='lower right', fontsize=12)
    
    # Info box
    textstr = f'Val set: {val_n} patients (fixed)\nTest set: {test_n} patients (fixed)\nTraining set: 1-{max(n_train_patients)} patients (growing)'
    props = dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray')
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    if len(n_patients) > 8:
        improvements = np.diff(mean_dice)
        
        max_dice = np.max(mean_dice)
        threshold_dice = 0.99 * max_dice
        
        plateau_idx = None
        for i in range(len(mean_dice)):
            if mean_dice[i] >= threshold_dice:
                if all(mean_dice[j] >= threshold_dice * 0.995 for j in range(i, len(mean_dice))):
                    plateau_idx = i
                    break
        
        if plateau_idx is None and len(mean_dice) > 10:
            window_size = 4
            variances = []
            
            for i in range(len(mean_dice) - window_size + 1):
                window = mean_dice[i:i+window_size]
                variances.append(np.var(window))
            
            variance_threshold = 0.0001
            for i in range(len(variances)):
                if variances[i] < variance_threshold:
                    if all(v < variance_threshold * 2 for v in variances[i:min(i+3, len(variances))]):
                        plateau_idx = i + window_size // 2
                        break
        
        if plateau_idx is None:
            consecutive_small = 0
            threshold_improvement = 0.003
            
            for i in range(1, len(mean_dice)):
                if i < len(improvements) and abs(improvements[i-1]) < threshold_improvement:
                    consecutive_small += 1
                    if consecutive_small >= 4:
                        plateau_idx = i - 3
                        break
                else:
                    consecutive_small = 0
        
        if plateau_idx is not None and plateau_idx < len(n_patients):
            plt.axvline(x=n_patients[plateau_idx], color='red', linestyle=':', linewidth=2, alpha=0.5)
            
            plateau_dice = mean_dice[plateau_idx]
            
            plt.text(n_patients[plateau_idx], 0.77, 
                    f'Performance stabilized\n({n_train_patients[plateau_idx]} training patients)\nDice: {plateau_dice:.3f}', 
                    ha='center', fontsize=10, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, 'convergence_plot_complete.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'convergence_plot_complete.pdf'), bbox_inches='tight')
    plt.show()
    
    print("\nDetailed analysis of improvements:")
    print("N_patients | Dice  | Improvement | % Change")
    print("-" * 55)
    for i in range(len(n_patients)):
        if i == 0:
            print(f"{n_patients[i]:10} | {mean_dice[i]:.3f} |     -      |    -")
        else:
            improvement = mean_dice[i] - mean_dice[i-1]
            pct_change = (improvement / mean_dice[i-1]) * 100
            print(f"{n_patients[i]:10} | {mean_dice[i]:.3f} | {improvement:+.3f}    | {pct_change:+.1f}%")
    
    print("\n" + "="*55)
    print("Plateau analysis:")
    if len(mean_dice) > 5:
        max_dice = np.max(mean_dice)
        dice_95 = 0.95 * max_dice
        dice_99 = 0.99 * max_dice
        
        idx_95 = next((i for i, d in enumerate(mean_dice) if d >= dice_95), None)
        idx_99 = next((i for i, d in enumerate(mean_dice) if d >= dice_99), None)
        
        print(f"- 95% of maximum ({dice_95:.3f}) reached at {n_patients[idx_95]} patients")
        print(f"- 99% of maximum ({dice_99:.3f}) reached at {n_patients[idx_99]} patients")
        
        if len(improvements) >= 5:
            last_5_avg = np.mean(improvements[-5:])
            print(f"- Average improvement last 5 points: {last_5_avg:.4f} ({last_5_avg*100:.2f}%)")
        
        print(f"\n- Original baseline: {baseline_dice:.3f}")
        print(f"- Maximum reached: {max_dice:.3f} (+{(max_dice-baseline_dice)*100:.1f}% compared to baseline)")

def main():
    print("="*80)
    print("INCREMENTAL FINE-TUNING CONVERGENCE STUDY")
    print("="*80)
    
    if sitkf is None:
        print("\nCRITICAL ERROR: pyKNEEr is required but not loaded!")
        print("This study MUST use pyKNEEr for Dice calculations.")
        print("Cannot proceed with SimpleITK fallback.")
        print("\nPlease:")
        print("1. Install ITK: pip install itk")
        print("2. Verify pyKNEEr path is correct")
        print("3. Restart the script")
        print("\nEXITING NOW.")
        sys.exit(1) 
    else:
        print("✓ pyKNEEr loaded successfully - using for all Dice calculations")
    
    all_patients = get_available_patients()
    if len(all_patients) < MIN_PATIENTS:
        print(f"ERROR: At least {MIN_PATIENTS} patients required!")
        return
    
    max_possible = min(MAX_PATIENTS, len(all_patients))
    print(f"Study from {MIN_PATIENTS} to {max_possible} patients")

    np.random.seed(RANDOM_SEED)
    shuffled = np.random.permutation(all_patients).tolist()
    n_all = len(shuffled)

    n_test_fixed = int(round(0.18 * n_all)) 
    n_val_fixed = int(round(0.12 * n_all))  

    n_test_fixed = max(5, n_test_fixed)  # At least 5 for test
    n_val_fixed = max(3, n_val_fixed)    # At least 3 for validation

    # fixed patients
    test_fixed = shuffled[-n_test_fixed:]
    val_fixed = shuffled[-n_test_fixed-n_val_fixed:-n_test_fixed]
    train_pool = [p for p in shuffled if p not in (set(val_fixed) | set(test_fixed))]

    start_n = n_test_fixed + n_val_fixed + 1

    print(f"Fixed sets: test={len(test_fixed)}, val={len(val_fixed)}")
    print(f"Train pool: {len(train_pool)} patients available")
    print(f"Starting from n={start_n}")

    
    results = {
        'n_patients': [],
        'mean_dice': [],
        'std_dice': [],
        'train_patients': [],
        'val_patients': [],
        'test_patients': []
    }
    
    for n_patients in range(max(MIN_PATIENTS, start_n), max_possible + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION: {n_patients} total patients")
        print(f"{'='*80}")
        
        n_train = n_patients - len(val_fixed) - len(test_fixed)
        train_p = train_pool[:n_train]
        val_p = val_fixed
        test_p = test_fixed

        print(f"  Split: {n_train} train, {len(val_p)} val, {len(test_p)} test")
                
        checkpoint = run_finetuning_iteration(train_p, val_p, n_patients)
        if checkpoint is None:
            continue
        
        mean_dice, std_dice, dice_scores = run_testing_iteration(checkpoint, test_p, n_patients)
        
        results['n_patients'].append(n_patients)
        results['mean_dice'].append(mean_dice)
        results['std_dice'].append(std_dice)
        results['train_patients'].append([p.replace('.mha','') for p in train_p])
        results['val_patients'].append([p.replace('.mha','') for p in val_p])
        results['test_patients'].append([p.replace('.mha','') for p in test_p])
        
        print(f"\n  RESULT: Dice = {mean_dice:.4f} ± {std_dice:.4f}")

        with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # cleaning (optional)
        #cleanup_iteration(n_patients)
    
    # final plot
    print("\nCreating plot")
    plot_convergence(results, baseline_dice=0.90)  
    print(f"\nStudy completed! Results in: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
