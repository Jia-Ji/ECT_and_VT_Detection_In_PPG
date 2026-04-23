import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Optional


class FinetuneDataSplitter:
    """
    Data splitting class for fine-tuning datasets.
    Allows specifying exact number of patients and samples per patient for each split.
    Ensures patients only exist in one split (train, valid, or test).
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize FinetuneDataSplitter with configuration from config file.
        
        Args:
            config: DictConfig containing splitting parameters
        """
        self.config = config
        self.split_cfg = config.finetune_split
        
        # Load parameters from config
        self.input_path = self.split_cfg.get('input_path')
        self.output_dir = self.split_cfg.get('output_dir')
        self.target_labels = self.split_cfg.get('target_labels', ['NORM', 'ECT', 'VT'])
        self.label_mapping = self.split_cfg.get('label_mapping', {'NORM': 0, 'ECT': 1, 'VT': 2})
        
        # Number of patients for each split
        self.n_train_patients = self.split_cfg.get('n_train_patients', 10)
        self.n_val_patients = self.split_cfg.get('n_val_patients', 5)
        self.n_test_patients = self.split_cfg.get('n_test_patients', -1)  # -1 means all remaining
        
        # Number of samples per patient
        self.n_samples_per_patient_train = self.split_cfg.get('n_samples_per_patient_train', -1)
        self.n_samples_per_patient_val = self.split_cfg.get('n_samples_per_patient_val', -1)
        self.n_samples_per_patient_test = self.split_cfg.get('n_samples_per_patient_test', -1)
        
        # Random seed
        self.random_seed = self.split_cfg.get('random_seed', 42)
        
        # Whether to stratify patient selection
        self.stratify_patients = self.split_cfg.get('stratify_patients', True)
        
        # Validate required paths
        if not self.input_path:
            raise ValueError("input_path must be specified in config.finetune_split")
        if not self.output_dir:
            raise ValueError("output_dir must be specified in config.finetune_split")
    
    def load_filtered_data(self) -> pd.DataFrame:
        """Load the pickle file from preprocessing"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        df = pd.read_pickle(self.input_path)
        print(f"Loaded dataframe with shape {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def filter_and_encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to only include target labels and encode them.
        Merges 'PAC' and 'PVC' into 'ECT' class.
        Keeps 'VT' as a separate class.
        """
        # Check for case variations (Norm/NORM, PAC, PVC)
        available_labels = df['label'].unique()
        print(f"\nAvailable labels in data: {available_labels}")
        
        # Define source labels that will be merged into target labels
        source_labels_to_target = {
            'NORM': 'NORM',
            'PAC': 'ECT',
            'PVC': 'ECT',
            'ECT': 'ECT',
            'VT': 'VT'
        }
        
        # Try to match source labels (case-insensitive)
        matched_source_labels = []
        label_mapping_normalized = {}
        
        for source_label, target_label in source_labels_to_target.items():
            # Try exact match first
            if source_label in available_labels:
                matched_source_labels.append(source_label)
                label_mapping_normalized[source_label] = self.label_mapping[target_label]
            # Try uppercase
            elif source_label.upper() in available_labels:
                matched_label = source_label.upper()
                matched_source_labels.append(matched_label)
                label_mapping_normalized[matched_label] = self.label_mapping[target_label]
            # Try case-insensitive match
            else:
                for avail_label in available_labels:
                    if str(avail_label).upper() == source_label.upper():
                        matched_source_labels.append(avail_label)
                        label_mapping_normalized[avail_label] = self.label_mapping[target_label]
                        break
        
        if not matched_source_labels:
            raise ValueError(f"None of the source labels {list(source_labels_to_target.keys())} found in data. Available labels: {available_labels}")
        
        print(f"\nMatched source labels: {matched_source_labels}")
        print(f"Label mapping (source -> encoded): {label_mapping_normalized}")
        
        # Filter dataframe to only include matched source labels
        df_filtered = df[df['label'].isin(matched_source_labels)].copy()
        print(f"\nFiltered dataframe shape: {df_filtered.shape}")
        print(f"Label distribution after filtering (before merging):\n{df_filtered['label'].value_counts()}")
        
        # Map source labels to target labels (NORM stays NORM, PAC/PVC become ECT)
        def map_to_target_label(source_label):
            for src, tgt in source_labels_to_target.items():
                if str(source_label).upper() == src.upper():
                    return tgt
            return source_label
        
        df_filtered['label'] = df_filtered['label'].apply(map_to_target_label)
        print(f"\nLabel distribution after merging to target labels:\n{df_filtered['label'].value_counts()}")
        
        # Encode labels using target label mapping
        df_filtered['encoded_label'] = df_filtered['label'].map(self.label_mapping)
        
        # Check for any unmapped labels
        if df_filtered['encoded_label'].isna().any():
            unmapped = df_filtered[df_filtered['encoded_label'].isna()]['label'].unique()
            raise ValueError(f"Some labels could not be mapped: {unmapped}")
        
        return df_filtered
    
    def get_patient_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics per patient.
        
        Returns:
            DataFrame with patient ID as index and label counts/proportions as columns
        """
        # Create a summary of labels per patient
        summary = (
            df.groupby(['ID0', 'label'])
            .size()
            .reset_index(name='count')
        )
        
        # Create pivot table: rows = ID0, columns = label, values = count
        pivot = summary.pivot(index='ID0', columns='label', values='count').fillna(0)
        
        # Calculate total beats per patient
        pivot['total'] = pivot.sum(axis=1)
        
        # Calculate ECT proportion for stratification
        label_cols = [col for col in pivot.columns if col != 'total']
        for col in label_cols:
            pivot[f'{col}_prop'] = pivot[col] / pivot['total']
        
        return pivot
    
    def select_patients_stratified(self, patient_summary: pd.DataFrame, n_patients: int, 
                                   exclude_patients: set = None) -> np.ndarray:
        """
        Select patients with stratification based on ECT proportion.
        
        Args:
            patient_summary: DataFrame with patient statistics
            n_patients: Number of patients to select
            exclude_patients: Set of patient IDs to exclude
            
        Returns:
            Array of selected patient IDs
        """
        np.random.seed(self.random_seed)
        
        # Get available patients
        available_patients = patient_summary.index.tolist()
        if exclude_patients:
            available_patients = [p for p in available_patients if p not in exclude_patients]
        
        if n_patients == -1 or n_patients >= len(available_patients):
            return np.array(available_patients)
        
        if not self.stratify_patients:
            # Simple random sampling
            selected = np.random.choice(available_patients, size=n_patients, replace=False)
            return selected
        
        # Stratified sampling based on ECT proportion
        # Bin patients by ECT proportion
        available_summary = patient_summary.loc[available_patients].copy()
        
        if 'ECT_prop' in available_summary.columns:
            # Create bins based on ECT proportion
            available_summary['ect_bin'] = pd.qcut(
                available_summary['ECT_prop'].rank(method='first'), 
                q=min(5, len(available_patients) // 2),
                labels=False,
                duplicates='drop'
            )
        else:
            available_summary['ect_bin'] = 0
        
        # Sample from each bin proportionally
        selected = []
        bins = available_summary['ect_bin'].unique()
        n_bins = len(bins)
        
        # Calculate patients per bin
        base_per_bin = n_patients // n_bins
        extra = n_patients % n_bins
        
        for i, bin_val in enumerate(bins):
            bin_patients = available_summary[available_summary['ect_bin'] == bin_val].index.tolist()
            n_to_select = base_per_bin + (1 if i < extra else 0)
            n_to_select = min(n_to_select, len(bin_patients))
            
            if n_to_select > 0:
                bin_selected = np.random.choice(bin_patients, size=n_to_select, replace=False)
                selected.extend(bin_selected)
        
        return np.array(selected)
    
    def sample_from_patients(self, df: pd.DataFrame, patient_ids: np.ndarray, 
                            n_samples_per_patient: int) -> pd.DataFrame:
        """
        Sample a fixed number of samples from each patient.
        
        Args:
            df: Full dataframe
            patient_ids: Array of patient IDs to include
            n_samples_per_patient: Number of samples per patient (-1 for all)
            
        Returns:
            DataFrame with sampled data
        """
        np.random.seed(self.random_seed)
        
        # Filter to selected patients
        df_patients = df[df['ID0'].isin(patient_ids)].copy()
        
        if n_samples_per_patient == -1:
            # Use all samples
            return df_patients
        
        # Sample n_samples_per_patient from each patient
        sampled_dfs = []
        
        for patient_id in patient_ids:
            patient_df = df_patients[df_patients['ID0'] == patient_id]
            n_available = len(patient_df)
            
            if n_available <= n_samples_per_patient:
                # Use all available samples
                sampled_dfs.append(patient_df)
            else:
                # Random sample
                sampled_indices = np.random.choice(
                    patient_df.index, 
                    size=n_samples_per_patient, 
                    replace=False
                )
                sampled_dfs.append(patient_df.loc[sampled_indices])
        
        if sampled_dfs:
            return pd.concat(sampled_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def split_by_patient_count(self, df: pd.DataFrame) -> tuple:
        """
        Split dataset by specifying exact number of patients for each split.
        
        Returns:
            Tuple of (df_train, df_val, df_test)
        """
        np.random.seed(self.random_seed)
        
        # Get patient summary
        patient_summary = self.get_patient_summary(df)
        all_patients = set(patient_summary.index)
        
        print(f"\nTotal unique patients (ID0): {len(all_patients)}")
        print(f"Requested: Train={self.n_train_patients}, Val={self.n_val_patients}, Test={self.n_test_patients}")
        
        # Select train patients
        train_patients = self.select_patients_stratified(
            patient_summary, 
            self.n_train_patients,
            exclude_patients=set()
        )
        train_patient_set = set(train_patients)
        print(f"\nSelected {len(train_patients)} patients for training")
        
        # Select validation patients (excluding train)
        val_patients = self.select_patients_stratified(
            patient_summary,
            self.n_val_patients,
            exclude_patients=train_patient_set
        )
        val_patient_set = set(val_patients)
        print(f"Selected {len(val_patients)} patients for validation")
        
        # Select test patients (all remaining or specified number)
        remaining_patients = all_patients - train_patient_set - val_patient_set
        
        if self.n_test_patients == -1:
            test_patients = np.array(list(remaining_patients))
        else:
            test_patients = self.select_patients_stratified(
                patient_summary,
                self.n_test_patients,
                exclude_patients=train_patient_set | val_patient_set
            )
        test_patient_set = set(test_patients)
        print(f"Selected {len(test_patients)} patients for testing")
        
        # Verify no overlap
        assert len(train_patient_set & val_patient_set) == 0, "Overlap between train and val patients!"
        assert len(train_patient_set & test_patient_set) == 0, "Overlap between train and test patients!"
        assert len(val_patient_set & test_patient_set) == 0, "Overlap between val and test patients!"
        print("\n✓ No patient overlap between splits")
        
        # Sample data from each split
        df_train = self.sample_from_patients(df, train_patients, self.n_samples_per_patient_train)
        df_val = self.sample_from_patients(df, val_patients, self.n_samples_per_patient_val)
        df_test = self.sample_from_patients(df, test_patients, self.n_samples_per_patient_test)
        
        print(f"\nSplit sizes (samples):")
        print(f"  Train: {len(df_train)} samples from {len(train_patients)} patients")
        print(f"  Val: {len(df_val)} samples from {len(val_patients)} patients")
        print(f"  Test: {len(df_test)} samples from {len(test_patients)} patients")
        
        # Print label distribution per split
        print(f"\nLabel distribution per split:")
        for name, df_split in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
            if len(df_split) > 0:
                dist = df_split['label'].value_counts(normalize=True) * 100
                print(f"  {name}: {dict(dist.round(2))}")
        
        return df_train, df_val, df_test
    
    def fp_to_mask(self, fp: pd.DataFrame, signal_length: int) -> np.ndarray:
        """
        Convert fiducial points DataFrame to a 4-channel mask.
        
        Args:
            fp: DataFrame with columns 'on', 'sp', 'dn', 'dp' containing fiducial point indices.
            signal_length: Length of the signal to create mask for
            
        Returns:
            4-channel mask array of shape (signal_length, 4)
        """
        mask = np.zeros((signal_length, 4), dtype=np.float32)
        
        fp_columns = ['on', 'sp', 'dn', 'dp']
        channel_idx = 0
        
        for col in fp_columns:
            if col in fp.columns:
                fp_values = fp[col]
                fp_values = fp_values.to_numpy()
                assert isinstance(fp_values, (list, np.ndarray)), \
                    f"fp_values must be a list or numpy array, but got {type(fp_values)}"
                
                if isinstance(fp_values, (list, np.ndarray)):
                    fp_indices = np.asarray(fp_values, dtype=float)
                    fp_indices = fp_indices[~np.isnan(fp_indices)]
                    fp_indices = fp_indices.astype(int)
                    fp_indices = fp_indices[(fp_indices >= 0) & (fp_indices < signal_length)]
                    if len(fp_indices) > 0:
                        mask[fp_indices, channel_idx] = 1.0
            channel_idx += 1
        
        return mask
    
    def extract_arrays(self, df_split: pd.DataFrame) -> tuple:
        """Extract ECG, PPG, label, filename, fp mask, hr, rmssd, sdsd, bm_stats, ID0, and sqi arrays from dataframe split"""
        df_split = df_split.reset_index(drop=True)
        
        ecg_list = []
        ppg_list = []
        labels_list = []
        filenames_list = []
        fp_masks_list = []
        hr_list = []
        rmssd_list = []
        sdsd_list = []
        bm_stats_list = []
        id0_list = []
        sqi_list = []
        sqi_sample_list = []
        
        for idx in range(len(df_split)):
            ecg_val = df_split.iloc[idx]['ECG']
            ppg_val = df_split.iloc[idx]['PPG']
            fp_value = df_split.iloc[idx]['fp']
            label_val = df_split.iloc[idx]['encoded_label']
            filename_val = df_split.iloc[idx]['Filename']
            hr_val = df_split.iloc[idx]['hr']
            rmssd_val = df_split.iloc[idx]['hrv'].get('RMSSD', None)
            sdsd_val = df_split.iloc[idx]['hrv'].get('SDSD', None)
            bm_stats_val = df_split.iloc[idx]['bm']
            id0_val = df_split.iloc[idx]['ID0']
            sqi_val = df_split.iloc[idx]['sqi']
            sqi_sample_val = df_split.iloc[idx]['sqi_sample']

            ecg_list.append(np.asarray(ecg_val))
            ppg_array = np.asarray(ppg_val)
            ppg_list.append(np.expand_dims(ppg_array, axis=0))
            
            signal_length = len(ppg_array)
            fp_mask = self.fp_to_mask(fp_value, signal_length)
            
            fp_masks_list.append(fp_mask)
            labels_list.append(label_val)
            filenames_list.append(str(filename_val))
            hr_list.append(hr_val)
            rmssd_list.append(rmssd_val)
            sdsd_list.append(sdsd_val)
            bm_stats_list.append(bm_stats_val)
            id0_list.append(str(id0_val))
            sqi_list.append(float(sqi_val))
            sqi_sample_list.append(sqi_sample_val)
        
        ecg = np.array(ecg_list)
        ppg = np.array(ppg_list)
        labels = np.array(labels_list, dtype=np.int64)
        filenames = np.array(filenames_list, dtype=object)
        fp_masks = np.array(fp_masks_list)
        hr = np.array(hr_list, dtype=np.float32)
        rmssd = np.array(rmssd_list, dtype=np.float32)
        sdsd = np.array(sdsd_list, dtype=np.float32)
        bm_stats = np.array(bm_stats_list, dtype=object)
        id0 = np.array(id0_list, dtype=object)
        sqi = np.array(sqi_list, dtype=np.float32)
        sqi_sample = np.array(sqi_sample_list)
        
        # Validate all arrays have the same length
        lengths = {
            'ECG': len(ecg),
            'PPG': len(ppg),
            'labels': len(labels),
            'filenames': len(filenames),
            'fp_masks': len(fp_masks),
            'hr': len(hr),
            'rmssd': len(rmssd),
            'sdsd': len(sdsd),
            'bm_stats': len(bm_stats),
            'ID0': len(id0),
            'sqi': len(sqi),
            'sqi_sample': len(sqi_sample)
        }
        
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Length mismatch in extracted arrays: {lengths}")
        
        return ecg, ppg, labels, filenames, fp_masks, hr, rmssd, sdsd, bm_stats, id0, sqi, sqi_sample
    
    def compute_label_distribution(self, df_split: pd.DataFrame) -> tuple:
        """
        Compute label distribution for a split.
        Returns both counts and percentages as dictionaries.
        """
        counts = df_split['label'].value_counts()
        percentages = df_split['label'].value_counts(normalize=True) * 100
        
        target_order = list(self.label_mapping.keys())
        
        counts_dict = {}
        percentages_dict = {}
        
        for label in target_order:
            if label in counts.index:
                counts_dict[label] = int(counts[label])
                percentages_dict[label] = float(percentages[label])
            else:
                counts_dict[label] = 0
                percentages_dict[label] = 0.0
        
        return counts_dict, percentages_dict
    
    def save_splits(self, ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train,
                    ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val,
                    ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test):
        """Save train/val/test splits to output directory"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        def validate_split(name, ecg, ppg, labels, filenames, fp_masks, hr, rmssd, sdsd, bm_stats, id0, sqi, sqi_sample):
            lengths = {
                'ECG': len(ecg),
                'PPG': len(ppg),
                'labels': len(labels),
                'filenames': len(filenames),
                'fp_masks': len(fp_masks),
                'hr': len(hr),
                'rmssd': len(rmssd),
                'sdsd': len(sdsd),
                'bm_stats': len(bm_stats),
                'ID0': len(id0),
                'sqi': len(sqi),
                'sqi_sample': len(sqi_sample)
            }
            if len(set(lengths.values())) > 1:
                raise ValueError(f"Length mismatch in {name} split: {lengths}")
            print(f"  {name}: All arrays have length {lengths['ECG']}")
            return lengths['ECG']
        
        print("\nValidating splits before saving...")
        train_len = validate_split("Train", ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train)
        val_len = validate_split("Val", ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val)
        test_len = validate_split("Test", ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test)
        
        # Save train split
        np.save(os.path.join(self.output_dir, 'ecg_train.npy'), ecg_train)
        np.save(os.path.join(self.output_dir, 'ppg_train.npy'), ppg_train)
        np.save(os.path.join(self.output_dir, 'labels_train.npy'), labels_train)
        np.save(os.path.join(self.output_dir, 'filenames_train.npy'), filenames_train)
        np.save(os.path.join(self.output_dir, 'fp_masks_train.npy'), fp_masks_train)
        np.save(os.path.join(self.output_dir, 'hr_train.npy'), hr_train)
        np.save(os.path.join(self.output_dir, 'rmssd_train.npy'), rmssd_train)
        np.save(os.path.join(self.output_dir, 'sdsd_train.npy'), sdsd_train)
        np.save(os.path.join(self.output_dir, 'id0_train.npy'), id0_train)
        np.save(os.path.join(self.output_dir, 'sqi_train.npy'), sqi_train)
        np.save(os.path.join(self.output_dir, 'sqi_sample_train.npy'), sqi_sample_train)
        if len(bm_stats_train) > 0:
            bm_stats_df_train = pd.DataFrame(bm_stats_train)
            bm_stats_df_train.to_csv(os.path.join(self.output_dir, 'bm_stats_train.csv'), index=False)
        
        # Save val split
        np.save(os.path.join(self.output_dir, 'ecg_val.npy'), ecg_val)
        np.save(os.path.join(self.output_dir, 'ppg_val.npy'), ppg_val)
        np.save(os.path.join(self.output_dir, 'labels_val.npy'), labels_val)
        np.save(os.path.join(self.output_dir, 'filenames_val.npy'), filenames_val)
        np.save(os.path.join(self.output_dir, 'fp_masks_val.npy'), fp_masks_val)
        np.save(os.path.join(self.output_dir, 'hr_val.npy'), hr_val)
        np.save(os.path.join(self.output_dir, 'rmssd_val.npy'), rmssd_val)
        np.save(os.path.join(self.output_dir, 'sdsd_val.npy'), sdsd_val)
        np.save(os.path.join(self.output_dir, 'id0_val.npy'), id0_val)
        np.save(os.path.join(self.output_dir, 'sqi_val.npy'), sqi_val)
        np.save(os.path.join(self.output_dir, 'sqi_sample_val.npy'), sqi_sample_val)
        if len(bm_stats_val) > 0:
            bm_stats_df_val = pd.DataFrame(bm_stats_val)
            bm_stats_df_val.to_csv(os.path.join(self.output_dir, 'bm_stats_val.csv'), index=False)
        
        # Save test split
        np.save(os.path.join(self.output_dir, 'ecg_test.npy'), ecg_test)
        np.save(os.path.join(self.output_dir, 'ppg_test.npy'), ppg_test)
        np.save(os.path.join(self.output_dir, 'labels_test.npy'), labels_test)
        np.save(os.path.join(self.output_dir, 'filenames_test.npy'), filenames_test)
        np.save(os.path.join(self.output_dir, 'fp_masks_test.npy'), fp_masks_test)
        np.save(os.path.join(self.output_dir, 'hr_test.npy'), hr_test)
        np.save(os.path.join(self.output_dir, 'rmssd_test.npy'), rmssd_test)
        np.save(os.path.join(self.output_dir, 'sdsd_test.npy'), sdsd_test)
        np.save(os.path.join(self.output_dir, 'id0_test.npy'), id0_test)
        np.save(os.path.join(self.output_dir, 'sqi_test.npy'), sqi_test)
        np.save(os.path.join(self.output_dir, 'sqi_sample_test.npy'), sqi_sample_test)
        if len(bm_stats_test) > 0:
            bm_stats_df_test = pd.DataFrame(bm_stats_test)
            bm_stats_df_test.to_csv(os.path.join(self.output_dir, 'bm_stats_test.csv'), index=False)
        
        print(f"\n✓ Saved all splits to {self.output_dir}")
        print(f"  Train: {train_len} samples, Val: {val_len} samples, Test: {test_len} samples")
    
    def save_distribution_report(self, train_counts, train_percentages, 
                                val_counts, val_percentages,
                                test_counts, test_percentages,
                                n_train_patients, n_val_patients, n_test_patients):
        """Save label distribution report to a text file"""
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, 'data_distribution.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINETUNE DATA SPLITTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Write split configuration
            f.write("Split Configuration:\n")
            f.write(f"  Random Seed: {self.random_seed}\n")
            f.write(f"  Stratify Patients: {self.stratify_patients}\n")
            f.write(f"  Train Patients: {n_train_patients} (requested: {self.n_train_patients})\n")
            f.write(f"  Val Patients: {n_val_patients} (requested: {self.n_val_patients})\n")
            f.write(f"  Test Patients: {n_test_patients} (requested: {self.n_test_patients})\n")
            f.write(f"  Samples per Patient (Train): {self.n_samples_per_patient_train}\n")
            f.write(f"  Samples per Patient (Val): {self.n_samples_per_patient_val}\n")
            f.write(f"  Samples per Patient (Test): {self.n_samples_per_patient_test}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Write label mapping
            f.write("Label Encoding:\n")
            for label, code in self.label_mapping.items():
                f.write(f"  {label}: {code}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Train set distribution
            f.write("TRAIN SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_train = sum(train_counts.values())
            f.write(f"Total samples: {total_train}\n")
            f.write(f"Number of patients: {n_train_patients}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = train_counts[label]
                pct = train_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Validation set distribution
            f.write("VALIDATION SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_val = sum(val_counts.values())
            f.write(f"Total samples: {total_val}\n")
            f.write(f"Number of patients: {n_val_patients}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = val_counts[label]
                pct = val_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Test set distribution
            f.write("TEST SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_test = sum(test_counts.values())
            f.write(f"Total samples: {total_test}\n")
            f.write(f"Number of patients: {n_test_patients}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = test_counts[label]
                pct = test_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Summary comparison
            f.write("=" * 80 + "\n")
            f.write("DISTRIBUTION COMPARISON ACROSS SPLITS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Label':<15} {'Train %':<15} {'Val %':<15} {'Test %':<15} {'Max Diff %':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                train_pct = train_percentages[label]
                val_pct = val_percentages[label]
                test_pct = test_percentages[label]
                max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
                f.write(f"{label:<15} {train_pct:>10.2f}%   {val_pct:>10.2f}%   {test_pct:>10.2f}%   {max_diff:>10.2f}%\n")
            f.write("\n")
            
            # Overall statistics
            f.write("=" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total samples across all splits: {total_train + total_val + total_test}\n")
            f.write(f"Total patients: {n_train_patients + n_val_patients + n_test_patients}\n")
            f.write(f"Train: {total_train} samples ({total_train/(total_train+total_val+total_test)*100:.2f}%)\n")
            f.write(f"Validation: {total_val} samples ({total_val/(total_train+total_val+total_test)*100:.2f}%)\n")
            f.write(f"Test: {total_test} samples ({total_test/(total_train+total_val+total_test)*100:.2f}%)\n")
        
        print(f"\n✓ Saved distribution report to {report_path}")
    
    def run(self) -> None:
        """
        Run the complete finetune data splitting pipeline:
        1. Load data
        2. Filter and encode labels
        3. Split by patient count
        4. Extract arrays
        5. Compute distributions
        6. Save splits and reports
        """
        print("=" * 80)
        print("FINETUNE DATA SPLITTING PIPELINE")
        print("=" * 80)
        print(f"Input file: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target labels: {self.target_labels}")
        print(f"Label mapping: {self.label_mapping}")
        print(f"Patient counts: Train={self.n_train_patients}, Val={self.n_val_patients}, Test={self.n_test_patients}")
        print(f"Samples per patient: Train={self.n_samples_per_patient_train}, Val={self.n_samples_per_patient_val}, Test={self.n_samples_per_patient_test}")
        print(f"Random seed: {self.random_seed}")
        print(f"Stratify patients: {self.stratify_patients}")
        print("=" * 80)
        
        # Load data
        df = self.load_filtered_data()
        
        # Filter and encode labels
        df_filtered = self.filter_and_encode_labels(df)
        
        # Split by patient count
        df_train, df_val, df_test = self.split_by_patient_count(df_filtered)
        
        # Count unique patients in each split
        n_train_patients = df_train['ID0'].nunique() if len(df_train) > 0 else 0
        n_val_patients = df_val['ID0'].nunique() if len(df_val) > 0 else 0
        n_test_patients = df_test['ID0'].nunique() if len(df_test) > 0 else 0
        
        # Print dataframe info before extraction
        print(f"\n=== Dataframe Info Before Extraction ===")
        print(f"Train dataframe: {len(df_train)} rows from {n_train_patients} patients")
        print(f"Val dataframe: {len(df_val)} rows from {n_val_patients} patients")
        print(f"Test dataframe: {len(df_test)} rows from {n_test_patients} patients")
        
        # Extract arrays
        print(f"\n=== Extracting Arrays ===")
        ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train = self.extract_arrays(df_train)
        print(f"✓ Extracted train arrays")
        ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val = self.extract_arrays(df_val)
        print(f"✓ Extracted val arrays")
        ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test = self.extract_arrays(df_test)
        print(f"✓ Extracted test arrays")
        
        # Print shapes
        print(f"\n=== Split Shapes ===")
        print(f"Train - ECG: {ecg_train.shape}, PPG: {ppg_train.shape}, Labels: {labels_train.shape}, Filenames: {filenames_train.shape}, FP Masks: {fp_masks_train.shape}, HR: {hr_train.shape}, rmssd: {rmssd_train.shape}, sdsd: {sdsd_train.shape}, bm_stats: {bm_stats_train.shape}, ID0: {id0_train.shape}, SQI: {sqi_train.shape}, SQI Sample: {sqi_sample_train.shape}")
        print(f"Val   - ECG: {ecg_val.shape}, PPG: {ppg_val.shape}, Labels: {labels_val.shape}, Filenames: {filenames_val.shape}, FP Masks: {fp_masks_val.shape}, HR: {hr_val.shape}, rmssd: {rmssd_val.shape}, sdsd: {sdsd_val.shape}, bm_stats: {bm_stats_val.shape}, ID0: {id0_val.shape}, SQI: {sqi_val.shape}, SQI Sample: {sqi_sample_val.shape}")
        print(f"Test  - ECG: {ecg_test.shape}, PPG: {ppg_test.shape}, Labels: {labels_test.shape}, Filenames: {filenames_test.shape}, FP Masks: {fp_masks_test.shape}, HR: {hr_test.shape}, rmssd: {rmssd_test.shape}, sdsd: {sdsd_test.shape}, bm_stats: {bm_stats_test.shape}, ID0: {id0_test.shape}, SQI: {sqi_test.shape}, SQI Sample: {sqi_sample_test.shape}")
        
        # Compute label distributions
        print(f"\n=== Label Distribution ===")
        train_counts, train_percentages = self.compute_label_distribution(df_train)
        val_counts, val_percentages = self.compute_label_distribution(df_val)
        test_counts, test_percentages = self.compute_label_distribution(df_test)
        
        # Print distributions
        print(f"\nTrain Set Distribution:")
        total_train = sum(train_counts.values())
        print(f"  Total samples: {total_train}, Patients: {n_train_patients}")
        for label in self.label_mapping.keys():
            count = train_counts[label]
            pct = train_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print(f"\nValidation Set Distribution:")
        total_val = sum(val_counts.values())
        print(f"  Total samples: {total_val}, Patients: {n_val_patients}")
        for label in self.label_mapping.keys():
            count = val_counts[label]
            pct = val_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print(f"\nTest Set Distribution:")
        total_test = sum(test_counts.values())
        print(f"  Total samples: {total_test}, Patients: {n_test_patients}")
        for label in self.label_mapping.keys():
            count = test_counts[label]
            pct = test_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        # Print distribution comparison
        print(f"\n=== Distribution Comparison Across Splits ===")
        print(f"{'Label':<15} {'Train %':<15} {'Val %':<15} {'Test %':<15} {'Max Diff %':<15}")
        print("-" * 80)
        for label in self.label_mapping.keys():
            train_pct = train_percentages[label]
            val_pct = val_percentages[label]
            test_pct = test_percentages[label]
            max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
            print(f"{label:<15} {train_pct:>10.2f}%   {val_pct:>10.2f}%   {test_pct:>10.2f}%   {max_diff:>10.2f}%")
        
        # Save splits
        self.save_splits(
            ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train,
            ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val,
            ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test
        )
        
        # Save distribution report
        self.save_distribution_report(
            train_counts, train_percentages,
            val_counts, val_percentages,
            test_counts, test_percentages,
            n_train_patients, n_val_patients, n_test_patients
        )
        
        print("\n✓ Finetune data splitting complete!")


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune Data Splitter")
    parser.add_argument("--config", type=str, default="config/finetune_split_config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Run splitter
    splitter = FinetuneDataSplitter(config)
    splitter.run()
