#!/usr/bin/env python3
"""
Organize HIV MRS data files from Downloads folder
Moves files from ~/Downloads to organized data/ structure
"""

import shutil
from pathlib import Path

# Source and destination
DOWNLOADS = Path.home() / "Downloads"
DATA_DIR = Path.home() / "Documents" / "Github" / "noise_decorrelation_HIV" / "data"

# File categorization - exactly as they appear in Downloads
FILES_TO_ORGANIZE = {
    "raw": [
        "MRSHIVSuppMatDahmaniRev.xlsx",
        "Table_6.xls",
        "Table3.xls",
        "Table_1_PerinatalHIVInfectionorExposureIsAssociatedWithLowNAcetylaspartateandGlutamateinBasalGangliaatAge9butNot7Years.docx",
        "Table_2_PerinatalHIVInfectionorExposureIsAssociatedWithLowNAcetylaspartateandGlutamateinBasalGangliaatAge9butNot7Years.DOCX",
        "Table_1_Fixel-Based_Analysis_and_Free_Water_Corrected_DTI_Evaluation_of_HIV-Associated_Neurocognitive_Disorders.docx",
        "Table_3_Atypical_Resting-State_Functional_Connectivity_Dynamics_Correlate_With_Early_Cognitive_Dysfunction_in_HIV_Infection.docx",
    ],
    "extracted": [
        "CHANG_2002_EXTRACTED.csv",
        "SAILASUTA_2012_EXTRACTED.csv",
        "SAILASUTA_2012_ACUTE_DATA.csv",
        "SAILASUTA_2016_LONGITUDINAL.csv",
        "YOUNG_2014_CROSS_SECTIONAL_DATA.csv",
        "YOUNG_2014_LONGITUDINAL.csv",
        "CRITICAL_STUDIES_COMPLETE_DATA.csv",
    ],
    "master": [
        "MASTER_HIV_MRS_DATABASE_v2.csv",
        "NAA_DATA_FOR_MODEL.csv",
    ],
    "model_outputs": [
        "posterior_predictive.csv",
        "summary_v4.csv",
    ],
    "individual": [
        "VALCOUR_2015_INDIVIDUAL_PATIENTS.csv",
        "VALCOUR_2015_ACUTE_DATA.csv",
        "VALCOUR_2015_DATA_FOR_MASTER.csv",
        "VALCOUR_2015_COMPLETE_EXTRACTION.md",
        "SAILASUTA_2012_COMPLETE_EXTRACTION.md",
    ],
}


def main():
    print("=" * 60)
    print("Organizing HIV MRS Data from Downloads")
    print("=" * 60)

    # Check if Downloads exists
    if not DOWNLOADS.exists():
        print(f"\n❌ Downloads folder not found: {DOWNLOADS}")
        return

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        print("Run organize_project_data.py first to create the structure!")
        return

    print(f"\nSource: {DOWNLOADS}")
    print(f"Destination: {DATA_DIR}\n")

    moved_count = 0
    missing_count = 0

    for category, files in FILES_TO_ORGANIZE.items():
        dest_dir = DATA_DIR / category
        print(f"\n{category.upper()}:")

        for filename in files:
            src = DOWNLOADS / filename
            dest = dest_dir / filename

            if src.exists():
                # Copy (not move) to be safe - you can delete from Downloads later
                shutil.copy2(src, dest)
                print(f"  ✓ {filename}")
                moved_count += 1
            else:
                print(f"  ✗ {filename} (not found in Downloads)")
                missing_count += 1

    print("\n" + "=" * 60)
    print(f"✓ Organization complete!")
    print(f"  Files copied: {moved_count}")
    print(f"  Files not found: {missing_count}")
    print("=" * 60)

    if missing_count > 0:
        print(f"\n⚠️  {missing_count} files not found in Downloads")
        print("This is OK if you don't have all files yet.")

    print(f"\nFiles were COPIED (not moved) from Downloads")
    print(f"Original files still in: {DOWNLOADS}")
    print(f"\nTo clean up Downloads after verifying:")
    print(f"  cd {DOWNLOADS}")
    print(f"  rm CHANG_2002*.csv SAILASUTA*.csv VALCOUR*.csv etc.")

    print(f"\nNext steps:")
    print(f"1. cd {DATA_DIR.parent}")
    print(f"2. git status  # See what was added")
    print(f"3. git add data/")
    print(f"4. git commit -m 'Add organized extracted datasets'")


if __name__ == "__main__":
    main()