import join_and_generate
import os
import time

if __name__ == "__main__":
    t = time.time()
    cat_numeric_filename = "../test_data/mvp_student_profile_scramble.txt"
    set_valued_files = [] #This will still run even without any such files
    key = "student_id"
    cat_cols = ["ethnicity","gender","isInternCandidate","isNonresidentAlien","isOptIn","major","degree","city","state","university","isVeteranOrMilitary"]
    num_cols = ["gpa","yob","grad_yr","grad_mo","baseScore"]
    output_df = join_and_generate.generate_synthetic_files(cat_numeric_filename, set_valued_files, key, cat_cols, num_cols, 100, 100, 1, delim='\t')
    print("TOTAL TIME: ", time.time()-t)
