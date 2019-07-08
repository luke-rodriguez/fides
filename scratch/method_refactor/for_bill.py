from FidesDataset import FidesDataset
import SyntheticPrivate
import Statistics
import UtilityMetric

######################## PARAMETERS ################################
#Shared Parameters
epsilon = 1
#Privbayes parameters
privbayes_k = 2 #Degree of bayesian network (i.e. every attribute has k parents)
#Sampling-Based Inference Parameters
cramers_v = 0.2 #what value of cramers_v to count as "correlated" (0.2 recommended by paper)
#Matrix Factorization Parameters
matrix_k = 100 #Degree of approximation
num_iterations = 1000 #Maximum number of iterations to run if it does not converge
lambda_ = 0.01 #Learning rate


if __name__ == "__main__":
    infile = "data/adult.csv"
    fd = FidesDataset()
    fd.read_categorical_numeric_file(infile)
    categorical_columns = ["occupation","workclass","race","sex",
                 "native-country","marital-status","relationship","education","income"]
    fd.set_categorical_columns(categorical_columns)
    fd.create_data_to_use()

    privbayes_fd = SyntheticPrivate.privbayes(fd, epsilon, privbayes_k)
    privbayes_fd.write("PRIVBAYES") #Write out results
    matrix_fd = SyntheticPrivate.matrix_factorization(fd, epsilon, matrix_k, num_iterations, lambda_)
    matrix_fd.write("MATRIX") #Write out results
    sampling_fd = SyntheticPrivate.sampling_based_inference(fd, epsilon, 0.2)
    sampling_fd.write("SAMPLING") #Write out results

    # Example metrics
    print("Privbayes Total MI")
    print(Statistics.total_mutual_information(privbayes_fd))
    print("Average marginal distance from original to privbayes (2-way marginals)")
    print(UtilityMetric.compare_marginals(fd, privbayes_fd, 2))
    print("Average marginal distance from sampling to matrix (3-way marginals)")
    print(UtilityMetric.compare_marginals(fd, privbayes_fd, 3))
    # Show and save the MI heatmap between attributes in the original and in the privbayes approx
    Statistics.mutual_information_heatmap(fd, "original_heatmap", show=True)
    Statistics.mutual_information_heatmap(privbayes_fd, "privbayes_heatmap", show=True)

    #We can do calssifier accuracy too, but that doesn't take a full approximated dataset, it needs splits.
    #There are helper methods in Statistics.py that help do this easily, though.
