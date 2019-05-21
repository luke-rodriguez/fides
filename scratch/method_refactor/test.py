from FidesDataset import FidesDataset
import SyntheticPrivate
import Statistics
import UtilityMetric
import os

if __name__ == "__main__":
    #infile = "data/not_connected_test.csv"
    infile = "data/adult.csv"
    fd = FidesDataset()
    fd.read_categorical_numeric_file(infile)
    cat_cols = ["occupation","workclass","race","sex",
                   "native-country","marital-status","relationship","education","income"]
    fd.set_categorical_columns(cat_cols)
    fd.create_data_to_use()
    dppro_fd = SyntheticPrivate.dppro(fd, 1, 0.01, 30)
    sampling_fd = SyntheticPrivate.sampling_based_inference(fd, 1, 0.2)
    privbayes_fd = SyntheticPrivate.privbayes(fd, 1, 2)
    matrix_cond_fd = SyntheticPrivate.matrix_factorization(fd, 1, 10, 1000, 0.01, normalization="conditional")
    matrix_all_fd = SyntheticPrivate.matrix_factorization(fd, 1, 10, 1000, 0.01, normalization="all")
    matrix_matr_fd = SyntheticPrivate.matrix_factorization(fd, 1, 10, 1000, 0.01, normalization="matrix")
    matrix_none_fd = SyntheticPrivate.matrix_factorization(fd, 1, 10, 1000, 0.01, normalization="none")

    print("PrivBayes")
    print(Statistics.total_mutual_information(privbayes_fd))
    print(UtilityMetric.compare_marginals(fd, privbayes_fd, 2))
    print(UtilityMetric.compare_marginals(fd, privbayes_fd, 3))
    print("DPPro")
    print(Statistics.total_mutual_information(dppro_fd))
    print(UtilityMetric.compare_marginals(fd, dppro_fd, 2))
    print(UtilityMetric.compare_marginals(fd, dppro_fd, 3))
    print("Sampling")
    print(Statistics.total_mutual_information(sampling_fd))
    print(UtilityMetric.compare_marginals(fd, sampling_fd, 2))
    print(UtilityMetric.compare_marginals(fd, sampling_fd, 3))
    print("Matrix cond")
    print(Statistics.total_mutual_information(matrix_cond_fd))
    print(UtilityMetric.compare_marginals(fd, matrix_cond_fd, 2))
    print(UtilityMetric.compare_marginals(fd, matrix_cond_fd, 3))
    print("Matrix all")
    print(Statistics.total_mutual_information(matrix_all_fd))
    print(UtilityMetric.compare_marginals(fd, matrix_all_fd, 2))
    print(UtilityMetric.compare_marginals(fd, matrix_all_fd, 3))
    print("Matrix matr")
    print(Statistics.total_mutual_information(matrix_matr_fd))
    print(UtilityMetric.compare_marginals(fd, matrix_matr_fd, 2))
    print(UtilityMetric.compare_marginals(fd, matrix_matr_fd, 3))
    print("Matrix none")
    print(Statistics.total_mutual_information(matrix_none_fd))
    print(UtilityMetric.compare_marginals(fd, matrix_none_fd, 2))
    print(UtilityMetric.compare_marginals(fd, matrix_none_fd, 3))
    # sampling_fd = SyntheticPrivate.sampling_based_inference(fd, 0.1, 0.2)
    # matrix_fd = SyntheticPrivate.matrix_factorization(fd, 0.1, 10, 1000, 0.01)
    #
    # print("ORIGINAL MI:", Statistics.total_mutual_information(fd))
    # print("Sampling MI:", Statistics.total_mutual_information(sampling_fd))
    # print("Matrix MI:", Statistics.total_mutual_information(matrix_fd))
    #
    # print("Sampling 2-way marginal distance:", UtilityMetric.compare_marginals(fd, sampling_fd, 2))
    # print("Matrix 2-way marginal distance:", UtilityMetric.compare_marginals(fd, matrix_fd, 2))
    # print("Sampling 3-way marginal distance:", UtilityMetric.compare_marginals(fd, sampling_fd, 3))
    # print("Matrix 3-way marginal distance:", UtilityMetric.compare_marginals(fd, matrix_fd, 3))
    #
    # print("Sampling to matrix 2-way:", UtilityMetric.compare_marginals(sampling_fd, matrix_fd, 2))
    # print("Sampling to matrix 3-way:", UtilityMetric.compare_marginals(sampling_fd, matrix_fd, 3))
    #
    # sampling_fd.write("adult_sampling")
    # matrix_fd.write("matrix_sampling")

    #fd = FidesDataset()
    #infile = "data/adult.csv"
    #fd.read_categorical_numeric_file(infile)
    #num_cols = ["age","hours-per-week"]
    #cat_cols = ["occupation","workclass","race","sex",
    #                "native-country","marital-status","relationship","education","income"]
    #fd.set_categorical_columns(cat_cols)
    #fd.set_numeric_columns(num_cols)
    #fd.create_data_to_use()
    # new_fd = SyntheticPrivate.matrix_factorization(fd, 1, 10, 1000, 0.01)
    # test_fd = SyntheticPrivate.matrix_factorization(new_fd, 1, 1000, 1000, 0.01)
    # fd.write("OLD")
    # new_fd.write("NEW")
    # test_fd.write("TEST")

    # train = FidesDataset()
    # train.read_categorical_numeric_file("data/classification/adult_train.csv")
    # test = FidesDataset()
    # test.read_categorical_numeric_file("data/classification/adult_test.csv")
    # num_cols = ["age","hours-per-week"]
    # cat_cols = ["occupation","workclass","race","sex",
    #                 "native-country","marital-status","relationship","education","income"]
    # train.set_categorical_columns(cat_cols)
    # train.set_numeric_columns(num_cols)
    # train.create_data_to_use()
    # test.set_categorical_columns(cat_cols)
    # test.set_numeric_columns(num_cols)
    # test.create_data_to_use()
    # synthetic_test = SyntheticPrivate.matrix_factorization(test, 1, 10, 1000, 0.01)
    # Statistics.svm_accuracy(train,test,"income")
    # Statistics.svm_accuracy(train,synthetic_test,"income")
    # Statistics.plot_data(train, "train")
    # Statistics.plot_data(test, "test")
    # Statistics.plot_data(synthetic_test, "synthetic_test")
    #print(Statistics.total_mutual_information(train))
    #print(Statistics.total_mutual_information(test))
    #print(Statistics.total_mutual_information(synthetic_test))
