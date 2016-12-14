package explorekit;

import explorekit.data.Dataset;
import explorekit.data.Loader;
import explorekit.search.FilterWrapperHeuristicSearch;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by giladkatz on 12/13/16.
 */
public class Program {
    public static void main(String[] args) throws Exception {
        Loader loader = new Loader();
        List<String> datasets = new ArrayList<>();

        HashMap<String, Integer> classAttributeIndices = new HashMap<>();
        datasets.add("/global/home/users/giladk/Datasets/heart.arff");
        datasets.add("/global/home/users/giladk/Datasets/cancer.arff");
        datasets.add("/global/home/users/giladk/Datasets/contraceptive.arff");
        datasets.add("/global/home/users/giladk/Datasets/credit.arff");
        datasets.add("/global/home/users/giladk/Datasets/credit-g.arff");
        datasets.add("/global/home/users/giladk/Datasets/diabetes.arff");
        datasets.add("/global/home/users/giladk/Datasets/Diabetic_Retinopathy_Debrecen.arff");
        datasets.add("/global/home/users/giladk/Datasets/horse-colic.arff");
        datasets.add("/global/home/users/giladk/Datasets/Indian_Liver_Patient_Dataset.arff");
        datasets.add("/global/home/users/giladk/Datasets/seismic-bumps.arff");
        datasets.add("/global/home/users/giladk/Datasets/cardiography_new.arff");


        for (int i = 0; i < 1; i++) {
            for (String datasetPath : datasets) {
                BufferedReader reader = new BufferedReader(new FileReader(datasetPath));

                Dataset dataset;
                if (!classAttributeIndices.containsKey(datasetPath)) {
                    dataset = loader.readArff(reader, i, null, -1, 0.66);
                } else {
                    dataset = loader.readArff(reader, i, null, classAttributeIndices.get(datasetPath), 0.66);
                }

                FilterWrapperHeuristicSearch exp = new FilterWrapperHeuristicSearch(15);
                exp.run(dataset, "_" + Integer.toString(i));
            }
        }
    }
}
