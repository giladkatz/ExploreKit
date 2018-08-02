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
        datasets.add("datasets/ionosphere.arff");
        datasets.add("datasets/winequality-white.arff");
        datasets.add("datasets/winequality-white-small.arff");



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
